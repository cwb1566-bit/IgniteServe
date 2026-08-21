# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.HotspotAnalyzer import HotspotAnalyzer
from vllm.v1.worker import kv_profile_cache
from vllm.v1.worker.startup_compile import (
    compile_only_forward, maybe_apply_offline_best_config, startup_compile_mode)
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)


def _start_full_fx_compile_worker_prewarm(vllm_config: VllmConfig) -> None:
    if os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() not in {
            "1", "true", "yes", "on"}:
        return

    compilation_config = vllm_config.compilation_config
    max_workers = getattr(compilation_config, "max_compile_workers", None)
    max_workers = max_workers or min(os.cpu_count() or 1, 4)
    os.environ.setdefault("TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS",
                          str(max_workers))
    os.environ.setdefault("TORCHINDUCTOR_FX_COMPILE_INNER_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA", "1")
    worker_imports = os.environ.get(
        "TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS", "")
    import_names = [name for name in worker_imports.split(",") if name]
    for module_name in ("vllm._custom_ops", "vllm.compilation.pass_manager"):
        if module_name not in import_names:
            import_names.append(module_name)
    os.environ["TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS"] = ",".join(
        import_names)

    try:
        from torch._inductor.compile_fx_full_subproc import FullFxCompilePool

        FullFxCompilePool.prewarm_workers()
        logger.info(
            "Started Full-FX compiler worker import/CUDA prewarm during "
            "GPU worker initialization.")
    except Exception as exc:
        logger.warning("Failed to start Full-FX compiler worker prewarm: %s",
                       exc)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

#增
import time
import concurrent.futures
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, get_world_group, graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model)
from vllm.utils import DeviceMemoryProfiler
from torch import inference_mode, no_grad
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)
from vllm.v1.phase_profiler import timer

class Worker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        """Initialize all device-side worker stages in dependency order."""
        self.initialize_cuda_device()
        self.initialize_distributed_environment()
        self.initialize_compile_worker_prewarm()
        self.initialize_model_runner()
        self.initialize_usage_reporting()

    def initialize_cuda_device(self) -> None:
        """Select CUDA, validate capacity, and take the initial snapshot."""
        if getattr(self, "_startup_cuda_device_ready", False):
            return
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        self._startup_cuda_device_ready = True

    def initialize_distributed_environment(self) -> None:
        """Create distributed/model-parallel state after CUDA selection."""
        if getattr(self, "_startup_distributed_ready", False):
            return
        self.initialize_cuda_device()
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        set_random_seed(self.model_config.seed)
        self._startup_distributed_ready = True

    def initialize_compile_worker_prewarm(self) -> None:
        """Start the reusable FX compilation worker outside model loading."""
        if getattr(self, "_startup_compile_prewarm_ready", False):
            return
        self.initialize_distributed_environment()
        _start_full_fx_compile_worker_prewarm(self.vllm_config)
        self._startup_compile_prewarm_ready = True

    def initialize_model_runner(self) -> None:
        """Construct GPUModelRunner and its persistent runtime buffers."""
        if getattr(self, "model_runner", None) is not None:
            return
        self.initialize_distributed_environment()
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device)

    def initialize_usage_reporting(self) -> None:
        if getattr(self, "_startup_usage_reporting_ready", False):
            return
        self.initialize_model_runner()
        if self.rank == 0:
            report_usage_stats(self.vllm_config)
        self._startup_usage_reporting_ready = True

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.load_model()

    def init_model(self) -> None:
        self.model_runner.init_model()

    def load_weights(self) -> None:
        self.model_runner.load_weights()

    def get_startup_state_snapshot(self) -> dict[str, object]:
        """Return exact worker-local facts used to build the startup DAG."""
        static_state = self.get_static_startup_resource_snapshot()
        cached_kv_bytes = os.environ.get(
            "VLLM_CACHED_AVAILABLE_KV_CACHE_BYTES")
        kv_value = None
        kv_key = "explicit"
        if cached_kv_bytes:
            try:
                kv_value = int(cached_kv_bytes)
            except ValueError:
                kv_value = None
        elif kv_profile_cache.enabled():
            kv_value, kv_key, _, _ = kv_profile_cache.lookup(
                self.vllm_config, self.device)
        return {
            **static_state,
            "rank": self.rank,
            "device": str(self.device),
            "kv_profile_hit": kv_value is not None and kv_value > 0,
            "kv_profile_key": kv_key,
        }

    def get_static_startup_resource_snapshot(self) -> dict[str, object]:
        """Probe filesystem-only resources before CUDA initialization ends."""
        from vllm.compilation.startup_resources import resource_dir

        def has_entries(path: Path) -> bool:
            try:
                return path.exists() and any(path.iterdir())
            except OSError:
                return False

        compile_root = Path(os.environ.get(
            "VLLM_CACHE_ROOT",
            str(Path.home() / ".cache" / "vllm")))
        best_config_path = resource_dir("best_config")
        transformed_path = resource_dir("transformed_code")
        metadata_path = resource_dir("parallel_compile_metadata")
        hotspot_path = Path(os.environ.get(
            "VLLM_CUDA_GRAPH_HOTSPOT_DIR",
            str(resource_dir("cuda_graph_hotspots"))))
        return {
            "rank": self.rank,
            "compile_cache_present": has_entries(compile_root),
            "best_config_store_present": has_entries(best_config_path),
            "transformed_code_store_present": has_entries(transformed_path),
            "metadata_store_present": has_entries(metadata_path),
            "hotspot_profile_present": has_entries(hotspot_path),
        }

    @torch.inference_mode()
    def run_to_trigger_compile(self) -> None:
        mode = startup_compile_mode()
        if mode in {"s0", "stage_separated", "stage_separated_baseline"}:
            # S0: compile-only and defer the real max-batch execution to
            # determine_available_memory(). No semantic caches or structural
            # reuse are enabled unless explicitly overridden.
            compile_only_forward(self.model_runner, autotune=False)
            return
        if mode in {"native", "baseline", "precompile_autotune",
                    "precompile_autotune_skip_profile"}:
            # Baseline split path: compile and run Triton autotune here, but do
            # not execute the max-batch dummy forward again. KV Cache Init will
            # perform the real profile/warmup when profile caching is disabled.
            compile_only_forward(self.model_runner, autotune=True)
            return
        if mode in {"precompile", "precompile_skip_profile"}:
            compile_only_forward(self.model_runner, autotune=False)
            return
        if mode in {"s4", "a100_optimized", "offline_best_config",
                    "offline_best_config_skip_profile"}:
            # S4: search the persisted best-config store (default
            # vllm_startup_resources/best_config) on every cold start. Load the
            # cached config on a hit; otherwise autotune now and persist the
            # winner back into the store. VLLM_OFFLINE_BEST_CONFIG_ROOT remains
            # an optional extra source but is no longer required.
            offline_root = os.environ.get("VLLM_OFFLINE_BEST_CONFIG_ROOT")
            timer.start("S4 Best-Config Resolve")
            compile_only_forward(
                self.model_runner,
                resolve_best_config=True,
                device=self.model_runner.device,
                offline_root=offline_root,
                persist_best_config=True,
            )
            timer.end("S4 Best-Config Resolve")
            return
        logger.warning(
            "Unknown VLLM_STARTUP_COMPILE_MODE=%s; falling back to baseline "
            "compile-only+autotune path.", mode)
        compile_only_forward(self.model_runner, autotune=True)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        if os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
                "1", "true", "yes", "on"}:
            from torch._inductor.compile_fx_full_subproc import FullFxCompilePool
            shutdown_wait_started = time.perf_counter()
            shutdown_elapsed_s = FullFxCompilePool.wait_for_shutdown_pool()
            logger.info(
                "Parallel compile timing deferred_shutdown_wait_s=%.6f "
                "shutdown_total_s=%.6f",
                time.perf_counter() - shutdown_wait_started,
                shutdown_elapsed_s,
            )

        GiB = lambda b: b / GiB_bytes

        mode = startup_compile_mode()
        cached_kv_bytes = os.environ.get("VLLM_CACHED_AVAILABLE_KV_CACHE_BYTES")
        if cached_kv_bytes:
            available_kv_cache_memory = int(cached_kv_bytes)
            logger.info(
                "Skipping profile_run and using explicit KV cache memory: "
                "%.2f GiB", GiB(available_kv_cache_memory))
            return available_kv_cache_memory

        use_persisted_profile_cache = kv_profile_cache.enabled()
        profile_cache_requested = (
            mode in {"s4", "a100_optimized", "precompile_skip_profile",
                     "precompile_autotune_skip_profile",
                     "offline_best_config_skip_profile"}
            or os.environ.get("VLLM_SKIP_PROFILE_RUN", "").lower()
            in {"1", "true", "yes", "on"}
            or use_persisted_profile_cache
        )
        if use_persisted_profile_cache:
            cached_value, cache_key, _, cache_entry = kv_profile_cache.lookup(
                self.vllm_config, self.device)
            if cached_value is not None:
                logger.info(
                    "Skipping profile_run using persisted KV profile cache "
                    "key=%s value=%.2f GiB updated_at=%s",
                    cache_key[:16], GiB(cached_value),
                    cache_entry.get("updated_at") if cache_entry else None)
                return cached_value

            logger.info(
                "KV profile cache miss; running real profile_run() and "
                "persisting the result afterwards.")
        elif profile_cache_requested:
            logger.warning(
                "Profile skip was requested, but persisted KV profile cache "
                "is disabled and VLLM_CACHED_AVAILABLE_KV_CACHE_BYTES is unset; "
                "falling back to real profile_run().")

        # Cache hits return above without perturbing the CUDA allocator. Only
        # the real profiling path needs a clean peak-memory accounting window.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(
                    self.model_runner.model_memory_usage)) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container.")
        available_kv_cache_memory = self.requested_memory \
            - profile_result.non_kv_cache_memory

        logger.debug(
            "Initial free memory: %.2f GiB, free memory: %.2f GiB, "
            "requested GPU memory: %.2f GiB",
            GiB(self.init_snapshot.free_memory), GiB(free_gpu_memory),
            GiB(self.requested_memory))
        logger.debug(profile_result)
        logger.info("Available KV cache memory: %.2f GiB",
                    GiB(available_kv_cache_memory))
        if profile_cache_requested:
            cache_key, cache_path = kv_profile_cache.save(
                self.vllm_config,
                self.device,
                int(available_kv_cache_memory),
                profile_result=profile_result)
            logger.info(
                "Persisted KV profile cache key=%s path=%s value=%.2f GiB",
                cache_key[:16], cache_path, GiB(available_kv_cache_memory))
        gc.collect()

        return int(available_kv_cache_memory)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True)
        if not self.model_config.enforce_eager:
            self.model_runner.pre_capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = \
                self.model_runner._dummy_run(
                    num_tokens=max_num_reqs,
                    skip_eplb=True,
                )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(
                    hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None
        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def capture_pending_cudagraphs(
        self,
        max_pieces_per_shape: Optional[int] = None,
        deadline_ms: Optional[float] = None,
        selected_shape: Optional[int] = None,
    ) -> int:
        if not self.model_runner.enable_adaptive_cudagraph:
            return 0
        controller = self.model_runner.adaptive_cudagraph_controller
        distributed = self.vllm_config.parallel_config.world_size > 1
        if distributed:
            torch.distributed.barrier(group=get_world_group().cpu_group)
        try:
            captured = controller.capture_pending(
                self.model_runner,
                max_pieces_per_shape=max_pieces_per_shape,
                deadline_ms=deadline_ms,
                selected_shape=selected_shape,
            )
        finally:
            if distributed:
                torch.distributed.barrier(group=get_world_group().cpu_group)
        controller.flush_gpu_samples()
        if captured == 0:
            controller.export_telemetry()
        return captured

    def get_pending_cudagraph_shapes(self) -> list[int]:
        if not self.model_runner.enable_adaptive_cudagraph:
            return []
        return self.model_runner.adaptive_cudagraph_controller.pending_shapes()

    def plan_pending_cudagraph_capture(
        self,
        slack_ms: float,
        max_pieces: int,
        default_fixed_ms: float,
        default_piece_ms: float,
        safety_factor: float,
    ) -> Optional[dict[str, object]]:
        if not self.model_runner.enable_adaptive_cudagraph:
            return None
        controller = self.model_runner.adaptive_cudagraph_controller
        plan = controller.plan_capture_maintenance(
            slack_ms=slack_ms,
            max_pieces=max_pieces,
            default_fixed_ms=default_fixed_ms,
            default_piece_ms=default_piece_ms,
            safety_factor=safety_factor,
        )
        return None if plan is None else dict(vars(plan))

    def export_adaptive_cudagraph_telemetry(self) -> None:
        if not self.model_runner.enable_adaptive_cudagraph:
            return
        controller = self.model_runner.adaptive_cudagraph_controller
        controller.flush_gpu_samples()
        controller.export_telemetry()

    def export_cudagraph_hotspot_profile(self) -> None:
        """Persist one logical workload profile from distributed rank zero."""
        if self.rank == 0:
            HotspotAnalyzer.check_and_export(True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")
