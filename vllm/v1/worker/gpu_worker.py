# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import concurrent.futures
import gc
import hashlib
import json
import os
import time
import types
import weakref
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
from vllm.v1.worker.LazyFreezeManager import LazyFreezeManager
from vllm.v1.worker import kv_profile_cache
from vllm.v1.worker.startup_compile import (
    compile_only_forward, maybe_apply_offline_best_config, startup_compile_mode)
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.engine.weight_pagecache import apply_rebind_runtime_env

logger = init_logger(__name__)


def _release_model_compilation_state(compilation_config: object) -> dict[str,
                                                                       object]:
    """Release model-owned piecewise callables and CUDA graph allocations."""
    started = time.perf_counter()
    backends = getattr(compilation_config, "cuda_piecewise_backends_list", {})
    released_graphs = 0
    released_backends = 0
    errors = []
    for backend in list(getattr(backends, "values", lambda: ())()):
        released_backends += 1
        entries = getattr(backend, "concrete_size_entries", {})
        for entry in list(getattr(entries, "values", lambda: ())()):
            graph = getattr(entry, "cudagraph", None)
            if graph is not None:
                try:
                    graph.reset()
                    released_graphs += 1
                except Exception as exc:
                    errors.append(f"cudagraph.reset:{exc!r}")
            entry.cudagraph = None
            entry.runnable = None
            entry.output = None
            entry.input_addresses = None
            entry.concurrent_capture_in_progress = False
            entry.concurrent_capture_gate = None
            entry.concurrent_capture_pool = None
            entry.concurrent_capture_stream = None
        getattr(entries, "clear", lambda: None)()
        # Break cycles rooted in CompilationConfig. These objects own the
        # compiled closures whose cells capture model parameters and KV cache
        # buffers even after GPUModelRunner itself has been collected.
        for name in ("compiled_graph_for_general_shape", "graph",
                     "vllm_backend", "vllm_config", "compilation_config"):
            if hasattr(backend, name):
                setattr(backend, name, None)
    getattr(backends, "clear", lambda: None)()
    getattr(compilation_config, "static_forward_context", {}).clear()

    # A graph-pool token is cheap to recreate and belongs to the model's set
    # of captured graphs. Do not carry the old token across a general rebind.
    try:
        import vllm.compilation.backends as backend_module
        backend_module.global_graph_pool = None
    except Exception as exc:
        errors.append(f"global_graph_pool:{exc!r}")
    return {
        "elapsed_s": time.perf_counter() - started,
        "released_backends": released_backends,
        "released_graphs": released_graphs,
        "errors": errors,
    }


def _release_model_module_state(model: nn.Module) -> dict[str, object]:
    """Break vLLM's parameter-loader cycles before discarding a model."""
    started = time.perf_counter()
    modules = list(model.modules())
    cleared_parameter_methods = 0
    cleared_parameters = 0
    cleared_kv_bindings = 0
    errors = []
    for module in modules:
        kv_cache = getattr(module, "kv_cache", None)
        if isinstance(kv_cache, list):
            cleared_kv_bindings += len(kv_cache)
            kv_cache.clear()
        for parameter in module._parameters.values():
            if parameter is None:
                continue
            for name, value in list(vars(parameter).items()):
                if not isinstance(value, types.MethodType):
                    continue
                try:
                    delattr(parameter, name)
                    cleared_parameter_methods += 1
                except Exception as exc:
                    errors.append(f"parameter.{name}:{exc!r}")

        # Compiled callables can retain leaf modules beyond the lifetime of
        # the top-level model. Detach old weight ownership explicitly while
        # leaving buffers intact: globally cached rotary modules legitimately
        # reuse their cos/sin buffers in a later model instance.
        cleared_parameters += len(module._parameters)
        module._parameters.clear()

    # Do not clear arbitrary module parameters or buffers here. vLLM caches
    # some small stateless modules (notably rotary embeddings) globally and a
    # later model instance can legitimately reuse them. Breaking the loader
    # cycles and model-specific KV bindings is sufficient for the old model
    # ownership tree to become collectible.
    return {
        "elapsed_s": time.perf_counter() - started,
        "modules": len(modules),
        "cleared_parameter_methods": cleared_parameter_methods,
        "cleared_parameters": cleared_parameters,
        "cleared_kv_bindings": cleared_kv_bindings,
        "errors": errors,
    }


def _clear_model_compile_runtime_caches() -> dict[str, object]:
    """Drop process-local compiled callables without touching disk caches."""
    started = time.perf_counter()
    cleared = []
    errors = []
    try:
        torch._dynamo.reset()
        cleared.append("torch._dynamo")
    except Exception as exc:
        errors.append(f"torch._dynamo:{exc!r}")
    try:
        from torch._inductor import codecache
        for name in ("PyCodeCache", "CppCodeCache", "CUDACodeCache"):
            cache = getattr(codecache, name, None)
            clear = getattr(cache, "cache_clear", None)
            if clear is not None:
                clear()
                cleared.append(name)
    except Exception as exc:
        errors.append(f"torch._inductor.codecache:{exc!r}")
    if os.environ.get(
            "VLLM_ENABLE_SUBPROCESS_COMPILE", "").strip().lower() in {
                "1", "true", "yes", "on"
            }:
        try:
            from torch._inductor.compile_fx_full_subproc import (
                FullFxCompilePool,
            )
            worker_results = FullFxCompilePool.reset_model_caches()
            cleared.extend(
                f"FullFxCompilePool:{result.worker_pid}"
                for result in worker_results)
        except Exception as exc:
            errors.append(f"FullFxCompilePool:{exc!r}")
    return {
        "elapsed_s": time.perf_counter() - started,
        "cleared": cleared,
        "errors": errors,
    }


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
    two_phase = os.environ.get(
        "VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM", "0") == "1"
    if two_phase:
        os.environ["TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA"] = "0"
    else:
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
        from vllm.compilation.compiler_interface import (
            start_inductor_factor_prewarm)
        from torch._inductor.compile_fx_full_subproc import FullFxCompilePool

        prewarm_identity = os.environ.get(
                "VLLM_ENABLE_ASYNC_COMPILER_HASH_PREWARM",
                "").strip().lower() in {"1", "true", "yes", "on"}
        if prewarm_identity:
            start_inductor_factor_prewarm()
        FullFxCompilePool.prewarm_workers()
        logger.info(
            "Started %sFull-FX worker prewarm during GPU worker "
            "initialization.",
            "compiler-identity and " if prewarm_identity else "")
    except Exception as exc:
        logger.warning("Failed to start Full-FX compiler worker prewarm: %s",
                       exc)


def _start_full_fx_compile_worker_cuda_prewarm(device_index: int) -> None:
    if os.environ.get(
            "VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM", "0") != "1":
        return
    try:
        from torch._inductor.compile_fx_full_subproc import FullFxCompilePool

        FullFxCompilePool.prewarm_cuda_workers(device_index)
        logger.info("Queued second-stage Full-FX CUDA prewarm on device %s.",
                    device_index)
    except Exception as exc:
        logger.warning("Failed to queue Full-FX CUDA prewarm: %s", exc)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

from vllm.distributed.parallel_state import (
    get_world_group, graph_capture,
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
        _start_full_fx_compile_worker_cuda_prewarm(self.local_rank)

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
        two_phase = os.environ.get(
            "VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM", "0") == "1"
        if not two_phase:
            self.initialize_distributed_environment()
        _start_full_fx_compile_worker_prewarm(self.vllm_config)
        if two_phase and getattr(self, "_startup_cuda_device_ready", False):
            _start_full_fx_compile_worker_cuda_prewarm(self.local_rank)
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

    @staticmethod
    def _shell_topology_signature(vllm_config: VllmConfig) -> tuple:
        parallel = vllm_config.parallel_config
        return (
            parallel.world_size,
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
            str(parallel.distributed_executor_backend),
            str(parallel.worker_cls),
        )

    @staticmethod
    def _inplace_model_signature(vllm_config: VllmConfig) -> str:
        """Hash shape- and runtime-defining configuration, excluding paths."""
        model = vllm_config.model_config
        hf_config = model.hf_config.to_dict()
        for key in ("_name_or_path", "name_or_path", "transformers_version"):
            hf_config.pop(key, None)
        payload = {
            "hf_config": hf_config,
            "dtype": str(model.dtype),
            "quantization": model.quantization,
            "task": str(model.task),
            "max_model_len": model.max_model_len,
            "is_multimodal": model.is_multimodal_model,
            "pooler_config": str(model.pooler_config),
            "topology": Worker._shell_topology_signature(vllm_config),
            "cache_dtype": str(vllm_config.cache_config.cache_dtype),
            "block_size": vllm_config.cache_config.block_size,
            "max_num_batched_tokens": (
                vllm_config.scheduler_config.max_num_batched_tokens),
            "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
            "compilation_level": str(vllm_config.compilation_config.level),
            "cudagraph_sizes": list(
                vllm_config.compilation_config.cudagraph_capture_sizes),
            "full_cuda_graph": (
                vllm_config.compilation_config.full_cuda_graph),
            "speculative": str(vllm_config.speculative_config),
            "lora": str(vllm_config.lora_config),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def can_rebind_weights_in_place(
            self, vllm_config: VllmConfig) -> dict[str, object]:
        old_digest = self._inplace_model_signature(self.vllm_config)
        new_digest = self._inplace_model_signature(vllm_config)
        reasons = []
        if old_digest != new_digest:
            reasons.append("structure_or_runtime_signature_mismatch")
        if self.model_config.quantization is not None:
            reasons.append("quantized_weight_postprocessing_is_not_stable")
        if self.lora_config is not None or vllm_config.lora_config is not None:
            reasons.append("lora_state_is_not_supported")
        if self.speculative_config is not None:
            reasons.append("speculative_model_state_is_not_supported")
        runner = getattr(self, "model_runner", None)
        if runner is None or runner.model is None:
            reasons.append("model_is_not_initialized")
        return {
            "rank": self.rank,
            "eligible": not reasons,
            "reasons": reasons,
            "stale_worker_requests": (
                len(runner.requests) if runner is not None else 0),
            "stale_input_batch_requests": (
                len(runner.input_batch.req_ids) if runner is not None else 0),
            "old_signature": old_digest,
            "new_signature": new_digest,
        }

    def rebind_weights_in_place(
            self, vllm_config: VllmConfig) -> dict[str, object]:
        """Overwrite a compatible model while retaining KV/graph storage."""
        eligibility = self.can_rebind_weights_in_place(vllm_config)
        if not eligibility["eligible"]:
            raise ValueError(
                "Unsafe in-place model rebind: "
                f"rank={self.rank} reasons={eligibility['reasons']}")

        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        runner = self.model_runner
        for req_id in list(runner.input_batch.req_ids):
            if req_id is not None:
                runner.input_batch.remove_request(req_id)
        runner.input_batch.condense()
        if runner.input_batch.req_ids:
            raise RuntimeError("Failed to clear worker input batch for rebind")
        runner.encoder_cache.clear()
        runner.requests.clear()
        runner.cuda_graph_manager.before_rebind()
        load_result = runner.rebind_weights_in_place(vllm_config)
        runner.cuda_graph_manager.after_rebind()
        vllm_config.parallel_config.rank = self.rank
        self.bind_vllm_config(vllm_config)
        result = {
            **eligibility,
            **load_result,
            "elapsed_s": time.perf_counter() - started,
            "kv_cache_tensors_retained": len(runner.kv_caches),
            "cuda_graph_shapes_retained": len(
                runner.cuda_graph_manager.registry.published if runner is not None else ()),
        }
        logger.info("In-place weight rebind finished: %s", result)
        return result

    def rebind_model(
        self,
        vllm_config: VllmConfig,
        runtime_env: Optional[dict[str, str]] = None,
    ) -> dict[str, object]:
        """Drop model-owned state while preserving CUDA/distributed runtime."""
        if os.environ.get(
                "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0").lower() not in {
                    "1", "true", "yes", "on"
                }:
            raise RuntimeError(
                "Persistent engine shell rebind is disabled. Set "
                "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL=1 to enable it.")
        old_signature = self._shell_topology_signature(self.vllm_config)
        new_signature = self._shell_topology_signature(vllm_config)
        if old_signature != new_signature:
            raise ValueError(
                "Persistent shell topology mismatch: "
                f"old={old_signature!r} new={new_signature!r}")

        started = time.perf_counter()
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        free_before, total_memory = torch.cuda.mem_get_info(self.device)

        runner = getattr(self, "model_runner", None)
        runner_ref = weakref.ref(runner) if runner is not None else None
        model_ref = (weakref.ref(runner.model)
                     if runner is not None and runner.model is not None
                     else None)
        old_runtime = {
            "worker_pid": os.getpid(),
            "runner_id": id(runner) if runner is not None else None,
            "model_id": (id(runner.model) if runner is not None
                         and runner.model is not None else None),
            "kv_cache_tensors": len(getattr(runner, "kv_caches", ())),
            "kv_cache_ptrs": [
                tensor.data_ptr()
                for cache in getattr(runner, "kv_caches", ())
                for tensor in (cache if isinstance(cache, (list, tuple))
                               else (cache, ))
                if isinstance(tensor, torch.Tensor)
            ],
            "cuda_graph_shapes": list(
                runner.cuda_graph_manager.registry.published if runner is not None else ()),
        }
        gc_was_unfrozen = False
        compilation_state_release = {
            "elapsed_s": 0.0,
            "released_backends": 0,
            "released_graphs": 0,
            "errors": [],
        }
        module_state_release = {
            "elapsed_s": 0.0,
            "modules": 0,
            "cleared_parameter_methods": 0,
            "cleared_parameters": 0,
            "cleared_kv_bindings": 0,
            "errors": [],
        }
        if runner is not None:
            runner.cuda_graph_manager.shutdown()
            gc_was_unfrozen = (
                LazyFreezeManager.get_instance().release_for_model_rebind())
            if runtime_env is not None:
                apply_rebind_runtime_env(runtime_env)
            getattr(runner, "kv_caches", []).clear()
            getattr(runner, "encoder_cache", {}).clear()
            getattr(runner, "requests", {}).clear()
            old_compilation_config = getattr(runner, "compilation_config", None)
            if old_compilation_config is not None:
                compilation_state_release = _release_model_compilation_state(
                    old_compilation_config)
            if runner.model is not None:
                module_state_release = _release_model_module_state(runner.model)
            runner.model = None
            self.model_runner = None
            del runner

        self._sleep_saved_buffers = {}
        gc.collect()
        free_after_runner_gc, _ = torch.cuda.mem_get_info(self.device)
        compile_cache_reset = {"elapsed_s": 0.0, "cleared": [], "errors": []}
        if os.environ.get(
                "VLLM_PERSISTENT_SHELL_CLEAR_INMEM_COMPILE",
                "1").strip().lower() in {"1", "true", "yes", "on"}:
            compile_cache_reset = _clear_model_compile_runtime_caches()
        # Dynamo/Inductor cache reset releases the last Python closures that
        # can own compiled model and CUDA graph objects. Collect once more
        # before asking the caching allocator to return unused blocks.
        gc.collect()
        free_after_compile_gc, _ = torch.cuda.mem_get_info(self.device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize(self.device)
        free_after_release, _ = torch.cuda.mem_get_info(self.device)
        allocated_after_release = torch.cuda.memory_allocated(self.device)
        reserved_after_release = torch.cuda.memory_reserved(self.device)
        logger.info(
            "Persistent shell release memory rank=%d: free_before=%.2f GiB "
            "free_after_runner_gc=%.2f GiB free_after_compile_gc=%.2f GiB "
            "free_after_empty_cache=%.2f GiB allocated=%.2f GiB "
            "reserved=%.2f GiB",
            self.rank, free_before / GiB_bytes,
            free_after_runner_gc / GiB_bytes,
            free_after_compile_gc / GiB_bytes,
            free_after_release / GiB_bytes,
            allocated_after_release / GiB_bytes,
            reserved_after_release / GiB_bytes)
        logger.info("Persistent shell compilation-state release rank=%d: %s",
                    self.rank, compilation_state_release)
        logger.info("Persistent shell module-state release rank=%d: %s",
                    self.rank, module_state_release)
        if os.environ.get(
                "VLLM_PERSISTENT_SHELL_DIAG_REFS", "0").lower() in {
                    "1", "true", "yes", "on"
                }:
            runner_alive = runner_ref() if runner_ref is not None else None
            model_alive = model_ref() if model_ref is not None else None
            logger.info(
                "Persistent shell weakrefs rank=%d runner_alive=%s "
                "model_alive=%s", self.rank, runner_alive is not None,
                model_alive is not None)
            for name, value in (("runner", runner_alive),
                                ("model", model_alive)):
                if value is None:
                    continue
                referrers = []
                for referrer in gc.get_referrers(value)[:20]:
                    if isinstance(referrer, dict):
                        keys = [str(key)[:80] for key, item in referrer.items()
                                if item is value][:8]
                        referrers.append(f"dict(keys={keys})")
                    else:
                        referrers.append(type(referrer).__qualname__)
                logger.info("Persistent shell %s referrers: %s", name,
                            referrers)
            seen_ptrs = set()
            cuda_tensors = []
            for value in gc.get_objects():
                try:
                    if not isinstance(value, torch.Tensor) or not value.is_cuda:
                        continue
                    ptr = value.untyped_storage().data_ptr()
                    if ptr in seen_ptrs:
                        continue
                    seen_ptrs.add(ptr)
                    cuda_tensors.append((
                        value.untyped_storage().nbytes(), value))
                except Exception:
                    continue
            cuda_tensors.sort(key=lambda item: item[0], reverse=True)
            closure_owners: dict[int, list[str]] = {}
            for value in gc.get_objects():
                if not isinstance(value, types.FunctionType):
                    continue
                for cell in value.__closure__ or ():
                    closure_owners.setdefault(id(cell), []).append(
                        f"{value.__module__}.{value.__qualname__}@"
                        f"{value.__code__.co_filename}")

            def find_cell_owners(cell: object, depth: int = 0,
                                 visited: Optional[set[int]] = None) -> list[str]:
                visited = set() if visited is None else visited
                if id(cell) in visited or depth > 8:
                    return []
                visited.add(id(cell))
                owners = list(closure_owners.get(id(cell), ()))
                if owners:
                    return owners
                for parent in gc.get_referrers(cell):
                    if isinstance(parent, types.CellType):
                        owners.extend(
                            find_cell_owners(parent, depth + 1, visited))
                return owners
            logger.info(
                "Persistent shell live CUDA storages rank=%d count=%d "
                "total=%.2f GiB", self.rank, len(cuda_tensors),
                sum(item[0] for item in cuda_tensors) / GiB_bytes)
            live_modules = []
            module_type_counts: dict[str, int] = {}
            for value in gc.get_objects():
                if not isinstance(value, nn.Module):
                    continue
                own_cuda_bytes = 0
                for parameter in value._parameters.values():
                    if isinstance(parameter, torch.Tensor) and parameter.is_cuda:
                        own_cuda_bytes += parameter.untyped_storage().nbytes()
                for buffer in value._buffers.values():
                    if isinstance(buffer, torch.Tensor) and buffer.is_cuda:
                        own_cuda_bytes += buffer.untyped_storage().nbytes()
                if not own_cuda_bytes:
                    continue
                type_name = type(value).__module__ + "." + type(
                    value).__qualname__
                module_type_counts[type_name] = module_type_counts.get(
                    type_name, 0) + 1
                live_modules.append((own_cuda_bytes, value))
            live_modules.sort(key=lambda item: item[0], reverse=True)
            logger.info(
                "Persistent shell live CUDA modules rank=%d count=%d "
                "types=%s", self.rank, len(live_modules),
                sorted(module_type_counts.items(),
                       key=lambda item: item[1], reverse=True)[:16])
            for own_cuda_bytes, module in live_modules[:8]:
                module_refs = []
                for referrer in gc.get_referrers(module):
                    if referrer is live_modules:
                        continue
                    if isinstance(referrer, dict):
                        keys = [str(key)[:100] for key, item in referrer.items()
                                if item is module][:8]
                        module_refs.append(f"dict(keys={keys})")
                    elif isinstance(referrer, types.MethodType):
                        method_parents = []
                        for parent in gc.get_referrers(referrer):
                            if isinstance(parent, dict):
                                keys = [str(key)[:80] for key, item in
                                        parent.items()
                                        if item is referrer][:6]
                                method_parents.append(f"dict(keys={keys})")
                            else:
                                method_parents.append(
                                    type(parent).__module__ + "." +
                                    type(parent).__qualname__)
                        module_refs.append(
                            "method=" + referrer.__func__.__module__ + "." +
                            referrer.__func__.__qualname__ + " parents=" +
                            repr(method_parents[:8]))
                    elif isinstance(referrer, types.CellType):
                        owners = find_cell_owners(referrer)
                        module_refs.append("cell owners=" + repr(owners[:6]))
                    else:
                        module_refs.append(type(referrer).__module__ + "." +
                                           type(referrer).__qualname__)
                logger.info(
                    "Persistent shell CUDA module %.3f GiB type=%s refs=%s",
                    own_cuda_bytes / GiB_bytes,
                    type(module).__module__ + "." + type(module).__qualname__,
                    module_refs[:12])
            for nbytes, tensor in cuda_tensors[:16]:
                refs = []
                for referrer in gc.get_referrers(tensor):
                    if referrer is cuda_tensors:
                        continue
                    if isinstance(referrer, dict):
                        keys = [str(key)[:100]
                                for key, item in referrer.items()
                                if item is tensor][:8]
                        refs.append(f"dict(keys={keys})")
                    else:
                        refs.append(type(referrer).__qualname__)
                        if isinstance(referrer, types.CellType):
                            recursive_owners = find_cell_owners(referrer)
                            refs.extend("function=" + owner
                                        for owner in recursive_owners[:4])
                            if id(referrer) not in closure_owners:
                                parent_types = []
                                for parent in gc.get_referrers(referrer):
                                    if isinstance(parent, dict):
                                        keys = [str(key)[:80] for key, item in
                                                parent.items()
                                                if item is referrer][:6]
                                        parent_types.append(
                                            f"dict(keys={keys})")
                                    else:
                                        parent_types.append(
                                            type(parent).__qualname__)
                                refs.append("cell_parents=" +
                                            repr(parent_types[:8]))
                    if len(refs) >= 8:
                        break
                logger.info(
                    "Persistent shell CUDA storage %.3f GiB tensor=%s "
                    "shape=%s refs=%s", nbytes / GiB_bytes,
                    type(tensor).__qualname__, tuple(tensor.shape), refs)

        vllm_config.parallel_config.rank = self.rank
        self.bind_vllm_config(vllm_config)
        self.init_snapshot = MemorySnapshot()
        self.requested_memory = (
            self.init_snapshot.total_memory
            * self.cache_config.gpu_memory_utilization)
        if self.init_snapshot.free_memory < self.requested_memory:
            raise ValueError(
                "Insufficient free GPU memory after persistent-shell reset: "
                f"free={self.init_snapshot.free_memory} "
                f"requested={self.requested_memory}")

        self.model_runner = GPUModelRunner(vllm_config, self.device)
        self._persistent_rebind_epoch = getattr(
            self, "_persistent_rebind_epoch", 0) + 1
        self.model_runner._persistent_rebind_epoch = (
            self._persistent_rebind_epoch)
        elapsed = time.perf_counter() - started
        released = max(0, free_after_release - free_before)
        logger.info(
            "Persistent engine shell rebound rank=%d in %.6f s; "
            "released=%.2f GiB retained_runtime=%.2f GiB",
            self.rank, elapsed, released / GiB_bytes,
            (total_memory - free_after_release) / GiB_bytes)
        return {
            "rank": self.rank,
            "worker_pid": os.getpid(),
            "elapsed_s": elapsed,
            "released_bytes": released,
            "retained_runtime_bytes": total_memory - free_after_release,
            "allocated_after_release_bytes": allocated_after_release,
            "reserved_after_release_bytes": reserved_after_release,
            "rebind_epoch": self._persistent_rebind_epoch,
            "old_runtime": old_runtime,
            "compile_runtime_cache_reset": compile_cache_reset,
            "compilation_state_release": compilation_state_release,
            "module_state_release": module_state_release,
            "gc_was_unfrozen": gc_was_unfrozen,
        }

    def get_general_rebind_runtime_state(self) -> dict[str, object]:
        """Return model-owned resource identities after a general rebind."""
        runner = self.model_runner
        kv_ptrs = [
            tensor.data_ptr()
            for cache in getattr(runner, "kv_caches", ())
            for tensor in (cache if isinstance(cache, (list, tuple))
                           else (cache, ))
            if isinstance(tensor, torch.Tensor)
        ]
        return {
            "rank": self.rank,
            "worker_pid": os.getpid(),
            "hotspot_dir": os.environ.get(
                "VLLM_CUDA_GRAPH_HOTSPOT_DIR"),
            "rebind_epoch": getattr(runner, "_persistent_rebind_epoch", 0),
            "runner_id": id(runner),
            "model_id": id(runner.model),
            "kv_cache_tensors": len(getattr(runner, "kv_caches", ())),
            "kv_cache_ptrs": kv_ptrs,
            "cuda_graph_shapes": list(
                runner.cuda_graph_manager.registry.published if runner is not None else ()),
            "static_forward_context_entries": len(
                runner.compilation_config.static_forward_context),
        }

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

    def _weight_compile_overlap_eligibility(self) -> tuple[bool, list[str]]:
        """Gate overlap by loading configuration, not architecture names."""
        reasons: list[str] = []
        model_config = self.vllm_config.model_config
        if model_config.quantization is not None:
            reasons.append(f"quantization={model_config.quantization}")
        if self.vllm_config.speculative_config is not None:
            reasons.append("speculative_config")
        if self.vllm_config.lora_config is not None:
            reasons.append("lora_config")
        load_format = str(self.vllm_config.load_config.load_format).lower()
        if load_format.rsplit(".", 1)[-1] not in {"auto", "safetensors"}:
            reasons.append(f"load_format={load_format}")
        return not reasons, reasons

    def load_weights_and_trigger_compile(self) -> dict[str, object]:
        """Overlap finalized-weight loading with value-independent compile-only.

        Model construction has already allocated every parameter with its final
        shape and dtype. S4 compile-only reads that structure but suppresses
        kernel launches, while the loader only copies values into the existing
        parameter storage. Keeping both tasks inside one worker RPC avoids
        interleaving two collective RPC streams.
        """
        started = time.perf_counter()
        mode = startup_compile_mode()
        safe_modes = {
            "s0", "stage_separated", "stage_separated_baseline",
            "precompile", "precompile_skip_profile", "s4",
            "a100_optimized", "offline_best_config",
            "offline_best_config_skip_profile",
        }
        overlap_eligible, eligibility_reasons = (
            self._weight_compile_overlap_eligibility())
        if mode not in safe_modes or not overlap_eligible:
            logger.warning(
                "Weight/compile overlap is not eligible for compile mode %s "
                "(%s); using the sequential path.", mode,
                ", ".join(eligibility_reasons) or "unsupported mode")
            load_started = time.perf_counter()
            self.load_weights()
            load_s = time.perf_counter() - load_started
            compile_started = time.perf_counter()
            self.run_to_trigger_compile()
            return {
                "rank": self.rank,
                "elapsed_s": time.perf_counter() - started,
                "load_weights_s": load_s,
                "compile_s": time.perf_counter() - compile_started,
                "overlapped": False,
                "eligibility_reasons": eligibility_reasons,
            }

        def on_worker_device(fn):
            torch.cuda.set_device(self.device)
            with torch.inference_mode():
                return fn()

        from vllm.compilation import weight_compile_overlap

        weight_compile_overlap.begin()

        def load_and_signal():
            load_started = time.perf_counter()
            try:
                on_worker_device(self.load_weights)
                sync_started = time.perf_counter()
                torch.cuda.synchronize(self.device)
                logger.debug("Weight/compile overlap finalized CUDA copies in "
                            "%.3f seconds", time.perf_counter() - sync_started)
                return time.perf_counter() - load_started
            finally:
                weight_compile_overlap.mark_weights_ready()

        def compile_timed():
            compile_started = time.perf_counter()
            on_worker_device(self.run_to_trigger_compile)
            return time.perf_counter() - compile_started

        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="vllm-weight-compile") as pool:
                load_future = pool.submit(load_and_signal)
                compile_future = pool.submit(compile_timed)
                load_s = load_future.result()
                compile_s = compile_future.result()
        finally:
            weight_compile_overlap.end()
        logger.debug("Weight/compile overlap finished in %.3f seconds",
                    time.perf_counter() - started)
        return {
            "rank": self.rank,
            "elapsed_s": time.perf_counter() - started,
            "load_weights_s": load_s,
            "compile_s": compile_s,
            "overlapped": True,
            "eligibility_reasons": [],
        }

    def prepare_startup_assets(self) -> dict[str, object]:
        """Prepare bounded metadata in its consuming worker before model loading.

        Exact graph/operator keys are not known yet. Speculative family reads
        remain untrusted until their existing consumers validate them.
        """
        from itertools import chain, islice
        from vllm.compilation import startup_assets as assets
        from vllm.compilation.startup_resources import resource_dir

        assets.clear_prepared()
        exact = []
        if kv_profile_cache.enabled():
            _, payload = kv_profile_cache.build_key(self.vllm_config, self.device)
            override = os.getenv("VLLM_KV_PROFILE_CACHE_FILE")
            exact.append(Path(override).expanduser() if override else
                         assets.asset_path("kv_profile", payload))
        hotspot_dir = self.model_runner.cuda_graph_manager.profile_dir
        exact.extend([hotspot_dir / "manifest.json",
                      hotspot_dir / "cuda_graph_hits.csv"])
        if os.getenv("VLLM_CUDA_GRAPH_HOTSPOT_CSV"):
            exact.append(Path(os.environ["VLLM_CUDA_GRAPH_HOTSPOT_CSV"]))
        # No recursive walk of large compiled bundles or executable payloads.
        candidates = chain(exact, *(islice(resource_dir(kind).glob("*.json"), 16)
                                   for kind in ("input_metadata", "best_config")))
        summary = assets.prefetch(candidates)
        logger.info("Startup assets prepared in worker: %s", summary)
        return summary

    def release_startup_assets(self) -> dict[str, object]:
        from vllm.compilation import startup_assets as assets
        summary = assets.clear_prepared()
        logger.info("Startup prepared assets released: %s", summary)
        return summary

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
                self.vllm_config, self.device,
                weights_memory=getattr(self.model_runner, "model_memory_usage", None),
                requested_memory=getattr(self, "requested_memory", None),
                init_snapshot=getattr(self, "init_snapshot", None))
        return {
            **static_state,
            "rank": self.rank,
            "device": str(self.device),
            "kv_profile_hit": kv_value is not None and kv_value > 0,
            "kv_profile_key": kv_key,
            "weight_compile_overlap_safe": (
                self._weight_compile_overlap_eligibility()[0]),
        }

    def get_static_startup_resource_snapshot(self) -> dict[str, object]:
        """Probe filesystem-only resources before CUDA initialization ends."""
        from vllm.compilation.startup_resources import resource_dir

        def has_entries(path: Path) -> bool:
            try:
                return path.exists() and any(path.iterdir())
            except OSError:
                return False

        compile_root = resource_dir("compile")
        best_config_path = resource_dir("best_config")
        transformed_path = resource_dir("transformed_code")
        metadata_path = resource_dir("input_metadata")
        hotspot_path = Path(os.environ.get(
            "VLLM_CUDA_GRAPH_HOTSPOT_DIR",
            str(resource_dir("hotspots"))))
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
        if (os.environ.get(
                "VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
                    "1", "true", "yes", "on"
                } and os.environ.get(
                    "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0").lower()
                not in {"1", "true", "yes", "on"}):
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
                self.vllm_config, self.device,
                weights_memory=int(self.model_runner.model_memory_usage),
                requested_memory=self.requested_memory,
                init_snapshot=self.init_snapshot)
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
                profile_result=profile_result,
                weights_memory=int(self.model_runner.model_memory_usage))
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
        if not self.model_runner.cuda_graph_manager.enabled:
            return 0
        controller = self.model_runner.cuda_graph_manager
        distributed = self.vllm_config.parallel_config.world_size > 1
        if distributed:
            torch.distributed.barrier(group=get_world_group().cpu_group)
        try:
            captured = controller.capture_pending(
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
        if not self.model_runner.cuda_graph_manager.enabled:
            return []
        return self.model_runner.cuda_graph_manager.pending_shapes()

    def get_completed_background_captures(self) -> list[int]:
        return self.model_runner.cuda_graph_manager.capture.completed_distributed_shapes()

    def publish_background_capture(self, shape: int) -> None:
        self.model_runner.cuda_graph_manager.capture.publish_distributed(shape)

    def plan_pending_cudagraph_capture(
        self,
        slack_ms: float,
        max_pieces: int,
        default_fixed_ms: float,
        default_piece_ms: float,
        safety_factor: float,
    ) -> Optional[dict[str, object]]:
        if not self.model_runner.cuda_graph_manager.enabled:
            return None
        controller = self.model_runner.cuda_graph_manager
        plan = controller.plan_capture_maintenance(
            slack_ms=slack_ms,
            max_pieces=max_pieces,
            default_fixed_ms=default_fixed_ms,
            default_piece_ms=default_piece_ms,
            safety_factor=safety_factor,
        )
        return None if plan is None else dict(vars(plan))

    def export_adaptive_cudagraph_telemetry(self) -> None:
        if not self.model_runner.cuda_graph_manager.enabled:
            return
        controller = self.model_runner.cuda_graph_manager
        controller.flush_gpu_samples()
        controller.export_telemetry()

    def export_cudagraph_hotspot_profile(self) -> None:
        """Persist one logical workload profile from distributed rank zero."""
        if self.rank == 0:
            self.model_runner.cuda_graph_manager.hotspots.check_and_export(True)

    def shutdown_cudagraphs(self) -> None:
        self.model_runner.cuda_graph_manager.shutdown()

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
