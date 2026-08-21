# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import sys
import time
from pathlib import Path
from collections import deque
from collections.abc import Generator
from concurrent.futures import Future
import concurrent.futures
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import make_zmq_socket, resolve_obj_by_qualname
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
from vllm.v1.engine.utils import EngineHandshakeMetadata, EngineZmqAddresses
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc

#增
import threading
from vllm.v1.phase_profiler import timer
from vllm.v1.metrics_recorder import recorder
from vllm.v1.worker.HotspotAnalyzer import HotspotAnalyzer
from vllm.v1.engine.weight_pagecache import stop_async_weight_prewarm_from_env

class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: type[Executor],
                 log_stats: bool,
                 executor_fail_callback: Optional[Callable] = None):

        timer.start("Total Init")
        requested_dag_mode = os.environ.get(
            "VLLM_STARTUP_DAG_MODE", "off").strip().lower()
        deferred_supported = bool(
            getattr(executor_class, "supports_deferred_startup",
                    lambda: False)())
        if requested_dag_mode == "execute" and deferred_supported:
            self._initialize_with_full_startup_dag(
                vllm_config, executor_class, log_stats,
                executor_fail_callback)
            timer.end("Total Init")
            return
        if requested_dag_mode == "execute" and not deferred_supported:
            logger.warning(
                "Executor %s does not expose deferred startup stages; "
                "using startup DAG shadow mode.", executor_class.__name__)
            self._startup_dag_mode_override = "shadow"
        # plugins need to be loaded at the engine/scheduler level too
        timer.start("Init execution environment")
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.vllm_config = vllm_config
        self._adaptive_cudagraph_enabled = os.environ.get(
            "VLLM_ENABLE_ADAPTIVE_CUDAGRAPH", "").lower() in (
                "1", "true", "yes", "on")
        self._adaptive_cudagraph_idle_grace_s = max(
            0.0,
            float(
                os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_IDLE_GRACE_MS", "10"))
            / 1000.0)
        self._adaptive_cudagraph_slo_capture_enabled = (
            self._adaptive_cudagraph_enabled and os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_ENABLE_SLO_CAPTURE",
                "1").lower() in ("1", "true", "yes", "on"))
        self._adaptive_cudagraph_tpot_slo_ms = max(
            0.0,
            float(
                os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_TPOT_SLO_MS",
                               "100")))
        self._adaptive_cudagraph_ttft_slo_ms = max(
            0.0,
            float(
                os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_TTFT_SLO_MS",
                               "2000")))
        self._adaptive_cudagraph_prefill_step_estimate_ms = max(
            0.1,
            float(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_PREFILL_STEP_ESTIMATE_MS",
                    "30")))
        self._adaptive_cudagraph_allow_waiting = os.environ.get(
            "VLLM_ADAPTIVE_CUDAGRAPH_ALLOW_WAITING_REQUESTS",
            "1").lower() in ("1", "true", "yes", "on")
        self._adaptive_cudagraph_slo_guard_ms = max(
            0.0,
            float(
                os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_SLO_GUARD_MS", "5")))
        self._adaptive_cudagraph_capture_estimate_ms = max(
            0.1,
            float(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_CAPTURE_CHUNK_ESTIMATE_MS",
                    "30")))
        self._adaptive_cudagraph_slo_interval_steps = max(
            1,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_SLO_CAPTURE_INTERVAL_STEPS",
                    "8")))
        # Zero selects a slack-driven piece budget. A positive value keeps the
        # fixed-budget mode used by controlled experiments.
        self._adaptive_cudagraph_slo_pieces = max(
            0,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_SLO_PIECES_PER_STEP", "0")))
        self._adaptive_cudagraph_slo_max_pieces = max(
            1,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_SLO_MAX_PIECES_PER_STEP", "64")))
        self._adaptive_cudagraph_capture_fixed_estimate_ms = max(
            0.1,
            float(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_CAPTURE_FIXED_ESTIMATE_MS",
                    "20")))
        self._adaptive_cudagraph_capture_safety_factor = max(
            1.0,
            float(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_CAPTURE_SAFETY_FACTOR",
                    "1.05")))
        self._adaptive_cudagraph_recent_step_ms: deque[float] = deque(
            maxlen=64)
        self._adaptive_cudagraph_recent_capture_ms: deque[float] = deque(
            maxlen=32)
        self._adaptive_cudagraph_capture_samples: deque[
            tuple[int, float]] = deque(maxlen=32)
        self._adaptive_cudagraph_model_steps = 0
        self._adaptive_cudagraph_last_slo_capture_step = 0
        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(
                executor_fail_callback)
        timer.end("Init execution environment")

        startup_dag_mode = getattr(
            self, "_startup_dag_mode_override",
            os.environ.get("VLLM_STARTUP_DAG_MODE", "off")).strip().lower()
        if startup_dag_mode not in {"off", "shadow", "execute"}:
            logger.warning("Unknown VLLM_STARTUP_DAG_MODE=%s; using off",
                           startup_dag_mode)
            startup_dag_mode = "off"
        if (startup_dag_mode == "execute"
                and self.vllm_config.parallel_config.world_size != 1):
            logger.warning(
                "Startup DAG execute mode is currently validated only for "
                "TP=PP=DP=1; emitting a shadow plan and using the existing "
                "distributed startup path for world_size=%d.",
                self.vllm_config.parallel_config.world_size)
            startup_dag_mode = "shadow"
        if startup_dag_mode == "execute":
            self._initialize_with_startup_dag()
            timer.end("Total Init")
            return
        if startup_dag_mode == "shadow":
            self._write_startup_dag_shadow_plan()

        timer.start("Init Model")
        self.model_executor.init_model()
        timer.end("Init Model")

        stop_async_weight_prewarm_from_env("EngineCore.load_weights")
        self._weight_loading_error = None
        # timer.start("Loading Weights")
        # self.model_executor.load_weights()
        # timer.end("Loading Weights")

        def _load_weights_target():
            try:
                self.model_executor.load_weights()
            except BaseException as exc:
                self._weight_loading_error = exc
                raise
        self._weight_loading_thread = threading.Thread(
            target=_load_weights_target,
            daemon=True,
            name="WeightLoadingThread"
        )
        self._weight_loading_thread.start()

        timer.start("Model Compilation")
        self.model_executor.run_to_trigger_compile()
        timer.end("Model Compilation")

        timer.start("KV Cache Init")
        available_gpu_memory = self.model_executor.determine_available_memory()
        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config, available_gpu_memory)

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))
        timer.end("KV Cache Init")

        weight_loading_thread = getattr(self, "_weight_loading_thread", None)
        if (weight_loading_thread is not None
                and weight_loading_thread.is_alive()):
            logger.info("Waiting for background weight loading before CUDA "
                        "graph pre-capture.")
            weight_loading_thread.join()
        if self._weight_loading_error is not None:
            raise self._weight_loading_error

        timer.start("CUDA Graph Pre-capture")
        self.model_executor.compile_or_warm_up_model()
        timer.end("CUDA Graph Pre-capture")

        timer.start("Others")
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                vllm_config.scheduler_config.scheduler_cls)

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size
            > 1,
            log_stats=self.log_stats,
        )

        # Setup MM Input Mapper.
        self.mm_input_cache_server = MirroredProcessingCache(
            vllm_config.model_config)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)
        timer.end("Others")
        timer.end("Total Init")
        model_name = vllm_config.model_config.model.split('/')[-1]
        # timer.save(
        #     filepath=f"./analysis_results/IgniteServe_tests/cold_without_caches/total_init_{model_name}_IgniteServe_without_caches.json",
        #     csv_filepath=f"./analysis_results/IgniteServe_tests/cold_without_caches/timing_summary_{model_name}_IgniteServe_without_caches.csv"
        # )

    def _initialize_engine_core_fields(self, vllm_config: VllmConfig,
                                       log_stats: bool) -> None:
        """Initialize EngineCore-owned state before executor construction."""
        self.vllm_config = vllm_config
        self.log_stats = log_stats
        enabled = os.environ.get(
            "VLLM_ENABLE_ADAPTIVE_CUDAGRAPH", "").lower() in (
                "1", "true", "yes", "on")
        self._adaptive_cudagraph_enabled = enabled
        self._adaptive_cudagraph_idle_grace_s = max(
            0.0, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_IDLE_GRACE_MS", "10")) / 1000.0)
        self._adaptive_cudagraph_slo_capture_enabled = (
            enabled and os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_ENABLE_SLO_CAPTURE", "1").lower()
            in ("1", "true", "yes", "on"))
        self._adaptive_cudagraph_tpot_slo_ms = max(
            0.0, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_TPOT_SLO_MS", "100")))
        self._adaptive_cudagraph_ttft_slo_ms = max(
            0.0, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_TTFT_SLO_MS", "2000")))
        self._adaptive_cudagraph_prefill_step_estimate_ms = max(
            0.1, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_PREFILL_STEP_ESTIMATE_MS", "30")))
        self._adaptive_cudagraph_allow_waiting = os.environ.get(
            "VLLM_ADAPTIVE_CUDAGRAPH_ALLOW_WAITING_REQUESTS", "1").lower()
        self._adaptive_cudagraph_allow_waiting = (
            self._adaptive_cudagraph_allow_waiting in
            ("1", "true", "yes", "on"))
        self._adaptive_cudagraph_slo_guard_ms = max(
            0.0, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_SLO_GUARD_MS", "5")))
        self._adaptive_cudagraph_capture_estimate_ms = max(
            0.1, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_CAPTURE_CHUNK_ESTIMATE_MS", "30")))
        self._adaptive_cudagraph_slo_interval_steps = max(
            1, int(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_SLO_CAPTURE_INTERVAL_STEPS", "8")))
        self._adaptive_cudagraph_slo_pieces = max(
            0, int(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_SLO_PIECES_PER_STEP", "0")))
        self._adaptive_cudagraph_slo_max_pieces = max(
            1, int(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_SLO_MAX_PIECES_PER_STEP", "64")))
        self._adaptive_cudagraph_capture_fixed_estimate_ms = max(
            0.1, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_CAPTURE_FIXED_ESTIMATE_MS", "20")))
        self._adaptive_cudagraph_capture_safety_factor = max(
            1.0, float(os.environ.get(
                "VLLM_ADAPTIVE_CUDAGRAPH_CAPTURE_SAFETY_FACTOR", "1.05")))
        self._adaptive_cudagraph_recent_step_ms = deque(maxlen=64)
        self._adaptive_cudagraph_recent_capture_ms = deque(maxlen=32)
        self._adaptive_cudagraph_capture_samples = deque(maxlen=32)
        self._adaptive_cudagraph_model_steps = 0
        self._adaptive_cudagraph_last_slo_capture_step = 0

    def _initialize_with_full_startup_dag(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Optional[Callable],
    ) -> None:
        """Plan from EngineCore construction and expand after state probing."""
        from vllm.compilation.startup_resources import resource_dir
        from vllm.v1.engine.startup_orchestrator import (
            ResourceAwareDAGExecutor,
            StartupContext,
            StartupCostHistory,
            StartupState,
            build_bootstrap_startup_dag,
            build_engine_startup_dag,
            combine_startup_dags,
            write_bootstrap_plan,
            write_plan,
        )

        root = resource_dir("startup_orchestration")
        model_name = Path(str(vllm_config.model_config.model)).name
        plan_path = root / "plans" / (
            f"{model_name}_{os.getpid()}_{time.time_ns()}.json")
        history = StartupCostHistory(root / "startup_costs.json")
        bootstrap_state = StartupState(
            model=str(vllm_config.model_config.model),
            compile_mode="bootstrap",
            kv_profile_hit=False,
            attributes={
                "planning_state": "before_worker_state",
                "executor_class": (executor_class.__module__ + "." +
                                   executor_class.__qualname__),
                "device": str(vllm_config.device_config.device),
                "dtype": str(vllm_config.model_config.dtype),
                "max_model_len": vllm_config.model_config.max_model_len,
                "max_num_batched_tokens": (
                    vllm_config.scheduler_config.max_num_batched_tokens),
                "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
                "tensor_parallel_size": (
                    vllm_config.parallel_config.tensor_parallel_size),
                "pipeline_parallel_size": (
                    vllm_config.parallel_config.pipeline_parallel_size),
                "data_parallel_size": (
                    vllm_config.parallel_config.data_parallel_size),
            },
        )

        def timed(stage_name, action):
            def invoke(context):
                timer.start(stage_name)
                try:
                    return action(context)
                finally:
                    timer.end(stage_name)
            return invoke

        def construct_core(_):
            self._initialize_engine_core_fields(vllm_config, log_stats)
            logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                        VLLM_VERSION, vllm_config)

        def load_plugins(_):
            from vllm.plugins import load_general_plugins
            load_general_plugins()

        def executor_shell(_):
            self.model_executor = executor_class(vllm_config, defer_init=True)
            if executor_fail_callback is not None:
                self.model_executor.register_failure_callback(
                    executor_fail_callback)

        def state_probe(context):
            state = self._collect_startup_state()
            static_states = context.get("static_worker_states")
            if static_states:
                state.attributes["static_worker_states"] = static_states
            return context.put("startup_state", state)

        def static_resource_probe(context):
            snapshots = self.collective_rpc(
                "get_static_startup_resource_snapshot")
            return context.put("static_worker_states", snapshots)

        bootstrap_actions = {
            "engine_core_construct": timed("EngineCore Construct",
                                             construct_core),
            "plugins": timed("Plugin Initialization", load_plugins),
            "executor_shell": timed("Executor Shell", executor_shell),
            "worker_wrapper": timed(
                "Worker Wrapper",
                lambda _: self.model_executor.initialize_worker_wrapper()),
            "worker_prepare": timed(
                "Worker Prepare", lambda _: self.model_executor.prepare_worker()),
            "worker_construct": timed(
                "Worker Construct",
                lambda _: self.model_executor.construct_worker()),
            "static_resource_probe": timed(
                "Static Resource Probe", static_resource_probe),
            "cuda_device": timed(
                "CUDA Device Initialization",
                lambda _: self.model_executor.initialize_cuda_device()),
            "distributed_environment": timed(
                "Distributed Environment",
                lambda _: self.model_executor.
                initialize_distributed_environment()),
            "compile_worker_prewarm": timed(
                "Compile Worker Prewarm",
                lambda _: self.model_executor.
                initialize_compile_worker_prewarm()),
            "model_runner": timed(
                "GPU ModelRunner Initialization",
                lambda _: self.model_executor.initialize_model_runner()),
            "usage_reporting": timed(
                "Usage Reporting",
                lambda _: self.model_executor.initialize_usage_reporting()),
            "state_probe": timed("Startup State Probe", state_probe),
        }
        bootstrap = build_bootstrap_startup_dag(bootstrap_actions)
        bootstrap_defaults = {
            name: node.estimated_cost_s for name, node in bootstrap.nodes.items()
        }
        bootstrap = build_bootstrap_startup_dag(
            bootstrap_actions,
            history.estimates(bootstrap_state, bootstrap_defaults))
        write_bootstrap_plan(plan_path, bootstrap)

        bootstrap_executor = ResourceAwareDAGExecutor(
            max_workers=max(1, int(os.environ.get(
                "VLLM_STARTUP_DAG_BOOTSTRAP_MAX_WORKERS", "2"))),
            resource_capacities={
                "worker_rpc": 1,
                "gpu_runtime": 1,
                "gpu_memory": 1,
                "compiler_cpu": 1,
                "cpu": 2,
            })
        context = StartupContext(bootstrap_state)
        timer.start("Startup DAG Bootstrap")
        try:
            bootstrap_results = bootstrap_executor.run(bootstrap, context)
        finally:
            timer.end("Startup DAG Bootstrap")
        history.update(bootstrap_state, bootstrap_results)

        state = context.require("startup_state")

        def init_model(_):
            self.model_executor.init_model()

        def stop_weight_prewarm(_):
            stop_async_weight_prewarm_from_env("EngineCore.load_weights")

        def load_weights(_):
            self.model_executor.load_weights()

        def kv_profile(ctx):
            return ctx.put(
                "available_gpu_memory",
                self.model_executor.determine_available_memory())

        def kv_cache_allocate(ctx):
            available = ctx.require("available_gpu_memory")
            gpu_blocks, cpu_blocks, config = self._initialize_kv_caches(
                self.vllm_config, available)
            self.vllm_config.cache_config.num_gpu_blocks = gpu_blocks
            self.vllm_config.cache_config.num_cpu_blocks = cpu_blocks
            self.collective_rpc("initialize_cache",
                                args=(gpu_blocks, cpu_blocks))
            return ctx.put("kv_cache_config", config)

        def structured_output(_):
            self.structured_output_manager = StructuredOutputManager(
                self.vllm_config)

        def mm_input_cache(_):
            self.mm_input_cache_server = MirroredProcessingCache(
                self.vllm_config.model_config)

        def create_scheduler(ctx):
            scheduler_cls = self.vllm_config.scheduler_config.scheduler_cls
            Scheduler = (resolve_obj_by_qualname(scheduler_cls)
                         if isinstance(scheduler_cls, str) else scheduler_cls)
            self.scheduler = Scheduler(
                vllm_config=self.vllm_config,
                kv_cache_config=ctx.require("kv_cache_config"),
                structured_output_manager=self.structured_output_manager,
                include_finished_set=(self.vllm_config.parallel_config.
                                      data_parallel_size > 1),
                log_stats=self.log_stats,
            )

        def create_batch_queue(_):
            self.batch_queue_size = self.model_executor.max_concurrent_batches
            self.batch_queue = None
            if self.batch_queue_size > 1:
                self.batch_queue = queue.Queue(self.batch_queue_size)

        runtime_actions = {
            "init_model": timed("Init Model", init_model),
            "stop_weight_prewarm": stop_weight_prewarm,
            "load_weights": timed("Loading Weights", load_weights),
            "compile_model": timed(
                "Model Compilation",
                lambda _: self.model_executor.run_to_trigger_compile()),
            "kv_profile": timed("KV Profile", kv_profile),
            "kv_cache_allocate": timed("KV Cache Allocation",
                                         kv_cache_allocate),
            "cuda_graph_precapture": timed(
                "CUDA Graph Pre-capture",
                lambda _: self.model_executor.compile_or_warm_up_model()),
            "structured_output": timed("Others.Structured Output",
                                         structured_output),
            "mm_input_cache": timed("Others.MM Input Cache", mm_input_cache),
            "scheduler": timed("Others.Scheduler", create_scheduler),
            "batch_queue": timed("Others.Batch Queue", create_batch_queue),
            "ready": lambda _: None,
        }
        runtime = build_engine_startup_dag(state, runtime_actions)
        runtime_defaults = {
            name: node.estimated_cost_s for name, node in runtime.nodes.items()
        }
        runtime = build_engine_startup_dag(
            state, runtime_actions, history.estimates(state, runtime_defaults))
        expanded = combine_startup_dags(bootstrap, runtime)
        write_plan(plan_path, state, expanded, bootstrap_results)

        runtime_executor = ResourceAwareDAGExecutor(
            # Two workers are the minimum needed to preserve the established
            # compile-only/weight-loading overlap. A one-worker schedule would
            # execute compile-only against unloaded parameters.
            max_workers=max(2, int(os.environ.get(
                "VLLM_STARTUP_DAG_MAX_WORKERS", "2"))),
            resource_capacities={
                "worker_rpc": 2,
                "disk_io": 1,
                "compiler_cpu": 1,
                "gpu_compute": 1,
                "gpu_memory": 1,
                "cpu": 2,
            })
        timer.start("Startup DAG Runtime")
        try:
            runtime_results = runtime_executor.run(runtime, context)
        finally:
            timer.end("Startup DAG Runtime")
        history.update(state, runtime_results)
        all_results = {**bootstrap_results, **runtime_results}
        write_plan(plan_path, state, expanded, all_results)
        logger.info(
            "Full startup DAG finished: plan=%s bootstrap_nodes=%d "
            "runtime_nodes=%d state=%s", plan_path, len(bootstrap.nodes),
            len(runtime.nodes), state.signature()[:16])

    def _collect_startup_state(self):
        from vllm.v1.engine.startup_orchestrator import StartupState
        from vllm.v1.worker.startup_compile import startup_compile_mode

        worker_states = self.collective_rpc("get_startup_state_snapshot")
        if not worker_states:
            raise RuntimeError("No worker startup state was returned")

        def all_workers(key: str) -> bool:
            return all(bool(state.get(key)) for state in worker_states)

        model_config = self.vllm_config.model_config
        scheduler_config = self.vllm_config.scheduler_config
        parallel_config = self.vllm_config.parallel_config
        return StartupState(
            model=str(model_config.model),
            compile_mode=startup_compile_mode(),
            kv_profile_hit=all_workers("kv_profile_hit"),
            compile_cache_present=all_workers("compile_cache_present"),
            best_config_store_present=all_workers(
                "best_config_store_present"),
            transformed_code_store_present=all_workers(
                "transformed_code_store_present"),
            metadata_store_present=all_workers("metadata_store_present"),
            hotspot_profile_present=all_workers("hotspot_profile_present"),
            attributes={
                "worker_states": worker_states,
                # The compile-only dummy forward and GraphPickler read module
                # parameters. Concurrent in-place loading is a data race even
                # for dense FP16 models, so compilation requires the complete
                # loader/post-load boundary for every configuration.
                "compile_requires_finalized_weights": True,
                "overlap_safe_weight_loading": False,
                "max_model_len": model_config.max_model_len,
                "max_num_batched_tokens":
                    scheduler_config.max_num_batched_tokens,
                "max_num_seqs": scheduler_config.max_num_seqs,
                "tensor_parallel_size": parallel_config.tensor_parallel_size,
                "pipeline_parallel_size":
                    parallel_config.pipeline_parallel_size,
                "data_parallel_size": parallel_config.data_parallel_size,
            },
        )

    def _startup_orchestration_paths(self) -> tuple[Path, Path]:
        from vllm.compilation.startup_resources import resource_dir

        root = resource_dir("startup_orchestration")
        plan_root = root / "plans"
        model_name = Path(str(self.vllm_config.model_config.model)).name
        plan_name = f"{model_name}_{os.getpid()}_{time.time_ns()}.json"
        return root / "startup_costs.json", plan_root / plan_name

    def _write_startup_dag_shadow_plan(self) -> None:
        from vllm.v1.engine.startup_orchestrator import (
            StartupCostHistory,
            build_engine_startup_dag,
            write_plan,
        )

        try:
            state = self._collect_startup_state()
            history_path, plan_path = self._startup_orchestration_paths()
            history = StartupCostHistory(history_path)
            dag = build_engine_startup_dag(state, {})
            defaults = {
                name: node.estimated_cost_s for name, node in dag.nodes.items()
            }
            costs = history.estimates(state, defaults)
            dag = build_engine_startup_dag(state, {}, costs)
            write_plan(plan_path, state, dag)
            path, predicted_s = dag.critical_path()
            logger.info(
                "Startup DAG shadow plan saved: path=%s state=%s "
                "critical_path=%s predicted_s=%.6f", plan_path,
                state.signature()[:16], " -> ".join(path), predicted_s)
        except Exception:
            # Shadow mode must never change legacy startup availability.
            logger.exception("Startup DAG shadow planning failed; continuing "
                             "with the legacy startup path")

    def _initialize_with_startup_dag(self) -> None:
        from vllm.v1.engine.startup_orchestrator import (
            ResourceAwareDAGExecutor,
            StartupContext,
            StartupCostHistory,
            build_engine_startup_dag,
            write_plan,
        )

        state = self._collect_startup_state()
        history_path, plan_path = self._startup_orchestration_paths()
        history = StartupCostHistory(history_path)

        def timed(stage_name, action):
            def invoke(context):
                timer.start(stage_name)
                try:
                    return action(context)
                finally:
                    timer.end(stage_name)
            return invoke

        def init_model(_):
            self.model_executor.init_model()

        def stop_weight_prewarm(_):
            stop_async_weight_prewarm_from_env("EngineCore.load_weights")

        def load_weights(_):
            self.model_executor.load_weights()

        def compile_model(_):
            self.model_executor.run_to_trigger_compile()

        def kv_profile(context):
            return context.put(
                "available_gpu_memory",
                self.model_executor.determine_available_memory())

        def kv_cache_allocate(context):
            available_gpu_memory = context.require("available_gpu_memory")
            num_gpu_blocks, num_cpu_blocks, kv_cache_config = (
                self._initialize_kv_caches(
                    self.vllm_config, available_gpu_memory))
            self.vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
            self.vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
            self.collective_rpc("initialize_cache",
                                args=(num_gpu_blocks, num_cpu_blocks))
            return context.put("kv_cache_config", kv_cache_config)

        def cuda_graph_precapture(_):
            self.model_executor.compile_or_warm_up_model()

        def structured_output(_):
            self.structured_output_manager = StructuredOutputManager(
                self.vllm_config)

        def mm_input_cache(_):
            self.mm_input_cache_server = MirroredProcessingCache(
                self.vllm_config.model_config)

        def create_scheduler(context):
            scheduler_cls = self.vllm_config.scheduler_config.scheduler_cls
            Scheduler = (resolve_obj_by_qualname(scheduler_cls)
                         if isinstance(scheduler_cls, str) else scheduler_cls)
            if Scheduler is not V1Scheduler:
                logger.warning(
                    "Using configured V1 scheduler class %s. This scheduler "
                    "interface is not public and compatibility may not be "
                    "maintained.", scheduler_cls)
            self.scheduler = Scheduler(
                vllm_config=self.vllm_config,
                kv_cache_config=context.require("kv_cache_config"),
                structured_output_manager=self.structured_output_manager,
                include_finished_set=(
                    self.vllm_config.parallel_config.data_parallel_size > 1),
                log_stats=self.log_stats,
            )

        def create_batch_queue(_):
            self.batch_queue_size = self.model_executor.max_concurrent_batches
            self.batch_queue = None
            if self.batch_queue_size > 1:
                logger.info("Batch queue is enabled with size %d",
                            self.batch_queue_size)
                self.batch_queue = queue.Queue(self.batch_queue_size)

        actions = {
            "init_model": timed("Init Model", init_model),
            "stop_weight_prewarm": stop_weight_prewarm,
            "load_weights": timed("Loading Weights", load_weights),
            "compile_model": timed("Model Compilation", compile_model),
            "kv_profile": timed("KV Profile", kv_profile),
            "kv_cache_allocate": timed("KV Cache Allocation",
                                        kv_cache_allocate),
            "cuda_graph_precapture": timed("CUDA Graph Pre-capture",
                                            cuda_graph_precapture),
            "structured_output": timed("Others.Structured Output",
                                       structured_output),
            "mm_input_cache": timed("Others.MM Input Cache", mm_input_cache),
            "scheduler": timed("Others.Scheduler", create_scheduler),
            "batch_queue": timed("Others.Batch Queue", create_batch_queue),
            "ready": lambda _: None,
        }
        initial_dag = build_engine_startup_dag(state, actions)
        defaults = {
            name: node.estimated_cost_s
            for name, node in initial_dag.nodes.items()
        }
        costs = history.estimates(state, defaults)
        dag = build_engine_startup_dag(state, actions, costs)
        write_plan(plan_path, state, dag)
        predicted_path, predicted_s = dag.critical_path()
        logger.info(
            "Startup DAG execution started: state=%s kv_profile_hit=%s "
            "critical_path=%s predicted_s=%.6f plan=%s",
            state.signature()[:16], state.kv_profile_hit,
            " -> ".join(predicted_path), predicted_s, plan_path)

        executor = ResourceAwareDAGExecutor(
            max_workers=max(2, int(os.environ.get(
                "VLLM_STARTUP_DAG_MAX_WORKERS", "2"))),
            resource_capacities={
                "worker_rpc": 2,
                "disk_io": 1,
                "compiler_cpu": 1,
                "gpu_compute": 1,
                "gpu_memory": 1,
                "cpu": 2,
            },
        )
        timer.start("Startup DAG Execution")
        context = StartupContext(state)
        try:
            results = executor.run(dag, context)
        finally:
            timer.end("Startup DAG Execution")
        history.update(state, results)
        write_plan(plan_path, state, dag, results)

        actual_path = sorted(
            results.values(), key=lambda result: result.started_at)
        logger.info(
            "Startup DAG execution finished: plan=%s stages=%s",
            plan_path,
            ", ".join(
                f"{result.name}:{result.elapsed_s:.6f}s"
                for result in actual_path),
        )
        # timer.save(
        #     filepath=f"./analysis_results/IgniteServe_tests/cold_with_caches/total_init_{model_name}_IgniteServe_with_caches.json",
        #     csv_filepath=f"./analysis_results/IgniteServe_tests/cold_with_caches/timing_summary_{model_name}_IgniteServe_with_caches.csv"
        # )
        # timer.save(
        #     filepath=f"./analysis_results/Startup/cold_without_caches/total_init_{model_name}_vLLM_without_caches.json",
        #     csv_filepath=f"./analysis_results/Startup/cold_without_caches/timing_summary_{model_name}_vLLM_without_caches.csv"
        # )
        # timer.save(
        #     filepath=f"./analysis_results/Startup/cold_with_caches/total_init_{model_name}_vLLM_with_caches.json",
        #     csv_filepath=f"./analysis_results/Startup/cold_with_caches/timing_summary_{model_name}_vLLM_with_caches.csv"
        # )

    def _initialize_kv_caches(
            self, vllm_config: VllmConfig, available_gpu_memory) -> tuple[int, int, KVCacheConfig]:
        # start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        # available_gpu_memory = self.model_executor.determine_available_memory()

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Initialize kv cache
        self.model_executor.initialize_from_config(kv_cache_configs)

        # elapsed = time.time() - start
        # logger.info(("init engine (profile, create kv cache, "
        #              "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for a multimodal input, then it will be
            # fetched from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client cache, so
            # anything that has a hash must have a HIT cache entry here
            # as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_cache_server.get_and_update_p1(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self.structured_output_manager.grammar_init(req)

        if req.kv_transfer_params is not None and (
                not self.scheduler.get_kv_connector()):
            logger.warning("Got kv_transfer_params, but no KVConnector found. "
                           "Disabling KVTransfer for this request.")

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def execute_model(self, scheduler_output: SchedulerOutput):
        try:
            return self.model_executor.execute_model(scheduler_output)
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(self.vllm_config, scheduler_output,
                                  self.scheduler.make_stats())
            raise err

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        model_output = self.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore

        return (engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0)

    def step_with_batch_queue(
            self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        assert self.batch_queue is not None

        engine_core_outputs = None
        scheduler_output = None
        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        if not self.batch_queue.full():
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens > 0:
                future = self.model_executor.execute_model(scheduler_output)
                self.batch_queue.put_nowait(
                    (future, scheduler_output))  # type: ignore

        scheduled_batch = (scheduler_output is not None
                           and scheduler_output.total_num_scheduled_tokens > 0)

        # If no more requests can be scheduled and the job queue is not empty,
        # block until the first batch in the job queue is finished.
        # TODO(comaniac): Ideally we should peek the first batch in the
        # job queue to check if it's finished before scheduling a new batch,
        # but peeking the first element in a queue is not thread-safe,
        # so we need more work.
        if not scheduled_batch and not self.batch_queue.empty():
            future, scheduler_output = self.batch_queue.get_nowait()
            # Blocking until the first result is available.
            model_output = future.result()
            self.batch_queue.task_done()
            engine_core_outputs = (self.scheduler.update_from_output(
                scheduler_output, model_output))

        return engine_core_outputs, scheduled_batch

    def shutdown(self):
        HotspotAnalyzer.check_and_export(True)
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            try:
                self.model_executor.collective_rpc(
                    "export_cudagraph_hotspot_profile")
            except Exception:
                logger.exception(
                    "Failed to export distributed CUDA graph hotspots")
            if self._adaptive_cudagraph_enabled:
                try:
                    self.model_executor.collective_rpc(
                        "export_adaptive_cudagraph_telemetry")
                except Exception:
                    logger.exception(
                        "Failed to export adaptive CUDA graph telemetry")
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 processor, P0 mirror, P1 mirror)
        if self.scheduler.has_unfinished_requests():
            logger.warning("Resetting the multi-modal cache when requests are "
                           "in progress may lead to desynced internal caches.")

        self.mm_input_cache_server.reset()

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_executor.save_sharded_state(path=path,
                                               pattern=pattern,
                                               max_size=max_size)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args,
                                                  kwargs)

    def save_tensorized_model(
        self,
        tensorizer_config,
    ) -> None:
        self.model_executor.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address)
            # Only publish request queue stats to coordinator for "internal"
            # LB mode.
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

            self._init_data_parallel(vllm_config)

            super().__init__(vllm_config, executor_class, log_stats,
                             executor_fail_callback)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        threading.Thread(target=self.process_input_sockets,
                         args=(addresses.inputs, addresses.coordinator_input,
                               identity),
                         daemon=True).start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  self.engine_index),
            daemon=True)
        self.output_thread.start()

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: Optional[str],
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal loadbalancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external loadbalancing, two handshakes are performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 engine itself which doesn't require
        the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        handshake = self._perform_handshake(input_ctx, handshake_address,
                                            identity, is_local, vllm_config,
                                            vllm_config.parallel_config)
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, local_client,
                vllm_config)
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: Optional[ParallelConfig] = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(ctx,
                             handshake_address,
                             zmq.DEALER,
                             identity=identity,
                             linger=5000,
                             bind=False) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(handshake_socket, local_client,
                                               parallel_config_to_update)
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address
            handshake_socket.send(
                msgspec.msgpack.encode({
                    "status": "READY",
                    "local": local_client,
                    "num_gpu_blocks": num_gpu_blocks,
                    "dp_stats_address": dp_stats_address,
                }))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> EngineZmqAddresses:

        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode({
                "status": "HELLO",
                "local": local_client,
            }))

        # Receive initialization message.
        logger.info("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError("Did not receive response from front-end "
                               f"process within {HANDSHAKE_TIMEOUT_MINS} "
                               f"minutes")
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata)
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[EngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.engines_running and not self.scheduler.has_requests():
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            if self._adaptive_cudagraph_enabled:
                try:
                    # Give newly arrived work priority over maintenance. Only
                    # capture after an uninterrupted idle grace period.
                    req = self.input_queue.get(
                        timeout=self._adaptive_cudagraph_idle_grace_s)
                except queue.Empty:
                    # A producer may enqueue work exactly as the grace timeout
                    # expires. Recheck before entering a non-preemptive CUDA
                    # stream-capture window.
                    if not self.input_queue.empty():
                        continue
                    if self._capture_pending_cudagraphs_coordinated() > 0:
                        # Re-enter the grace period before capturing another
                        # graph, so a request can preempt the maintenance queue.
                        continue
                    req = self.input_queue.get()
            else:
                req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        step_started = time.perf_counter()
        outputs, model_executed = self.step_fn()
        step_elapsed_ms = (time.perf_counter() - step_started) * 1000.0
        if model_executed:
            self._adaptive_cudagraph_recent_step_ms.append(step_elapsed_ms)
            self._adaptive_cudagraph_model_steps += 1
        # Put EngineCoreOutputs into the output queue.
        for output in (outputs.items() if outputs else ()):
            self.output_queue.put_nowait(output)

        if model_executed:
            self._maybe_capture_cudagraph_with_slo_slack()

        return model_executed

    def _capture_pending_cudagraphs_coordinated(
        self,
        max_pieces_per_shape: Optional[int] = None,
        deadline_ms: Optional[float] = None,
        selected_shape: Optional[int] = None,
    ) -> int:
        """Capture only a shape that is pending on every distributed rank."""
        world_size = self.vllm_config.parallel_config.world_size
        if world_size == 1:
            return self.model_executor.capture_pending_cudagraphs(
                max_pieces_per_shape=max_pieces_per_shape,
                deadline_ms=deadline_ms,
                selected_shape=selected_shape,
            )
        if os.environ.get(
                "VLLM_ENABLE_DISTRIBUTED_ADAPTIVE_CUDAGRAPH",
                "").strip().lower() not in ("1", "true", "yes", "on"):
            return 0

        pending_by_rank = self.model_executor.collective_rpc(
            "get_pending_cudagraph_shapes")
        if len(pending_by_rank) != world_size or not pending_by_rank:
            return 0
        common = set(int(shape) for shape in pending_by_rank[0])
        for pending in pending_by_rank[1:]:
            common.intersection_update(int(shape) for shape in pending)
        if selected_shape is not None:
            candidate = int(selected_shape)
            if candidate not in common:
                return 0
        else:
            candidate = next(
                (int(shape) for shape in pending_by_rank[0]
                 if int(shape) in common), None)
            if candidate is None:
                return 0
        logger.info(
            "Distributed adaptive CUDA graph capture admitted: "
            "shape=%d world_size=%d", candidate, world_size)
        return self.model_executor.capture_pending_cudagraphs(
            max_pieces_per_shape=max_pieces_per_shape,
            deadline_ms=deadline_ms,
            selected_shape=candidate,
        )

    @staticmethod
    def _percentile(values: deque[float], quantile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = int(round((len(ordered) - 1) * quantile))
        return ordered[index]

    def _request_slo_ms(self, request: Request, key: str,
                        default: float) -> float:
        sampling_params = request.sampling_params
        extra_args = (sampling_params.extra_args
                      if sampling_params is not None else None)
        if not extra_args or key not in extra_args:
            return default
        try:
            return max(0.0, float(extra_args[key]))
        except (TypeError, ValueError):
            return default

    def _select_slo_capture_piece_budget(
            self, slack_ms: float) -> tuple[int, float]:
        """Choose the largest conservative capture unit fitting the slack."""
        fixed_budget = self._adaptive_cudagraph_slo_pieces
        if fixed_budget > 0:
            estimate_ms = (self._percentile(
                self._adaptive_cudagraph_recent_capture_ms, 0.90)
                           if self._adaptive_cudagraph_recent_capture_ms else
                           self._adaptive_cudagraph_capture_estimate_ms)
            return ((fixed_budget, estimate_ms)
                    if estimate_ms <= slack_ms else (0, estimate_ms))

        max_pieces = self._adaptive_cudagraph_slo_max_pieces
        candidates = sorted({
            pieces
            for pieces in (1, 2, 4, 8, 16, 32, max_pieces)
            if pieces <= max_pieces
        })
        fixed_ms = min(self._adaptive_cudagraph_capture_fixed_estimate_ms,
                       self._adaptive_cudagraph_capture_estimate_ms)
        scalable_ms = max(
            0.0, self._adaptive_cudagraph_capture_estimate_ms - fixed_ms)

        selected = 0
        selected_estimate_ms = fixed_ms
        for candidate in candidates:
            matching_samples = [
                elapsed_ms for budget, elapsed_ms in
                self._adaptive_cudagraph_capture_samples
                if budget == candidate
            ]
            if matching_samples:
                estimate_ms = (self._percentile(matching_samples, 0.90) *
                               self._adaptive_cudagraph_capture_safety_factor)
            else:
                estimate_ms = (fixed_ms + scalable_ms * candidate /
                               max_pieces)
            if estimate_ms <= slack_ms:
                selected = candidate
                selected_estimate_ms = estimate_ms
        return selected, selected_estimate_ms

    def _maybe_capture_cudagraph_with_slo_slack(self) -> None:
        """Borrow a bounded inter-token gap when every active request fits."""
        if not self._adaptive_cudagraph_slo_capture_enabled:
            return
        parallel_config = self.vllm_config.parallel_config
        distributed_capture_enabled = os.environ.get(
            "VLLM_ENABLE_DISTRIBUTED_ADAPTIVE_CUDAGRAPH",
            "").strip().lower() in ("1", "true", "yes", "on")
        if ((parallel_config.world_size != 1
             and not distributed_capture_enabled)
                or parallel_config.data_parallel_size != 1
                or self.batch_queue is not None):
            return
        if (self._adaptive_cudagraph_model_steps
                - self._adaptive_cudagraph_last_slo_capture_step
                < self._adaptive_cudagraph_slo_interval_steps):
            return
        if not self.input_queue.empty():
            return

        running = getattr(self.scheduler, "running", None)
        waiting = getattr(self.scheduler, "waiting", None)
        if not running:
            return
        if waiting and not self._adaptive_cudagraph_allow_waiting:
            return
        # Never spend capture slack before first-token delivery or while a
        # chunked prefill remains in the active batch.
        if any(request.num_output_tokens <= 0
               or request.num_computed_tokens < request.num_prompt_tokens
               for request in running):
            return
        if len(self._adaptive_cudagraph_recent_step_ms) < 4:
            return

        step_p90_ms = self._percentile(
            self._adaptive_cudagraph_recent_step_ms, 0.90)
        tpot_slo_ms = min(
            self._request_slo_ms(request, "cudagraph_tpot_slo_ms",
                                 self._adaptive_cudagraph_tpot_slo_ms)
            for request in running)
        slack_ms = (tpot_slo_ms - step_p90_ms
                    - self._adaptive_cudagraph_slo_guard_ms)

        if waiting:
            max_batched_tokens = max(
                1, int(
                    getattr(self.scheduler, "max_num_scheduled_tokens", 1)))
            queued_prompt_tokens = 0
            prefill_step_ms = max(
                step_p90_ms,
                self._adaptive_cudagraph_prefill_step_estimate_ms)
            for request in waiting:
                queued_prompt_tokens += max(
                    1, request.num_prompt_tokens
                    - request.num_computed_tokens)
                prefill_steps_ahead = (
                    queued_prompt_tokens + max_batched_tokens - 1
                ) // max_batched_tokens
                ttft_slo_ms = self._request_slo_ms(
                    request, "cudagraph_ttft_slo_ms",
                    self._adaptive_cudagraph_ttft_slo_ms)
                elapsed_ms = (time.time() - request.arrival_time) * 1000.0
                waiting_slack_ms = (
                    ttft_slo_ms - elapsed_ms
                    - prefill_steps_ahead * prefill_step_ms
                    - self._adaptive_cudagraph_slo_guard_ms)
                slack_ms = min(slack_ms, waiting_slack_ms)

        # An optional end-to-end SLO is checked per request. A value of zero
        # leaves total-latency admission disabled while TPOT remains active.
        now = time.time()
        for request in running:
            e2e_slo_ms = self._request_slo_ms(
                request, "cudagraph_e2e_slo_ms", 0.0)
            if e2e_slo_ms <= 0.0:
                continue
            remaining_tokens = max(
                0, request.max_tokens - request.num_output_tokens)
            elapsed_ms = (now - request.arrival_time) * 1000.0
            request_slack_ms = (e2e_slo_ms - elapsed_ms
                                - remaining_tokens * step_p90_ms)
            slack_ms = min(slack_ms, request_slack_ms)

        default_piece_ms = max(
            0.01,
            (self._adaptive_cudagraph_capture_estimate_ms
             - self._adaptive_cudagraph_capture_fixed_estimate_ms)
            / self._adaptive_cudagraph_slo_max_pieces,
        )
        capture_plans = self.model_executor.collective_rpc(
            "plan_pending_cudagraph_capture",
            args=(
                slack_ms,
                self._adaptive_cudagraph_slo_max_pieces,
                self._adaptive_cudagraph_capture_fixed_estimate_ms,
                default_piece_ms,
                self._adaptive_cudagraph_capture_safety_factor,
            ),
        )
        plans = [plan for plan in capture_plans if plan is not None]
        if not plans or len(plans) != parallel_config.world_size:
            return
        plan_shapes = {int(plan["shape"]) for plan in plans}
        if len(plan_shapes) != 1:
            logger.warning(
                "Skipping distributed CUDA graph capture because rank "
                "plans disagree: shapes=%s", sorted(plan_shapes))
            return
        selected_shape = next(iter(plan_shapes))
        piece_budget = min(int(plan["piece_budget"]) for plan in plans)
        predicted_capture_ms = max(
            float(plan["predicted_ms"]) for plan in plans)
        if piece_budget <= 0 or predicted_capture_ms > slack_ms:
            return
        # A final queue check narrows the race with newly arrived work. The
        # admitted unit is one non-preemptive piece, so the remaining race is
        # bounded by the measured chunk estimate rather than a whole model.
        if not self.input_queue.empty():
            return

        self._adaptive_cudagraph_last_slo_capture_step = (
            self._adaptive_cudagraph_model_steps)
        capture_started = time.perf_counter()
        try:
            progressed = self._capture_pending_cudagraphs_coordinated(
                max_pieces_per_shape=piece_budget,
                deadline_ms=slack_ms,
                selected_shape=selected_shape,
            )
        except Exception:
            logger.exception(
                "Adaptive CUDA graph SLO capture maintenance failed")
            return
        capture_elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
        if progressed > 0:
            self._adaptive_cudagraph_recent_capture_ms.append(
                capture_elapsed_ms)
            self._adaptive_cudagraph_capture_samples.append(
                (piece_budget, capture_elapsed_ms))
            logger.info(
                "Adaptive CUDA graph SLO capture progressed: workers=%d "
                "piece_budget=%d predicted_ms=%.3f elapsed_ms=%.3f "
                "admitted_slack_ms=%.3f "
                "step_p90_ms=%.3f tpot_slo_ms=%.3f", progressed,
                piece_budget, predicted_capture_ms, capture_elapsed_ms,
                slack_ms, step_p90_ms, tpot_slo_ms)
            if capture_elapsed_ms > slack_ms:
                logger.warning(
                    "Adaptive CUDA graph capture chunk exceeded admitted "
                    "slack: elapsed_ms=%.3f slack_ms=%.3f",
                    capture_elapsed_ms, slack_ms)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output)))
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: %s",
                         request_type)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal("vLLM shutdown signal from EngineCore failed "
                         "to send. Please report this issue.")

    def process_input_sockets(self, input_addresses: list[str],
                              coord_input_address: Optional[str],
                              identity: bytes):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx,
                                    input_address,
                                    zmq.DEALER,
                                    identity=identity,
                                    bind=False))
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(ctx,
                                    coord_input_address,
                                    zmq.XSUB,
                                    identity=identity,
                                    bind=False))
                # Send subscription message to coordinator.
                coord_socket.send(b'\x01')

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b'')
                poller.register(input_socket, zmq.POLLIN)
            if coord_socket is not None:
                poller.register(coord_socket, zmq.POLLIN)

            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(
                        copy=False)
                    request_type = EngineCoreRequestType(
                        bytes(type_frame.buffer))

                    # Deserialize the request data.
                    decoder = add_request_decoder if (
                        request_type
                        == EngineCoreRequestType.ADD) else generic_decoder
                    request = decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(self, output_paths: list[str],
                               coord_output_path: Optional[str],
                               engine_index: int):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
                for output_path in output_paths
            ]
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx, coord_output_path, zmq.PUSH, bind=False,
                    linger=4000)) if coord_output_path is not None else None
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(buffers,
                                                               copy=False,
                                                               track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
    ):
        self._decorate_logs()

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(vllm_config, local_client, handshake_address,
                         executor_class, log_stats, client_handshake_address,
                         dp_rank)

    def _decorate_logs(self):
        # Add process-specific prefix to stdout and stderr before
        # we initialize the engine.
        from multiprocessing import current_process
        process_name = current_process().name
        pid = os.getpid()
        _add_prefix(sys.stdout, process_name, pid)
        _add_prefix(sys.stderr, process_name, pid)

    def _init_data_parallel(self, vllm_config: VllmConfig):

        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug("Setting kv_transfer_config.engine_id to %s",
                         vllm_config.kv_transfer_config.engine_id)

        from vllm.platforms import current_platform
        device_control_env_var = current_platform.device_control_env_var
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            os.environ[device_control_env_var] = ",".join(
                str(current_platform.device_id_to_physical_device_id(i))
                for i in range(local_dp_rank *
                               world_size, (local_dp_rank + 1) * world_size))
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f"base value: \"{os.getenv(device_control_env_var)}\"") from e

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: EngineCoreRequest):
        if self.has_coordinator and request.current_wave != self.current_wave:
            if request.current_wave > self.current_wave:
                self.current_wave = request.current_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave)))

        super().add_request(request)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                    new_wave >= self.current_wave):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.",
                                 new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(*counts)
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs)

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave %d finished, pausing engine loop.",
                                 self.current_wave)
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (client_index,
                         EngineCoreOutputs(wave_complete=self.current_wave)))
                self.current_wave += 1

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:

        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.counter += 1
        if self.counter != 32:
            return True
        self.counter = 0

        return ParallelConfig.has_unfinished_dp(self.dp_group,
                                                local_unfinished)


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = \
            local_dp_rank

        # Ray sets CUDA_VISIBLE_DEVICES to empty string,
        # we clean this up to be able to properly initialize
        # data parallel groups.
        del os.environ['CUDA_VISIBLE_DEVICES']

        super().__init__(vllm_config, local_client, "", executor_class,
                         log_stats)

    def _decorate_logs(self):
        pass

    @contextmanager
    def _perform_handshakes(self, handshake_address: str, identity: bytes,
                            local_client: bool, vllm_config: VllmConfig,
                            client_handshake_address: Optional[str]):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()
