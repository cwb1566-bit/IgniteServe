# SPDX-License-Identifier: Apache-2.0
"""Reviewed EngineCore action bindings; no graph construction or startup at import."""
import os
import queue

def _timed(stage_name, action, *, summary=None):
    from vllm.v1.phase_profiler import timer

    def invoke(context):
        timer.start(stage_name)
        try:
            return action(context)
        finally:
            timer.end(stage_name, summary=summary)
    return invoke


def build_runtime_actions(self, *, overlap_weight_compile, after_load=None):
    """Shared actions for cold startup and persistent-engine rebind.

    The caller owns lifecycle-specific asset probes and readiness publication.
    The fused weight action joins loading and compilation before returning.
    """
    from vllm.v1.engine.core import (
        logger, StructuredOutputManager, MirroredProcessingCache,
        resolve_obj_by_qualname, stop_async_weight_prewarm_from_env, V1Scheduler,
    )

    def init_model(_):
        self.model_executor.init_model()

    def stop_weight_prewarm(_):
        stop_async_weight_prewarm_from_env("EngineCore.load_weights")

    def load_weights(ctx):
        if overlap_weight_compile:
            result = self.model_executor.load_weights_and_trigger_compile()
            ctx.put("weight_compile_worker_results", result)
        else:
            result = self.model_executor.load_weights()
        if after_load is not None:
            return after_load(ctx)
        return result

    def compile_model(_):
        # With overlap enabled, load_weights has already completed both loading
        # and compilation. Keep this node as a readiness marker for downstream
        # stages; do not compile twice.
        if not overlap_weight_compile:
            self.model_executor.run_to_trigger_compile()

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
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. This scheduler "
                "interface is not public and compatibility may not be "
                "maintained.", scheduler_cls)
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
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)

    runtime_actions = {
        "init_model": _timed("Init Model", init_model),
        "stop_weight_prewarm": stop_weight_prewarm,
        "load_weights": _timed(
            "Weight-Compile Overlap" if overlap_weight_compile else
            "Loading Weights", load_weights),
        # The fused action already compiled; this node only signals readiness.
        "compile_model": _timed("Model Compilation", compile_model,
                                summary=not overlap_weight_compile),
        "kv_profile": _timed("KV Profile", kv_profile),
        "kv_cache_allocate": _timed("KV Cache Allocation",
                                     kv_cache_allocate),
        "cuda_graph_precapture": _timed(
            "CUDA Graph Pre-capture",
            lambda _: self.model_executor.compile_or_warm_up_model()),
        "structured_output": _timed("Others.Structured Output",
                                     structured_output),
        "mm_input_cache": _timed("Others.MM Input Cache", mm_input_cache),
        "scheduler": _timed("Others.Scheduler", create_scheduler),
        "batch_queue": _timed("Others.Batch Queue", create_batch_queue),
        "ready": lambda _: None,
    }

    return runtime_actions


def build_action_registry(host):
    self, vllm_config, executor_class, log_stats, executor_fail_callback = host
    from vllm.v1.engine.core import logger, VLLM_VERSION

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
        context.state = state
        history = context.get("input.cost_history")
        if history:
            estimates = history.estimates(state, context.require("input.cost_defaults"))
            for name, seconds in estimates.items():
                context.estimate(name, seconds)
        context.estimate("kv_profile", 0.01 if state.kv_profile_hit else 2.0)
        return context.put("startup_state", state)

    def static_resource_probe(context):
        snapshots = self.collective_rpc(
            "get_static_startup_resource_snapshot")
        return context.put("static_worker_states", snapshots)

    def prepare_assets(context):
        return context.put("assets.prepared", self.collective_rpc("prepare_startup_assets"))

    bootstrap_actions = {
        "engine_core_construct": _timed("EngineCore Construct",
                                         construct_core),
        "plugins": _timed("Plugin Initialization", load_plugins),
        "executor_shell": _timed("Executor Shell", executor_shell),
        "worker_wrapper": _timed(
            "Worker Wrapper",
            lambda _: self.model_executor.initialize_worker_wrapper()),
        "worker_prepare": _timed(
            "Worker Prepare", lambda _: self.model_executor.prepare_worker()),
        "worker_construct": _timed(
            "Worker Construct",
            lambda _: self.model_executor.construct_worker()),
        "static_resource_probe": _timed(
            "Static Resource Probe", static_resource_probe),
        "prepare_startup_assets": _timed("Prepare Startup Assets", prepare_assets),
        "cuda_device": _timed(
            "CUDA Device Initialization",
            lambda _: self.model_executor.initialize_cuda_device()),
        "distributed_environment": _timed(
            "Distributed Environment",
            lambda _: self.model_executor.
            initialize_distributed_environment()),
        "compile_worker_prewarm": _timed(
            "Compile Worker Prewarm",
            lambda _: self.model_executor.
            initialize_compile_worker_prewarm()),
        "model_runner": _timed(
            "GPU ModelRunner Initialization",
            lambda _: self.model_executor.initialize_model_runner()),
        "usage_reporting": _timed(
            "Usage Reporting",
            lambda _: self.model_executor.initialize_usage_reporting()),
        "state_probe": _timed("Startup State Probe", state_probe),
    }
    overlap_weight_compile = os.environ.get(
        "VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP", "").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def after_load(ctx):
        ctx.put("weights.compile_completed", overlap_weight_compile)
        # This selects a try-cache route; consumption still checks memory.
        states = self.collective_rpc("get_startup_state_snapshot")
        ctx.put("assets.kv_profile_usable", bool(states) and
                all(state.get("kv_profile_hit", False) for state in states))

    runtime_actions = build_runtime_actions(
        self, overlap_weight_compile=overlap_weight_compile, after_load=after_load)
    runtime_actions["ready"] = lambda ctx: ctx.put(
        "assets.released", self.collective_rpc("release_startup_assets"))

    actions = {**bootstrap_actions, **runtime_actions}
    def publish(name, action):
        def run(context):
            value = action(context)
            context.put("ready." + name, True)
            return value
        return run
    registry = {name: publish(name, action) for name, action in actions.items()}
    # The fused action already joined both operations. No worker RPC or
    # compiler capacity is needed to publish this completion marker.
    registry["compile_already_ready"] = publish("compile_model", lambda _: None)
    return registry

