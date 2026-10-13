# SPDX-License-Identifier: Apache-2.0
"""Reviewed EngineCore action bindings; no graph construction or startup at import."""
import os
import queue

def build_action_registry(host):
    self, vllm_config, executor_class, log_stats, executor_fail_callback = host
    from vllm.v1.engine.core import (
        logger, timer, VLLM_VERSION, StructuredOutputManager,
        MirroredProcessingCache, resolve_obj_by_qualname,
        stop_async_weight_prewarm_from_env,
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
    overlap_weight_compile = os.environ.get(
        "VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP", "").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def init_model(_):
        self.model_executor.init_model()

    def stop_weight_prewarm(_):
        stop_async_weight_prewarm_from_env("EngineCore.load_weights")

    def load_weights(_):
        if overlap_weight_compile:
            self.model_executor.load_weights_and_trigger_compile()
        else:
            self.model_executor.load_weights()

    def compile_model(_):
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
        "load_weights": timed(
            "Weight-Compile Overlap" if overlap_weight_compile else
            "Loading Weights", load_weights),
        "compile_model": timed("Model Compilation", compile_model),
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

    actions = {**bootstrap_actions, **runtime_actions}
    def publish(name, action):
        def run(context):
            value = action(context)
            context.put("ready." + name, True)
            return value
        return run
    return {name: publish(name, action) for name, action in actions.items()}

