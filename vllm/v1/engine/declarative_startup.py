# SPDX-License-Identifier: Apache-2.0
"""The only vLLM-specific entry into the file-driven startup runtime."""
import json
import os
from pathlib import Path

from igniteserve_startup.contract import ContractRejected, identity, prepare
from igniteserve_startup.runtime import ResourceAwareDAGExecutor


def uses_v3(path):
    if not path:
        return True
    try:
        return json.loads(Path(path).read_text()).get("schema") == "igniteserve-startup-v3"
    except (OSError, ValueError):
        return False


def run(engine, config, executor_class, log_stats, fail_callback):
    from vllm.logger import init_logger
    from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
    logger = init_logger(__name__)
    if (str(config.device_config.device).split(":")[0] != "cuda"
            or config.parallel_config.world_size != 1
            or config.parallel_config.data_parallel_size != 1):
        logger.warning("Packaged startup contract is qualified for single-GPU CUDA; "
                       "using the existing sequential/distributed path")
        return False
    directory = Path(__file__).parent
    contract = Path(os.environ.get("VLLM_STARTUP_CONTRACT_PATH",
                                   str(directory / "startup_contract.json")))
    adapter = Path(os.environ.get("VLLM_STARTUP_ADAPTER_PATH",
                                  str(contract.parent / "vllm_startup_adapter.py")))
    host = (engine, config, executor_class, log_stats, fail_callback)
    facts = {"engine": "vllm-v1-deferred",
             "two_phase_prewarm": os.environ.get(
                 "VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM", "0") == "1"}
    try:
        dag, context, capacities, payload = prepare(
            contract, adapter, host, {}, facts, directory.parents[2],
            recorder=get_startup_recorder(),
            allow_candidate=os.environ.get(
                "VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION", "0") == "1")
    except ContractRejected as exc:
        logger.warning("Declarative startup rejected before execution: %s; "
                       "using sequential startup", exc)
        return False
    from vllm.compilation.startup_resources import resource_dir
    from vllm.v1.engine.startup_orchestrator import StartupCostHistory
    history = StartupCostHistory(resource_dir("startup_cost"))
    context.put("input.cost_history", history)
    context.put("input.cost_defaults", {n.name: n.estimated_cost_s
                                       for n in dag.nodes.values()})
    executor = ResourceAwareDAGExecutor(
        max_workers=max(1, int(os.environ.get("VLLM_STARTUP_DAG_MAX_WORKERS", "2"))),
        resource_capacities=capacities)
    logger.info("Declarative startup: contract=%s nodes=%d identity=%s",
                contract, len(dag.nodes), identity(payload)[:16])
    try:
        results = executor.run(dag, context)
    except BaseException:
        model_executor = getattr(engine, "model_executor", None)
        if model_executor is not None:
            try:
                model_executor.shutdown()
            except Exception:
                logger.exception("Failed to clean up startup executor")
        raise
    finally:
        if context.recorder is not None:
            context.recorder.flush()
    if context.state is not None:
        history.update(context.state, results)
    report = os.environ.get("VLLM_STARTUP_GENERATED_REPORT")
    if report:
        Path(report).write_text(json.dumps({
            "identity": identity(payload), "execution_kind": "generated_adapter",
            "results": {name: value.__dict__ for name, value in results.items()},
            "states": sorted(context.snapshot()),
            "prepared_assets": context.get("assets.prepared", []),
            "released_assets": context.get("assets.released", []),
        }, indent=2))
    logger.info("Declarative startup completed: %d/%d actions", len(results), len(dag.nodes))
    return True
