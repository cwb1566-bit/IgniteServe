# SPDX-License-Identifier: Apache-2.0
"""vLLM compatibility facade and legacy plan builders."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional

from igniteserve_startup.runtime import (
    ResourceAwareDAGExecutor, StageAction, StageNode, StageResult,
    StartupContext as BaseStartupContext, StartupDAG, StartupExecutionError,
    StartupState,
)


class StartupContext(BaseStartupContext):
    def __init__(self, state):
        from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
        super().__init__(state, recorder=get_startup_recorder())


class StartupCostHistory:
    """Per-state median costs; history is a hint, never a correctness guard."""

    def __init__(self, path: Path, max_samples: int = 9) -> None:
        self.path = path
        self.max_samples = max(1, max_samples)

    def _identity(self, state):
        from vllm.compilation.startup_assets import source_hash
        return {"state": state.signature(),
                "source": source_hash("v1/engine/startup_orchestrator.py")}

    def _path(self, identity):
        from vllm.compilation.startup_assets import digest
        return self.path / f"{digest(identity)}.json" if not self.path.suffix else self.path

    def estimates(self, state, defaults):
        from vllm.compilation.startup_assets import read_record
        identity = self._identity(state)
        stages = read_record(self._path(identity), identity) or {}
        estimates = dict(defaults)
        for name, samples in stages.items():
            if isinstance(samples, list) and samples:
                estimates[name] = float(median(float(x) for x in samples))
        return estimates

    def update(self, state, results):
        from vllm.compilation.startup_assets import locked, read_record, record, write_json
        identity = self._identity(state)
        path = self._path(identity)
        with locked(path):
            stages = read_record(path, identity) or {}
            for name, result in results.items():
                if result.status == "completed":
                    samples = stages.setdefault(name, [])
                    samples.append(result.elapsed_s)
                    del samples[:-self.max_samples]
            write_json(path, record(identity, stages))


def build_engine_startup_dag(
    state: StartupState,
    actions: Mapping[str, StageAction],
    costs: Optional[Mapping[str, float]] = None,
) -> StartupDAG:
    """Build the executable EngineCore DAG described by IgniteServe 4.2/4.3.

    Compilation remains a composite action. Its tracing, transformed-code
    restore, metadata/dataflow restore, canonical materialization, best-config
    resolution, and finalization sub-DAG is recorded in ``details`` and is
    executed by the existing Compilation Manager.
    """
    estimate = {
        "init_model": 1.0,
        "stop_weight_prewarm": 0.01,
        "load_weights": 10.0,
        "compile_model": 10.0,
        "kv_profile": 2.0 if state.kv_profile_hit else 10.0,
        "kv_cache_allocate": 1.0,
        "cuda_graph_precapture": 3.0,
        "structured_output": 0.2,
        "mm_input_cache": 0.1,
        "scheduler": 0.2,
        "batch_queue": 0.01,
        "ready": 0.0,
    }
    estimate.update(costs or {})
    dag = StartupDAG()

    def add(name: str, deps: set[str], resources: Mapping[str, int],
            category: str, details: Optional[Mapping[str, Any]] = None) -> None:
        dag.add_node(StageNode(
            name=name,
            action=actions.get(name),
            dependencies=deps,
            estimated_cost_s=estimate[name],
            resources=resources,
            category=category,
            details=details or {},
        ))

    add("init_model", set(), {"worker_rpc": 1}, "model")
    add("stop_weight_prewarm", {"init_model"}, {}, "io")
    add("load_weights", {"stop_weight_prewarm"},
        {"worker_rpc": 1, "disk_io": 1}, "model", {
            "includes_post_load": bool(state.attributes.get(
                "compile_requires_finalized_weights", False)),
        })
    requires_finalized_weights = bool(state.attributes.get(
        "compile_requires_finalized_weights", True))
    compile_dependencies = {"init_model"}
    if requires_finalized_weights:
        compile_dependencies.add("load_weights")
    add("compile_model", compile_dependencies,
        {"worker_rpc": 1, "compiler_cpu": 1}, "compilation", {
            "sub_dag": [
                "transformed_code_lookup", "dynamo_tracing",
                "dataflow_metadata_restore", "canonical_graph_mapping",
                "parallel_materialization", "best_config_resolve",
                "compile_finalize",
            ],
            "state_dependent": True,
            "requires_finalized_weights": requires_finalized_weights,
        })
    # KV profiling and all real execution always observe finalized weights.
    kv_dependencies = {"compile_model", "load_weights"}
    kv_resources = ({"worker_rpc": 1} if state.kv_profile_hit else
                    {"worker_rpc": 1, "gpu_compute": 1})
    add("kv_profile", kv_dependencies, kv_resources,
        "kv_cache", {"cache_hit": state.kv_profile_hit})
    # Allocation mutates the same CUDA memory arena as weight loading. Keep
    # this dependency even on a profile-cache hit.
    add("kv_cache_allocate", {"kv_profile", "load_weights"},
        {"worker_rpc": 1, "gpu_memory": 1}, "kv_cache")
    add("cuda_graph_precapture", {"compile_model", "kv_cache_allocate"},
        {"worker_rpc": 1, "gpu_compute": 1, "gpu_memory": 1}, "cuda_graph")
    add("structured_output", {"init_model"}, {"cpu": 1}, "runtime")
    add("mm_input_cache", {"init_model"}, {"cpu": 1}, "runtime")
    add("scheduler", {"structured_output", "kv_cache_allocate"},
        {"cpu": 1}, "runtime")
    add("batch_queue", {"scheduler"}, {"cpu": 1}, "runtime")
    add("ready", {"cuda_graph_precapture", "batch_queue", "mm_input_cache"},
        {}, "runtime")
    _apply_configured_contract(dag)
    return dag


def build_bootstrap_startup_dag(
    actions: Mapping[str, StageAction],
    costs: Optional[Mapping[str, float]] = None,
) -> StartupDAG:
    """Build the executable DAG from EngineCore semantic construction.

    This graph deliberately ends at ``state_probe``. The model/KV/graph
    portion cannot be specialized until the worker has selected its CUDA
    device and can validate device-dependent persistent resources.
    """
    estimate = {
        "engine_core_construct": 0.01,
        "plugins": 0.1,
        "executor_shell": 0.01,
        "worker_wrapper": 0.05,
        "worker_prepare": 0.1,
        "worker_construct": 0.2,
        "static_resource_probe": 0.05,
        "cuda_device": 0.8,
        "distributed_environment": 0.5,
        "compile_worker_prewarm": 0.2,
        "model_runner": 0.8,
        "usage_reporting": 0.01,
        "state_probe": 0.05,
    }
    estimate.update(costs or {})
    dag = StartupDAG()

    def add(name: str, deps: set[str], resources: Mapping[str, int],
            category: str, thread_affinity: Optional[str] = None) -> None:
        details = ({"thread_affinity": thread_affinity}
                   if thread_affinity else {})
        dag.add_node(StageNode(
            name=name,
            action=actions.get(name),
            dependencies=deps,
            estimated_cost_s=estimate[name],
            resources=resources,
            category=category,
            details=details,
        ))

    add("engine_core_construct", set(), {"cpu": 1}, "engine_core")
    add("plugins", {"engine_core_construct"}, {"cpu": 1}, "environment")
    add("executor_shell", {"engine_core_construct"}, {"cpu": 1},
        "executor")
    add("worker_wrapper", {"executor_shell"}, {"cpu": 1}, "executor")
    add("worker_prepare", {"worker_wrapper", "plugins"}, {"cpu": 1},
        "worker")
    add("worker_construct", {"worker_prepare"}, {"cpu": 1}, "worker")
    add("static_resource_probe", {"worker_construct"}, {"disk_io": 1},
        "startup_manager")
    two_phase_compile_prewarm = os.environ.get(
        "VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM", "0") == "1"
    if two_phase_compile_prewarm:
        compile_prewarm_dependencies = {"worker_construct"}
        cuda_dependencies = {"compile_worker_prewarm"}
        distributed_dependencies = {"cuda_device"}
    else:
        compile_prewarm_dependencies = {"distributed_environment"}
        cuda_dependencies = {"worker_construct"}
        distributed_dependencies = {"cuda_device"}
    add("cuda_device", cuda_dependencies,
        {"worker_rpc": 1, "gpu_runtime": 1}, "worker", "worker-0")
    add("compile_worker_prewarm", compile_prewarm_dependencies,
        {"worker_rpc": 1, "compiler_cpu": 1}, "compilation", "worker-0")
    add("distributed_environment", distributed_dependencies,
        {"worker_rpc": 1, "gpu_runtime": 1}, "worker", "worker-0")
    # The helper warms asynchronously while distributed setup and the runner
    # allocate their persistent state.
    add("model_runner", {"compile_worker_prewarm", "distributed_environment"},
        {"worker_rpc": 1, "gpu_memory": 1}, "model", "worker-0")
    add("usage_reporting", {"model_runner"}, {"worker_rpc": 1}, "runtime",
        "worker-0")
    add("state_probe", {"usage_reporting", "compile_worker_prewarm",
                        "static_resource_probe"},
        {"worker_rpc": 1, "cpu": 1}, "startup_manager", "worker-0")
    _apply_configured_contract(dag)
    return dag


def _apply_configured_contract(dag: StartupDAG) -> None:
    contract = os.environ.get("VLLM_STARTUP_CONTRACT_PATH")
    if not contract:
        return
    from vllm.v1.engine.startup_analysis.contract import apply_startup_contract
    apply_startup_contract(dag, Path(contract), strict=False)


def combine_startup_dags(bootstrap: StartupDAG,
                         runtime: StartupDAG) -> StartupDAG:
    """Return the expanded full DAG without mutating executable sub-DAGs."""
    combined = StartupDAG()
    for source in (bootstrap, runtime):
        for node in source.nodes.values():
            combined.add_node(StageNode(
                name=node.name,
                action=None,
                dependencies=set(node.dependencies),
                estimated_cost_s=node.estimated_cost_s,
                resources=dict(node.resources),
                enabled=node.enabled,
                category=node.category,
                details=dict(node.details),
            ))
    combined.add_dependency("init_model", "state_probe")
    return combined


def write_bootstrap_plan(path: Path, dag: StartupDAG) -> None:
    """Persist the pre-state plan before device-dependent expansion."""
    payload = {
        "schema": "vllm-startup-dag-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "planning_state": "awaiting_worker_state",
        "expansion_point": "state_probe",
        "plan": dag.serializable(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def build_logical_startup_dag(state: StartupState) -> StartupDAG:
    """Describe manager-owned leaf stages that are not independently callable.

    These nodes make the full startup dependency structure visible for cost
    modeling and future interface extraction. They are never passed to the
    executable DAG scheduler.
    """
    dag = StartupDAG()

    def add(name: str, deps: set[str], category: str,
            resources: Mapping[str, int], enabled: bool = True,
            replacement: Optional[str] = None) -> None:
        details = {"owner": category, "logical_only": True}
        if replacement:
            details["replacement"] = replacement
        dag.add_node(StageNode(
            name=name,
            dependencies=deps,
            resources=resources,
            enabled=enabled,
            category=category,
            details=details,
        ))

    add("environment", set(), "startup_manager", {"cpu": 1})
    add("pagecache_probe", {"environment"}, "startup_manager", {"cpu": 1})
    add("pagecache_prewarm", {"pagecache_probe"}, "startup_manager",
        {"disk_io": 1}, state.weight_pagecache_warm is not True,
        "warm-pagecache-hit")
    add("model_structure", {"environment"}, "model_loader",
        {"worker_rpc": 1})
    add("weight_shard_map", {"model_structure"}, "model_loader", {"cpu": 1})
    add("weight_io_copy", {"weight_shard_map", "pagecache_prewarm"},
        "model_loader", {"disk_io": 1, "worker_rpc": 1})

    add("transformed_code_lookup", {"model_structure"},
        "compilation_manager", {"cpu": 1},
        state.transformed_code_store_present is True)
    # The transformed-code resource avoids depyf decompilation and artifact
    # regeneration; it does not persist Dynamo's guarded frame cache.
    add("dynamo_trace", {"model_structure"},
        "compilation_manager", {"cpu": 1}, True)
    add("transformed_code_decompile",
        {"dynamo_trace", "transformed_code_lookup"},
        "compilation_manager", {"cpu": 1},
        state.transformed_code_store_present is not True,
        "transformed-code-artifact-hit")
    add("graph_split", {"dynamo_trace"},
        "compilation_manager", {"cpu": 1})
    add("dataflow_metadata_restore", {"graph_split"},
        "compilation_manager", {"cpu": 1},
        state.metadata_store_present is True)
    add("fake_tensor_propagation", {"graph_split", "dataflow_metadata_restore"},
        "compilation_manager", {"cpu": 1},
        state.metadata_store_present is not True, "metadata-hit")
    add("canonical_graph_mapping",
        {"dataflow_metadata_restore", "fake_tensor_propagation"},
        "compilation_manager", {"cpu": 1})
    add("compile_cache_materialization", {"canonical_graph_mapping"},
        "compilation_manager", {"compiler_cpu": 1},
        state.compile_cache_present is True)
    add("parallel_canonical_compile", {"canonical_graph_mapping"},
        "compilation_manager", {"compiler_cpu": 1},
        state.compile_cache_present is not True, "compile-cache-hit")
    add("best_config_resolve",
        {"compile_cache_materialization", "parallel_canonical_compile"},
        "compilation_manager", {"gpu_compute": 1},
        True,
        "semantic-best-config-hit" if state.best_config_store_present else None)
    add("compile_finalize", {"best_config_resolve"},
        "compilation_manager", {"cpu": 1})

    add("kv_profile_lookup", {"environment"}, "kv_manager", {"cpu": 1})
    add("kv_real_profile", {"kv_profile_lookup", "compile_finalize",
                            "weight_io_copy"}, "kv_manager",
        {"gpu_compute": 1}, not state.kv_profile_hit, "kv-profile-hit")
    add("kv_allocate", {"kv_profile_lookup", "kv_real_profile",
                        "weight_io_copy"}, "kv_manager", {"gpu_memory": 1})

    add("hotspot_manifest_validate", {"environment"}, "graph_manager",
        {"cpu": 1})
    add("selected_shape_warmup", {"compile_finalize", "kv_allocate",
                                  "hotspot_manifest_validate"},
        "graph_manager", {"gpu_compute": 1})
    add("selected_shape_capture", {"selected_shape_warmup"},
        "graph_manager", {"gpu_compute": 1, "gpu_memory": 1})
    add("runtime_objects", {"model_structure", "kv_allocate"},
        "runtime_manager", {"cpu": 1})
    add("engine_ready", {"selected_shape_capture", "runtime_objects"},
        "startup_manager", {})
    return dag


def write_plan(path: Path, state: StartupState, dag: StartupDAG,
               results: Optional[Mapping[str, StageResult]] = None,
               planning_state: str = "expanded") -> None:
    logical_dag = build_logical_startup_dag(state)
    payload = {
        "schema": "vllm-startup-dag-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "planning_state": planning_state,
        "expansion_point": "state_probe",
        "state": {
            "signature": state.signature(),
            **{key: value for key, value in state.__dict__.items()
               if key != "attributes"},
            "attributes": dict(state.attributes),
        },
        "plan": dag.serializable(),
        "logical_leaf_plan": logical_dag.serializable(),
    }
    if results is not None:
        payload["results"] = {
            name: result.__dict__ for name, result in results.items()
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
