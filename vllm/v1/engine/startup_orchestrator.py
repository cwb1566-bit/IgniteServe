# SPDX-License-Identifier: Apache-2.0
"""State-aware startup DAG primitives for EngineCore initialization.

The orchestrator deliberately has no dependency on CUDA or worker classes.
EngineCore supplies actions and a state snapshot, which keeps the scheduler
unit-testable and makes the legacy startup path a straightforward fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import ExitStack
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, MutableMapping, Optional


StageAction = Callable[["StartupContext"], Any]


@dataclass(frozen=True)
class StartupState:
    """Facts known before constructing an executable startup plan."""

    model: str
    compile_mode: str
    kv_profile_hit: bool
    weight_pagecache_warm: Optional[bool] = None
    compile_cache_present: Optional[bool] = None
    best_config_store_present: Optional[bool] = None
    transformed_code_store_present: Optional[bool] = None
    metadata_store_present: Optional[bool] = None
    hotspot_profile_present: Optional[bool] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        payload = {
            "model": self.model,
            "compile_mode": self.compile_mode,
            "kv_profile_hit": self.kv_profile_hit,
            "weight_pagecache_warm": self.weight_pagecache_warm,
            "compile_cache_present": self.compile_cache_present,
            "best_config_store_present": self.best_config_store_present,
            "transformed_code_store_present": self.transformed_code_store_present,
            "metadata_store_present": self.metadata_store_present,
            "hotspot_profile_present": self.hotspot_profile_present,
            "attributes": dict(self.attributes),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class StartupContext:
    """Thread-safe data exchange between startup stages."""

    def __init__(self, state: StartupState) -> None:
        self.state = state
        self._values: dict[str, Any] = {}
        self._lock = threading.Lock()

    def put(self, key: str, value: Any) -> Any:
        from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
        recorder = get_startup_recorder()
        if recorder is not None:
            recorder.state_access("put", key, value_type=type(value).__name__)
        with self._lock:
            self._values[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
        recorder = get_startup_recorder()
        if recorder is not None:
            recorder.state_access("get", key)
        with self._lock:
            return self._values.get(key, default)

    def require(self, key: str) -> Any:
        from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
        recorder = get_startup_recorder()
        if recorder is not None:
            recorder.state_access("require", key)
        with self._lock:
            if key not in self._values:
                raise KeyError(f"Startup value is not ready: {key}")
            return self._values[key]

    def snapshot(self) -> dict[str, Any]:
        from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
        recorder = get_startup_recorder()
        if recorder is not None:
            recorder.record("state_snapshot")
        with self._lock:
            return dict(self._values)


@dataclass
class StageNode:
    name: str
    action: Optional[StageAction] = None
    dependencies: set[str] = field(default_factory=set)
    estimated_cost_s: float = 0.0
    resources: Mapping[str, int] = field(default_factory=dict)
    enabled: bool = True
    category: str = "startup"
    details: Mapping[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dependencies": sorted(self.dependencies),
            "estimated_cost_s": self.estimated_cost_s,
            "resources": dict(self.resources),
            "enabled": self.enabled,
            "category": self.category,
            "details": dict(self.details),
        }


@dataclass
class StageResult:
    name: str
    status: str
    started_at: float
    finished_at: float
    elapsed_s: float
    thread_name: str
    error: Optional[str] = None


class StartupDAG:
    def __init__(self) -> None:
        self.nodes: dict[str, StageNode] = {}

    def add_node(self, node: StageNode) -> StageNode:
        if node.name in self.nodes:
            raise ValueError(f"Duplicate startup stage: {node.name}")
        self.nodes[node.name] = node
        return node

    def add_dependency(self, node: str, dependency: str) -> None:
        self.nodes[node].dependencies.add(dependency)

    def topological_order(self) -> list[str]:
        missing = {
            dep for node in self.nodes.values() for dep in node.dependencies
            if dep not in self.nodes
        }
        if missing:
            raise ValueError(f"Unknown startup dependencies: {sorted(missing)}")
        indegree = {name: len(node.dependencies)
                    for name, node in self.nodes.items()}
        children: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                children[dep].append(name)
        ready = sorted(name for name, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(self.nodes):
            cyclic = sorted(name for name, degree in indegree.items() if degree)
            raise ValueError(f"Cycle in startup DAG: {cyclic}")
        return order

    def critical_path(self) -> tuple[list[str], float]:
        order = self.topological_order()
        finish: dict[str, float] = {}
        predecessor: dict[str, Optional[str]] = {}
        for name in order:
            node = self.nodes[name]
            enabled_cost = node.estimated_cost_s if node.enabled else 0.0
            if node.dependencies:
                pred = max(node.dependencies, key=lambda dep: finish[dep])
                finish[name] = finish[pred] + enabled_cost
                predecessor[name] = pred
            else:
                finish[name] = enabled_cost
                predecessor[name] = None
        if not finish:
            return [], 0.0
        tail = max(finish, key=finish.get)
        path: list[str] = []
        while tail is not None:
            path.append(tail)
            tail = predecessor[tail]
        path.reverse()
        return path, finish[path[-1]]

    def downstream_costs(self) -> dict[str, float]:
        order = self.topological_order()
        children: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                children[dep].append(name)
        costs: dict[str, float] = {}
        for name in reversed(order):
            own = self.nodes[name].estimated_cost_s if self.nodes[name].enabled else 0.0
            costs[name] = own + max((costs[child] for child in children[name]),
                                    default=0.0)
        return costs

    def serializable(self) -> dict[str, Any]:
        path, cost = self.critical_path()
        return {
            "nodes": [self.nodes[name].serializable()
                      for name in self.topological_order()],
            "predicted_critical_path": path,
            "predicted_critical_path_s": cost,
        }


class StartupExecutionError(RuntimeError):
    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"Startup stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


class ResourceAwareDAGExecutor:
    """Critical-path-first executor with integer resource capacities."""

    def __init__(self, max_workers: int = 4,
                 resource_capacities: Optional[Mapping[str, int]] = None) -> None:
        self.max_workers = max(1, max_workers)
        self.capacities = dict(resource_capacities or {})

    def run(self, dag: StartupDAG, context: StartupContext) -> dict[str, StageResult]:
        from vllm.v1.engine.startup_analysis.recorder import get_startup_recorder
        recorder = get_startup_recorder()
        dag.topological_order()
        priority = dag.downstream_costs()
        pending = set(dag.nodes)
        completed: set[str] = set()
        in_use: dict[str, int] = {}
        running: dict[Future[Any], tuple[str, dict[str, int]]] = {}
        results: dict[str, StageResult] = {}

        def fits(node: StageNode) -> bool:
            for resource, units in node.resources.items():
                capacity = self.capacities.get(resource, 1)
                if units < 0 or units > capacity:
                    return False
                if in_use.get(resource, 0) + units > capacity:
                    return False
            return True

        def reserve(node: StageNode) -> dict[str, int]:
            allocation = dict(node.resources)
            for resource, units in allocation.items():
                in_use[resource] = in_use.get(resource, 0) + units
            return allocation

        def release(allocation: Mapping[str, int]) -> None:
            for resource, units in allocation.items():
                in_use[resource] -= units

        def invoke(node: StageNode) -> tuple[Any, StageResult]:
            started = time.perf_counter()
            status = "completed"
            value: Any = None
            try:
                stage_scope = (recorder.stage(node.name)
                               if recorder is not None else ExitStack())
                with stage_scope:
                    if not node.enabled:
                        status = "skipped"
                    elif node.action is not None:
                        value = node.action(context)
            except BaseException as exc:
                finished = time.perf_counter()
                return exc, StageResult(
                    node.name, "failed", started, finished,
                    finished - started, threading.current_thread().name,
                    repr(exc))
            finished = time.perf_counter()
            return value, StageResult(
                node.name, status, started, finished, finished - started,
                threading.current_thread().name)

        failure: Optional[StartupExecutionError] = None
        affinity_pools: dict[str, ThreadPoolExecutor] = {}
        with ExitStack() as stack, ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="vllm-startup") as pool:
            while pending or running:
                ready = [
                    name for name in pending
                    if dag.nodes[name].dependencies <= completed
                ]
                ready.sort(key=lambda name: (-priority[name], name))
                launched = False
                for name in ready:
                    if len(running) >= self.max_workers:
                        break
                    node = dag.nodes[name]
                    if not fits(node):
                        continue
                    allocation = reserve(node)
                    pending.remove(name)
                    affinity = node.details.get("thread_affinity")
                    submit_pool = pool
                    if affinity:
                        submit_pool = affinity_pools.get(str(affinity))
                        if submit_pool is None:
                            submit_pool = stack.enter_context(ThreadPoolExecutor(
                                max_workers=1,
                                thread_name_prefix=(
                                    "vllm-startup-" + str(affinity))))
                            affinity_pools[str(affinity)] = submit_pool
                    running[submit_pool.submit(invoke, node)] = (
                        name, allocation)
                    launched = True

                if not running:
                    if pending:
                        blocked = sorted(pending)
                        raise RuntimeError(
                            "Startup DAG cannot make progress; resource "
                            f"requirements are unsatisfied: {blocked}")
                    break

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    name, allocation = running.pop(future)
                    release(allocation)
                    value, result = future.result()
                    results[name] = result
                    if result.status == "failed":
                        failure = StartupExecutionError(name, value)
                    else:
                        context.put(f"stage_result.{name}", value)
                        completed.add(name)
                if failure is not None:
                    for future in running:
                        future.cancel()
                    wait(running)
                    break
                if not launched and not done and pending:
                    raise RuntimeError("Startup DAG scheduler stalled")

        if failure is not None:
            raise failure
        return results


class StartupCostHistory:
    """Small persistent median model keyed by startup state signature."""

    def __init__(self, path: Path, max_samples: int = 9) -> None:
        self.path = path
        self.max_samples = max(1, max_samples)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "states": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "states": {}}
        if not isinstance(data, dict):
            return {"version": 1, "states": {}}
        data.setdefault("version", 1)
        data.setdefault("states", {})
        return data

    def estimates(self, state: StartupState,
                  defaults: Mapping[str, float]) -> dict[str, float]:
        data = self._read()
        stages = data.get("states", {}).get(state.signature(), {})
        estimates = dict(defaults)
        for name, samples in stages.items():
            if isinstance(samples, list) and samples:
                estimates[name] = float(median(float(x) for x in samples))
        return estimates

    def update(self, state: StartupState,
               results: Mapping[str, StageResult]) -> None:
        data = self._read()
        stages: MutableMapping[str, Any] = data.setdefault(
            "states", {}).setdefault(state.signature(), {})
        for name, result in results.items():
            if result.status != "completed":
                continue
            samples = stages.setdefault(name, [])
            samples.append(result.elapsed_s)
            del samples[:-self.max_samples]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False,
                                         dir=str(self.path.parent),
                                         encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            temp_name = handle.name
        os.replace(temp_name, self.path)


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
    compile_dependencies = {"init_model", "load_weights"}
    add("compile_model", compile_dependencies,
        {"worker_rpc": 1, "compiler_cpu": 1}, "compilation", {
            "sub_dag": [
                "transformed_code_lookup", "dynamo_tracing",
                "dataflow_metadata_restore", "canonical_graph_mapping",
                "parallel_materialization", "best_config_resolve",
                "compile_finalize",
            ],
            "state_dependent": True,
            "requires_finalized_weights": True,
        })
    # The KV phase always observes finalized model weights. Compilation also
    # waits for this boundary in the current implementation because its dummy
    # forward reads module parameters.
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
    add("cuda_device", {"worker_construct"},
        {"worker_rpc": 1, "gpu_runtime": 1}, "worker", "worker-0")
    add("distributed_environment", {"cuda_device"},
        {"worker_rpc": 1, "gpu_runtime": 1}, "worker", "worker-0")
    # Launch the FX helper before GPUModelRunner construction. The helper then
    # warms asynchronously while the runner allocates its persistent buffers.
    add("compile_worker_prewarm", {"distributed_environment"},
        {"worker_rpc": 1, "compiler_cpu": 1}, "compilation", "worker-0")
    add("model_runner", {"compile_worker_prewarm"},
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
    add("dynamo_trace", {"model_structure", "transformed_code_lookup"},
        "compilation_manager", {"cpu": 1},
        state.transformed_code_store_present is not True,
        "transformed-code-hit")
    add("graph_split", {"dynamo_trace", "transformed_code_lookup"},
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
