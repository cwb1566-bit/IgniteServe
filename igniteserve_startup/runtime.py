# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral, event-driven startup scheduling with resource budgets."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import ExitStack
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional


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

    def __init__(self, state: Any = None, recorder: Any = None) -> None:
        self.state = state
        self.recorder = recorder
        self.changed = threading.Event()
        self._values: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.costs: dict[str, float] = {}

    def estimate(self, node: str, seconds: float) -> None:
        import math
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid startup cost")
        with self._lock:
            self.costs[node] = seconds
        self.changed.set()

    def put(self, key: str, value: Any) -> Any:
        recorder = self.recorder
        if recorder is not None:
            recorder.state_access("put", key, value_type=type(value).__name__)
        with self._lock:
            self._values[key] = value
        self.changed.set()
        return value

    def get(self, key: str, default: Any = None) -> Any:
        recorder = self.recorder
        if recorder is not None:
            recorder.state_access("get", key)
        with self._lock:
            return self._values.get(key, default)

    def require(self, key: str) -> Any:
        recorder = self.recorder
        if recorder is not None:
            recorder.state_access("require", key)
        with self._lock:
            if key not in self._values:
                raise KeyError(f"Startup value is not ready: {key}")
            return self._values[key]

    def snapshot(self) -> dict[str, Any]:
        recorder = self.recorder
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
    required_state: set[str] = field(default_factory=set)
    produced_state: set[str] = field(default_factory=set)
    routes: list[dict[str, Any]] = field(default_factory=list)

    def serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dependencies": sorted(self.dependencies),
            "estimated_cost_s": self.estimated_cost_s,
            "resources": dict(self.resources),
            "enabled": self.enabled,
            "category": self.category,
            "details": dict(self.details),
            "required_state": sorted(self.required_state),
            "produced_state": sorted(self.produced_state),
            "routes": [{k: v for k, v in route.items() if k != "action"}
                       for route in self.routes],
        }

    def resolve(self, state: Mapping[str, Any]) -> "StageNode":
        matches = [r for r in self.routes if all(
            key in state and type(state[key]) is bool and state[key] == value
            for key, value in r["when"].items())]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous startup routes: {self.name}")
        if not matches:
            return self
        route = matches[0]
        return replace(self, action=route["action"], resources=route["resources"],
                       estimated_cost_s=route["estimated_cost_s"],
                       details={**self.details, "route": route["name"],
                                "thread_affinity": route["thread_affinity"]})


@dataclass
class StageResult:
    name: str
    status: str
    started_at: float
    finished_at: float
    elapsed_s: float
    thread_name: str
    error: Optional[str] = None
    route: str = "default"


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
        dependencies = self.ordering_dependencies()
        missing = {
            dep for deps in dependencies.values() for dep in deps
            if dep not in self.nodes
        }
        if missing:
            raise ValueError(f"Unknown startup dependencies: {sorted(missing)}")
        indegree = {name: len(deps) for name, deps in dependencies.items()}
        children: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, deps in dependencies.items():
            for dep in deps:
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

    def ordering_dependencies(self) -> dict[str, set[str]]:
        producers: dict[str, str] = {}
        for name, node in self.nodes.items():
            if not node.enabled:
                continue
            for key in node.produced_state:
                if key in producers:
                    raise ValueError(f"Multiple producers for state {key!r}")
                producers[key] = name
        return {
            name: set(node.dependencies) | {
                producers[key] for key in node.required_state if key in producers
            }
            for name, node in self.nodes.items()
        }

    def critical_path(self) -> tuple[list[str], float]:
        order = self.topological_order()
        dependencies = self.ordering_dependencies()
        finish: dict[str, float] = {}
        predecessor: dict[str, Optional[str]] = {}
        for name in order:
            node = self.nodes[name]
            enabled_cost = node.estimated_cost_s if node.enabled else 0.0
            if dependencies[name]:
                pred = max(dependencies[name], key=lambda dep: finish[dep])
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

    def downstream_costs(self, completed=(), estimates=None) -> dict[str, float]:
        order = self.topological_order()
        children: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, deps in self.ordering_dependencies().items():
            for dep in deps:
                children[dep].append(name)
        costs: dict[str, float] = {}
        for name in reversed(order):
            own = ((estimates or {}).get(name, self.nodes[name].estimated_cost_s)
                   if self.nodes[name].enabled and name not in completed else 0.0)
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
        dag.topological_order()
        recorder = context.recorder
        pending, completed = set(dag.nodes), set()
        in_use: dict[str, int] = {}
        running: dict[Future, tuple[StageNode, float, str]] = {}
        results: dict[str, StageResult] = {}
        active_affinities: set[str] = set()

        def fits(node):
            return all(0 <= units <= self.capacities.get(resource, 1)
                       and in_use.get(resource, 0) + units
                       <= self.capacities.get(resource, 1)
                       for resource, units in node.resources.items())

        def invoke(node):
            scope = recorder.stage(node.name) if recorder else ExitStack()
            with scope:
                if node.enabled and node.action:
                    return node.action(context)

        def wake(_):
            context.changed.set()

        def fail(node, exc):
            # Once an action starts, retrying the original constructor is unsafe.
            for future in running:
                future.cancel()
            for future in list(running):
                try:
                    value = future.result()
                    if isinstance(value, Future):
                        value.result()
                except BaseException:
                    pass
            raise StartupExecutionError(node.name, exc) from exc

        with ExitStack() as stack:
            pool = stack.enter_context(ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="startup"))
            affinity_pools = {}
            while pending or running:
                context.changed.clear()
                for future, (node, started, thread_name) in list(running.items()):
                    if not future.done():
                        continue
                    del running[future]
                    try:
                        value = future.result()
                        # A returned Future represents actual readiness, not submission.
                        if isinstance(value, Future):
                            running[value] = (node, started, thread_name)
                            value.add_done_callback(wake)
                            continue
                        missing = node.produced_state - context.snapshot().keys()
                        if node.enabled and missing:
                            raise ValueError(f"Action did not publish {sorted(missing)}")
                    except BaseException as exc:
                        fail(node, exc)
                    for resource, units in node.resources.items():
                        in_use[resource] -= units
                    active_affinities.discard(node.details.get("thread_affinity"))
                    ended = time.perf_counter()
                    results[node.name] = StageResult(
                        node.name, "completed" if node.enabled else "skipped",
                        started, ended, ended - started, thread_name,
                        route=node.details.get("route", "default"))
                    completed.add(node.name)
                    context.put(f"stage_result.{node.name}", value)

                with context._lock:
                    for name, seconds in context.costs.items():
                        if name in pending:
                            dag.nodes[name].estimated_cost_s = seconds
                state = context.snapshot()
                resolved = {}
                for name in pending:
                    try:
                        resolved[name] = dag.nodes[name].resolve(state)
                    except ValueError as exc:
                        fail(dag.nodes[name], exc)
                estimates = {name: node.estimated_cost_s for name, node in resolved.items()}
                now = time.perf_counter()
                for node, started, _ in running.values():
                    estimates[node.name] = max(0., node.estimated_cost_s - (now - started))
                priority = dag.downstream_costs(completed, estimates)
                ready_state = state.keys()
                ready = sorted(
                    (node for node in resolved.values()
                     if node.dependencies <= completed
                     and (not node.enabled or node.required_state <= ready_state)),
                    key=lambda node: (-priority[node.name], node.name))
                for node in ready:
                    affinity = node.details.get("thread_affinity")
                    if len(running) >= self.max_workers:
                        break
                    if not fits(node) or (affinity and affinity in active_affinities):
                        continue
                    pending.remove(node.name)
                    for resource, units in node.resources.items():
                        in_use[resource] = in_use.get(resource, 0) + units
                    if affinity:
                        active_affinities.add(affinity)
                    started = time.perf_counter()
                    if affinity == "main_thread":
                        future = Future()
                        try:
                            future.set_result(invoke(node))
                        except BaseException as exc:
                            future.set_exception(exc)
                        thread_name = threading.current_thread().name
                    else:
                        target = pool
                        if affinity:
                            if affinity not in affinity_pools:
                                affinity_pools[affinity] = stack.enter_context(
                                    ThreadPoolExecutor(max_workers=1,
                                                       thread_name_prefix=affinity))
                            target = affinity_pools[affinity]
                        future = target.submit(invoke, node)
                        thread_name = affinity or "startup"
                    running[future] = (node, started, thread_name)
                    future.add_done_callback(wake)
                if running:
                    context.changed.wait()
                elif pending:
                    raise StartupExecutionError(
                        "<readiness>", RuntimeError(
                            f"No producer can make pending nodes ready: {sorted(pending)}"))
        return results



