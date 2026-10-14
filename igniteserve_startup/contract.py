# SPDX-License-Identifier: Apache-2.0
"""Trusted adapter binding. This loader is not a sandbox for Python plugins."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from .runtime import StageNode, StartupContext, StartupDAG


class ContractRejected(ValueError):
    pass


def names(value, field):
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))):
        raise ContractRejected(f"{field} must be a list of unique nonempty names")
    return set(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(payload):
    body = {key: value for key, value in payload.items()
            if key not in {"status", "validation"}}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def variant_nodes(payload, variant):
    nodes = deepcopy(variant.get("nodes", payload.get("nodes", [])))
    index = {node["name"]: node for node in nodes}
    for name, changes in variant.get("overrides", {}).items():
        if name not in index or set(changes) - {"required_state", "dependencies", "estimated_cost_s"}:
            raise ContractRejected(f"Invalid variant override: {name}")
        index[name].update(changes)
    return nodes


def prepare(path, adapter_path, host, inputs, facts, source_root, *,
            recorder=None, allow_candidate=False):
    """Reject before binding; adapter import/factory must be side-effect-free."""
    try:
        return _prepare(Path(path), Path(adapter_path), host, inputs, facts,
                        Path(source_root), recorder, allow_candidate)
    except ContractRejected:
        raise
    except Exception as exc:
        raise ContractRejected(f"Invalid startup contract: {exc}") from exc


def _prepare(path, adapter_path, host, inputs, facts, root, recorder, candidate):
    payload = json.loads(path.read_text())
    if payload.get("schema") != "igniteserve-startup-v3":
        raise ContractRejected("Unsupported schema")
    if payload.get("status") != "final" and not candidate:
        raise ContractRejected("Contract has not passed finalization")
    if not candidate and payload.get("validation", {}).get("identity") != identity(payload):
        raise ContractRejected("Finalization evidence does not match contract")
    if digest(adapter_path) != payload["adapter_sha256"]:
        raise ContractRejected("Adapter hash mismatch")
    if not payload.get("sources"):
        raise ContractRejected("Missing source compatibility evidence")
    for name, expected in payload["sources"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root.resolve()) or digest(source) != expected:
            raise ContractRejected(f"Source mismatch: {name}")
    for key, expected in payload["compatibility"].items():
        if facts.get(key) != expected:
            raise ContractRejected(f"Incompatible {key}: {facts.get(key)!r}")
    selected = [v for v in payload["variants"] if all(
        facts.get(key) == value for key, value in v["match"].items())]
    if len(selected) != 1:
        raise ContractRejected("Expected exactly one compatible variant")
    capacities = payload["resource_capacities"]
    if any(type(v) is not int or v <= 0 for v in capacities.values()):
        raise ContractRejected("Resource capacities must be positive integers")
    initial = names(payload["initial_state"], "initial_state")
    if not initial <= inputs.keys():
        raise ContractRejected(f"Missing actual inputs: {sorted(initial - inputs.keys())}")
    dag, producers = StartupDAG(), {}
    for spec in variant_nodes(payload, selected[0]):
        if not spec.get("executable", True) or not spec.get("enabled", True):
            raise ContractRejected("Unresolved/disabled nodes must be resolved offline")
        if not isinstance(spec["name"], str) or not spec["name"]:
            raise ContractRejected("Empty node name")
        if not isinstance(spec["action_id"], str) or not spec["action_id"]:
            raise ContractRejected("Invalid action_id")
        affinity = spec.get("thread_affinity")
        if affinity is not None and (not isinstance(affinity, str) or not affinity):
            raise ContractRejected("Invalid thread affinity")
        required = names(spec["required_state"], "required_state")
        produced = names(spec["produced_state"], "produced_state")
        dependencies = names(spec["dependencies"], "dependencies")
        resources = spec["resources"]
        if any(type(v) is not int or v < 0 or k not in capacities
               or v > capacities[k] for k, v in resources.items()):
            raise ContractRejected(f"Invalid resource demand: {spec['name']}")
        cost = spec["estimated_cost_s"]
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ContractRejected("Invalid cost")
        routes = spec.get("routes", [])
        if not isinstance(routes, list):
            raise ContractRejected("Routes must be a list")
        route_names = set()
        for route in routes:
            if set(route) != {"name", "when", "action_id", "resources",
                              "estimated_cost_s", "thread_affinity"}:
                raise ContractRejected("Routes may not change dependencies or outputs")
            if (not isinstance(route["name"], str) or not route["name"]
                    or route["name"] in route_names):
                raise ContractRejected("Invalid route name")
            route_names.add(route["name"])
            when = route["when"]
            if (not isinstance(when, dict) or not when or not when.keys() <= required
                    or any(type(v) is not bool for v in when.values())):
                raise ContractRejected("Route predicates require declared boolean input state")
            if not isinstance(route["action_id"], str) or not route["action_id"]:
                raise ContractRejected("Invalid route action")
            demand, seconds = route["resources"], route["estimated_cost_s"]
            if (not isinstance(demand, dict) or any(
                    type(v) is not int or v < 0 or k not in capacities or v > capacities[k]
                    for k, v in demand.items())):
                raise ContractRejected("Invalid route resources")
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise ContractRejected("Invalid route cost")
            thread = route["thread_affinity"]
            if thread is not None and (not isinstance(thread, str) or not thread):
                raise ContractRejected("Invalid route affinity")
        for i, route in enumerate(routes):
            for other in routes[:i]:
                if not any(key in other["when"] and value != other["when"][key]
                           for key, value in route["when"].items()):
                    raise ContractRejected("Overlapping route predicates")
        for key in produced:
            if key in producers or key in inputs:
                raise ContractRejected(f"Ambiguous state producer: {key}")
            producers[key] = spec["name"]
        dag.add_node(StageNode(
            name=spec["name"], dependencies=dependencies,
            resources=resources, estimated_cost_s=cost,
            required_state=required,
            produced_state=produced,
            routes=routes,
            details={"action_id": spec["action_id"],
                     "thread_affinity": spec.get("thread_affinity")},
        ))
    if not dag.nodes:
        raise ContractRejected("Empty startup DAG")
    for node in dag.nodes.values():
        missing = node.required_state - producers.keys() - inputs.keys()
        if missing:
            raise ContractRejected(f"No producer for {sorted(missing)}")
    dag.topological_order()
    # Import only after the complete declarative plan has passed validation.
    module_spec = importlib.util.spec_from_file_location(
        "_igniteserve_adapter_" + payload["adapter_sha256"], adapter_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    registry = module.build_action_registry(host)
    if not isinstance(registry, Mapping):
        raise ContractRejected("Adapter did not return an action registry")
    for node in dag.nodes.values():
        node.action = registry.get(node.details["action_id"])
        if not callable(node.action):
            raise ContractRejected(f"No action for {node.name}")
        for route in node.routes:
            route["action"] = registry.get(route["action_id"])
            if not callable(route["action"]):
                raise ContractRejected(f"No action for route {node.name}.{route['name']}")
    context = StartupContext(recorder=recorder)
    for key, value in inputs.items():
        context.put(key, value)
    return dag, context, capacities, payload
