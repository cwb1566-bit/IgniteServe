# SPDX-License-Identifier: Apache-2.0
"""Load generated startup contracts without serializing Python actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from vllm.v1.engine.startup_orchestrator import StartupDAG


def apply_startup_contract(
    dag: "StartupDAG",
    contract_path: Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Overlay dependencies, resources, and costs while preserving actions."""
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "igniteserve-startup-contract-v1":
        raise ValueError(f"Unsupported startup contract: {contract_path}")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Startup contract has no node list")
    contract_names = {str(node.get("name")) for node in nodes}
    unknown = sorted(name for name in contract_names if name not in dag.nodes)
    if strict and unknown:
        raise ValueError(f"Contract contains unknown stages: {unknown}")
    applied: list[str] = []
    for spec in nodes:
        name = str(spec.get("name", ""))
        if name not in dag.nodes:
            continue
        node = dag.nodes[name]
        if "dependencies" in spec:
            dependencies = {str(item) for item in spec["dependencies"]}
            missing_dependencies = dependencies - set(dag.nodes)
            if strict and missing_dependencies:
                raise ValueError(
                    f"Contract stage {name} has unknown dependencies: "
                    f"{sorted(missing_dependencies)}")
            node.dependencies = dependencies & set(dag.nodes)
        if "resources" in spec:
            node.resources = {
                str(key): int(value)
                for key, value in dict(spec["resources"]).items()
            }
        if "estimated_cost_s" in spec:
            node.estimated_cost_s = float(spec["estimated_cost_s"])
        details = dict(node.details)
        details["offline_contract"] = {
            "confidence": float(spec.get("confidence", 0.0)),
            "required_state": list(spec.get("required_state", [])),
            "produced_state": list(spec.get("produced_state", [])),
        }
        node.details = details
        applied.append(name)
    dag.topological_order()
    return {"applied": sorted(applied), "unknown": unknown}
