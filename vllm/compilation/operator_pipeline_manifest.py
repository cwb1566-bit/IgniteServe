# SPDX-License-Identifier: Apache-2.0
"""Export the piecewise FX dependency graph for pipeline experiments."""

import json
import os
import re
from pathlib import Path
from typing import Any

import torch.fx as fx


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "model"


def _node_value(value: Any) -> Any:
    if isinstance(value, fx.Node):
        return {"node": value.name}
    if isinstance(value, tuple):
        return [_node_value(item) for item in value]
    if isinstance(value, list):
        return [_node_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _node_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _upstream_submodules(node: fx.Node) -> list[str]:
    pending = list(node.all_input_nodes)
    visited: set[fx.Node] = set()
    upstream: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current.op == "call_module":
            upstream.add(str(current.target))
            continue
        pending.extend(current.all_input_nodes)
    return sorted(upstream)


def export_operator_pipeline_manifest(
    split_gm: fx.GraphModule,
    piecewise_graphs: list[Any],
    model_name: str,
    prefix: str,
    splitting_ops: list[str],
) -> Path | None:
    output_dir = os.environ.get("VLLM_OPERATOR_PIPELINE_MANIFEST_DIR")
    if not output_dir:
        return None

    units_by_name: dict[str, dict[str, Any]] = {}
    for ordinal, item in enumerate(piecewise_graphs):
        graph = item.graph.graph
        output_node = next(node for node in graph.nodes if node.op == "output")
        units_by_name[item.submod_name] = {
            "ordinal": ordinal,
            "name": item.submod_name,
            "graph_id": item.graph_id,
            "kind": "attention" if item.is_splitting_graph else "compiled",
            "inputs": [node.name for node in graph.nodes
                       if node.op == "placeholder"],
            "outputs": _node_value(output_node.args),
            "nodes": [{"op": node.op, "target": str(node.target)}
                      for node in graph.nodes
                      if node.op not in {"placeholder", "output"}],
            "predecessors": [],
        }

    for node in split_gm.graph.nodes:
        if node.op != "call_module" or str(node.target) not in units_by_name:
            continue
        units_by_name[str(node.target)]["predecessors"] = (
            _upstream_submodules(node))

    units = sorted(units_by_name.values(), key=lambda unit: unit["ordinal"])
    manifest = {
        "version": 1,
        "model": model_name,
        "prefix": prefix,
        "splitting_ops": list(splitting_ops),
        "units": units,
        "edges": [
            {"from": predecessor, "to": unit["name"]}
            for unit in units for predecessor in unit["predecessors"]
        ],
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (
        f"{_safe_name(model_name.split('/')[-1])}__{_safe_name(prefix)}"
        "__piecewise_dag.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path
