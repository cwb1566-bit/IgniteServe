# SPDX-License-Identifier: Apache-2.0
"""Manual-review gate and final artifact generation for startup contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional


_DECIDED = {"accepted", "resolved", "rejected", "waived"}


class FinalizationBlocked(RuntimeError):
    def __init__(self, blockers: list[dict[str, Any]], report: Path) -> None:
        super().__init__(
            f"startup contract finalization blocked by {len(blockers)} item(s); "
            f"see {report}")
        self.blockers = blockers
        self.report = report


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_node_coarsening_suggestions(
    contract: dict[str, Any],
    *,
    maximum_combined_cost_s: float = 0.05,
) -> list[dict[str, Any]]:
    """Propose only serial, same-resource pairs; never mutate the contract."""
    suggestions = []
    for variant in contract.get("variants", []):
        nodes = {str(item["name"]): item for item in variant.get("nodes", [])}
        children: dict[str, set[str]] = {name: set() for name in nodes}
        for name, node in nodes.items():
            for dependency in node.get("dependencies", []):
                if str(dependency) in children:
                    children[str(dependency)].add(name)
        consumers: dict[str, set[str]] = {}
        for name, node in nodes.items():
            for state in node.get("required_state", []):
                consumers.setdefault(str(state), set()).add(name)
        for left, left_node in nodes.items():
            if len(children[left]) != 1:
                continue
            right = next(iter(children[left]))
            right_node = nodes[right]
            if set(right_node.get("dependencies", [])) != {left}:
                continue
            if dict(left_node.get("resources", {})) != dict(
                    right_node.get("resources", {})):
                continue
            combined_cost = float(left_node.get("estimated_cost_s", 0.0)) + float(
                right_node.get("estimated_cost_s", 0.0))
            if combined_cost > maximum_combined_cost_s:
                continue
            produced = set(str(item) for item in
                           left_node.get("produced_state", []))
            if any(consumers.get(state, set()) - {right} for state in produced):
                continue
            suggestions.append({
                "id": f"coarsen:{variant.get('name')}:{left}+{right}",
                "mandatory": False,
                "status": "pending",
                "variant": variant.get("name"),
                "nodes": [left, right],
                "combined_cost_s": combined_cost,
                "reason": "strict serial pair with equal resources and no external intermediate consumer",
            })
    return suggestions


def _review_files(review_dir: Path) -> Iterable[Path]:
    expected = (
        "action_bindings.json",
        "state_aliases.json",
        "opaque_constraints.json",
        "failed_edge_tests.json",
        "node_coarsening.json",
    )
    for name in expected:
        path = review_dir / name
        if path.exists():
            yield path


def finalize_startup_contract(
    candidate_contract: Path,
    candidate_adapter: Path,
    review_dir: Path,
    output_dir: Path,
    *,
    compatibility_override: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    """Emit final artifacts only after every mandatory item is decided."""
    candidate_contract = Path(candidate_contract)
    candidate_adapter = Path(candidate_adapter)
    review_dir = Path(review_dir)
    output_dir = Path(output_dir)
    contract = _read(candidate_contract)
    if contract.get("schema") != "igniteserve-startup-contract-v2":
        raise ValueError("finalize requires a v2 startup contract")
    compile(candidate_adapter.read_text(encoding="utf-8"),
            str(candidate_adapter), "exec")

    reviews = []
    blockers: list[dict[str, Any]] = []
    action_decisions: dict[str, dict[str, Any]] = {}
    for path in _review_files(review_dir):
        payload = _read(path)
        for item in payload.get("items", []):
            record = {"file": str(path), **dict(item)}
            reviews.append(record)
            status = str(item.get("status", "pending"))
            if bool(item.get("mandatory", False)) and status not in _DECIDED:
                blockers.append({
                    "kind": "pending_manual_review",
                    "file": str(path),
                    "item": item,
                })
            if payload.get("kind") == "action_bindings":
                action_decisions[str(item.get("node", ""))] = dict(item)

    result = copy.deepcopy(contract)
    validation = dict(result.get("validation", {}))
    serial_validation = dict(validation.get("serial_equivalence", {}))
    if serial_validation.get("status") != "passed":
        blockers.append({
            "kind": "serial_equivalence_not_passed",
            "validation": validation.get("serial_equivalence"),
        })
    elif serial_validation.get("execution_kind") != "generated_adapter":
        blockers.append({
            "kind": "generated_adapter_serial_validation_missing",
            "validation": validation.get("serial_equivalence"),
        })
    if dict(validation.get("edge_perturbations", {})).get("status") != "complete":
        blockers.append({
            "kind": "edge_validation_not_complete",
            "validation": validation.get("edge_perturbations"),
        })
    for variant in result.get("variants", []):
        names = set()
        ranges = []
        for node in variant.get("nodes", []):
            name = str(node.get("name", ""))
            if not name or name in names:
                blockers.append({"kind": "invalid_node_name", "node": name})
            names.add(name)
            decision = action_decisions.get(name)
            if decision and str(decision.get("status")) in {"accepted", "resolved"}:
                if decision.get("action_id"):
                    node["action_id"] = str(decision["action_id"])
                node["executable"] = True
            if bool(node.get("enabled", True)) and not bool(
                    node.get("executable", False)):
                blockers.append({
                    "kind": "unresolved_action",
                    "variant": variant.get("name"),
                    "node": name,
                })
            source = dict(node.get("source", {}))
            if source.get("lineno") is not None:
                ranges.append((int(source["lineno"]),
                               int(source.get("end_lineno", source["lineno"])),
                               name))
            marker = f"raise UnresolvedStartupAction({name!r})"
            if bool(node.get("enabled", True)) and marker in candidate_adapter.read_text(
                    encoding="utf-8"):
                blockers.append({
                    "kind": "adapter_stub_remaining",
                    "variant": variant.get("name"),
                    "node": name,
                })
        for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
            if right[0] <= left[1]:
                blockers.append({
                    "kind": "overlapping_source_regions",
                    "left": left,
                    "right": right,
                })

    report_path = review_dir / "blocking_items.json"
    report = {
        "schema": "igniteserve-finalization-report-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidate_contract": str(candidate_contract),
        "candidate_adapter": str(candidate_adapter),
        "mandatory_review_count": sum(
            bool(item.get("mandatory", False)) for item in reviews),
        "blockers": blockers,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                           encoding="utf-8")
    if blockers:
        raise FinalizationBlocked(blockers, report_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output = output_dir / "startup_adapter.py"
    shutil.copy2(candidate_adapter, adapter_output)
    result["status"] = "final"
    if compatibility_override:
        result.setdefault("compatibility", {}).update(compatibility_override)
    result["adapter"] = {
        "path": adapter_output.name,
        "sha256": _sha256(adapter_output),
    }
    result["finalization"] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_contract": str(candidate_contract),
        "reviewed_items": len(reviews),
    }
    contract_output = output_dir / "startup_contract.json"
    contract_output.write_text(json.dumps(result, indent=2, sort_keys=True),
                               encoding="utf-8")
    # An exportable bundle is published only for a complete runtime identity.
    identity = dict(result.get("compatibility", {}))
    identity.pop("cache_state", None)
    required = {"model", "model_architecture", "dtype", "max_model_len",
                "max_num_batched_tokens", "max_num_seqs", "tensor_parallel_size",
                "pipeline_parallel_size", "data_parallel_size", "torch_version",
                "cuda_version", "gpu_compute_capability", "gpu_name", "source_sha256"}
    if required <= identity.keys() and all(identity[k] is not None for k in required):
        from vllm.compilation.startup_assets import publish_bundle
        publish_bundle(output_dir, identity, [adapter_output.name, contract_output.name])
    report["outputs"] = {
        "adapter": str(adapter_output),
        "contract": str(contract_output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                           encoding="utf-8")
    return {
        "adapter": adapter_output,
        "contract": contract_output,
        "report": report_path,
    }
