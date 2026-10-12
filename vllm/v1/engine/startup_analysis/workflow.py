# SPDX-License-Identifier: Apache-2.0
"""Offline equivalence checks and perturbation-result application."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _result_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = payload.get("artifacts") or {}
    return {
        "output_prefix": payload.get(
            "validation_output_fingerprint", payload.get("fingerprint")),
        "memory_invariant": payload.get("memory_invariant"),
        "artifact_digest": artifacts.get("digest"),
        "artifact_file_count": artifacts.get("file_count"),
    }


def _semantically_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Ignore byte hashes that embed case-local absolute paths."""
    return (
        left.get("output_prefix") == right.get("output_prefix")
        and left.get("memory_invariant") == right.get("memory_invariant")
        and left.get("artifact_file_count")
        == right.get("artifact_file_count")
    )


def verify_serial_equivalence(
    contract_path: Path,
    baseline_results: Mapping[str, Sequence[Path]],
    candidate_results: Mapping[str, Sequence[Path]],
    candidate_traces: Mapping[str, Sequence[Path]],
    output: Path,
    *,
    execution_kind: str = "source_instrumentation",
) -> dict[str, Any]:
    """Verify one serial candidate path covers all nodes without behavior drift."""
    if execution_kind not in {"source_instrumentation", "generated_adapter"}:
        raise ValueError(f"unsupported serial execution kind: {execution_kind}")
    contract = _read(contract_path)
    variant_nodes = {
        str(variant["name"]): {str(node["name"])
                              for node in variant.get("nodes", [])}
        for variant in contract.get("variants", [])
    }
    states = sorted(set(baseline_results) | set(candidate_results)
                    | set(candidate_traces))
    state_reports = []
    passed = True
    for state in states:
        baseline = [_result_fingerprint(_read(path))
                    for path in baseline_results.get(state, [])]
        candidates = [_result_fingerprint(_read(path))
                      for path in candidate_results.get(state, [])]
        expected = baseline[0] if baseline else None
        baseline_consistent = bool(baseline) and all(
            _semantically_equal(item, expected) for item in baseline)
        candidate_matches = bool(candidates) and all(
            _semantically_equal(item, expected) for item in candidates)
        artifact_byte_hashes_match = bool(candidates) and expected is not None and all(
            item.get("artifact_digest") == expected.get("artifact_digest")
            for item in candidates)
        expected_nodes = variant_nodes.get(state)
        if expected_nodes is None and len(variant_nodes) == 1:
            expected_nodes = next(iter(variant_nodes.values()))
        coverage = []
        for path in candidate_traces.get(state, []):
            counts = Counter(
                str(event.get("name")) for event in _events(path)
                if event.get("event") == "stage_end"
                and str(event.get("name")) != "engine_startup.root")
            missing = sorted((expected_nodes or set()) - set(counts))
            duplicates = sorted(name for name, count in counts.items()
                                if name in (expected_nodes or set())
                                and count != 1)
            unexpected = sorted(set(counts) - (expected_nodes or set()))
            coverage.append({
                "trace": str(path),
                "missing": missing,
                "duplicates": duplicates,
                "unexpected": unexpected,
                "passed": not missing and not duplicates and not unexpected,
            })
        coverage_passed = bool(coverage) and all(
            item["passed"] for item in coverage)
        state_passed = (baseline_consistent and candidate_matches
                        and coverage_passed)
        passed = passed and state_passed
        state_reports.append({
            "state": state,
            "baseline_count": len(baseline),
            "candidate_count": len(candidates),
            "baseline_consistent": baseline_consistent,
            "candidate_matches": candidate_matches,
            "artifact_byte_hashes_match": artifact_byte_hashes_match,
            "artifact_hash_note": (
                None if artifact_byte_hashes_match else
                "byte hashes differ; semantic file-count, output, and memory "
                "checks are used because runtime metadata embeds case-local paths"),
            "coverage": coverage,
            "passed": state_passed,
        })
    report = {
        "schema": "igniteserve-serial-equivalence-v1",
        "contract": str(contract_path),
        "execution_kind": execution_kind,
        "passed": passed,
        "states": state_reports,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True),
                      encoding="utf-8")
    return report


def apply_edge_validation_report(
    contract_path: Path,
    validation_report_path: Path,
    output: Path,
    *,
    review_file: Path,
) -> dict[str, Any]:
    """Remove only proven-safe edges and expose inconclusive cases to review."""
    contract = copy.deepcopy(_read(contract_path))
    report = _read(validation_report_path)
    if report.get("schema") != "igniteserve-startup-contract-validation-v2":
        raise ValueError("edge application requires a v2 validation report")
    variants = {str(item["name"]): item
                for item in contract.get("variants", [])}
    review_items = []
    applied = 0
    required = 0
    for result in report.get("results", []):
        perturbation = dict(result.get("perturbation", {}))
        variant_name = str(perturbation.get("variant", ""))
        source = str(perturbation.get("source", ""))
        target = str(perturbation.get("target", ""))
        classification = str(result.get("classification", "inconclusive"))
        variant = variants.get(variant_name)
        if variant is None:
            raise ValueError(f"validation references unknown variant {variant_name!r}")
        if classification == "safe_remove":
            for node in variant.get("nodes", []):
                if str(node.get("name")) == target:
                    node["dependencies"] = [
                        item for item in node.get("dependencies", [])
                        if str(item) != source]
            variant["edges"] = [
                edge for edge in variant.get("edges", [])
                if not (str(edge.get("source")) == source
                        and str(edge.get("target")) == target)]
            applied += 1
        elif classification == "required":
            for edge in variant.get("edges", []):
                if (str(edge.get("source")) == source
                        and str(edge.get("target")) == target):
                    edge["protected"] = True
                    edge["validation"] = "required"
            required += 1
        else:
            review_items.append({
                "id": f"edge:{variant_name}:{source}->{target}",
                "mandatory": True,
                "status": "pending",
                "variant": variant_name,
                "source": source,
                "target": target,
                "reason": "isolated edge test was inconclusive",
                "validation_result": result,
            })
    summary = report.get("summary", {})
    expected = int(summary.get("candidate_count", 0))
    complete = expected > 0 and expected == applied + required + len(review_items)
    contract.setdefault("validation", {})["edge_perturbations"] = {
        "status": "complete" if complete and not review_items else "review_required",
        "report": str(validation_report_path),
        "candidate_count": expected,
        "safe_remove": applied,
        "required": required,
        "inconclusive": len(review_items),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True),
                      encoding="utf-8")
    review_file = Path(review_file)
    review_file.parent.mkdir(parents=True, exist_ok=True)
    review_file.write_text(json.dumps({
        "schema": "igniteserve-manual-review-v1",
        "kind": "failed_edge_tests",
        "items": review_items,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return contract
