# SPDX-License-Identifier: Apache-2.0
"""Generate and execute isolated perturbations of candidate startup edges."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def _variants(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if contract.get("schema") == "igniteserve-startup-contract-v2":
        return list(contract.get("variants", []))
    return [{
        "name": "legacy",
        "nodes": contract.get("nodes", []),
        "edges": contract.get("edges", []),
    }]


def generate_edge_removal_candidates(
    contract_path: Path,
    output_dir: Path,
    *,
    maximum_confidence: float = 0.89,
    protected_edges: Iterable[tuple[str, str]] = (),
    variant_name: Optional[str] = None,
) -> list[Path]:
    """Create one isolated contract for each reviewable conservative edge."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protected = set(protected_edges)
    candidates: list[Path] = []
    for variant in _variants(contract):
        current_name = str(variant.get("name", "legacy"))
        if variant_name is not None and current_name != variant_name:
            continue
        edge_details = {
            (str(edge["source"]), str(edge["target"])): edge
            for edge in variant.get("edges", [])
        }
        for node in variant.get("nodes", []):
            target = str(node.get("name", ""))
            for source in list(node.get("dependencies", [])):
                edge = (str(source), target)
                details = edge_details.get(edge, {})
                confidence = float(details.get("confidence", 0.5))
                if (edge in protected or bool(details.get("protected", False))
                        or confidence > maximum_confidence):
                    continue
                payload = json.loads(json.dumps(contract))
                selected = (_variants(payload)[0] if payload.get("schema") !=
                            "igniteserve-startup-contract-v2" else next(
                                item for item in payload["variants"]
                                if str(item.get("name")) == current_name))
                for candidate_node in selected["nodes"]:
                    if str(candidate_node.get("name")) == target:
                        candidate_node["dependencies"] = [
                            item for item in candidate_node.get("dependencies", [])
                            if str(item) != str(source)
                        ]
                        break
                selected["edges"] = [
                    item for item in selected.get("edges", [])
                    if not (str(item.get("source")) == str(source)
                            and str(item.get("target")) == target)
                ]
                payload["perturbation"] = {
                    "kind": "remove_edge",
                    "variant": current_name,
                    "source": source,
                    "target": target,
                    "original_confidence": confidence,
                }
                digest = hashlib.sha256(
                    f"{current_name}:{source}->{target}".encode("utf-8")
                ).hexdigest()[:10]
                safe_source = str(source).replace("/", "_").replace(".", "_")
                safe_target = target.replace("/", "_").replace(".", "_")
                path = output_dir / (
                    f"remove_{digest}_{safe_source}_to_{safe_target}.json")
                _write_json(path, payload)
                candidates.append(path)
    return candidates


def _load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _edge_exercised(data: dict[str, Any], perturbation: dict[str, Any]
                    ) -> Optional[bool]:
    stages = data.get("stages") or data.get("stage_results")
    if not isinstance(stages, dict):
        return None
    source = stages.get(str(perturbation.get("source")))
    target = stages.get(str(perturbation.get("target")))
    if not isinstance(source, dict) or not isinstance(target, dict):
        return None
    try:
        source_start = float(source["started_at"])
        source_finish = float(source["finished_at"])
        target_start = float(target["started_at"])
        target_finish = float(target["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return target_start < source_finish or target_finish <= source_start


def run_contract_validation(
    command: Sequence[str],
    candidates: Iterable[Path],
    output_dir: Path,
    *,
    timeout_s: float,
    base_env: Optional[dict[str, str]] = None,
    expected_fingerprint: Optional[str] = None,
    isolate_compile_cache: bool = False,
    repeats: int = 1,
    baseline_elapsed_s: Optional[Sequence[float]] = None,
    maximum_regression_fraction: float = 0.05,
) -> dict[str, Any]:
    """Run each perturbation in fresh processes and classify its safety."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repeats = max(1, int(repeats))
    results = []
    baseline_fingerprint = expected_fingerprint
    baseline_times = [float(item) for item in (baseline_elapsed_s or [])]
    for index, contract in enumerate(candidates):
        perturbation = json.loads(
            Path(contract).read_text(encoding="utf-8")).get(
                "perturbation", {})
        attempts = []
        for repeat in range(repeats):
            case_dir = output_dir / f"case_{index:04d}" / f"run_{repeat:02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            result_path = case_dir / "result.json"
            startup_report_path = case_dir / "generated_startup.json"
            env = os.environ.copy()
            env.update(base_env or {})
            env["VLLM_STARTUP_CONTRACT_PATH"] = str(contract)
            env["VLLM_STARTUP_VALIDATION_RESULT"] = str(result_path)
            env["VLLM_STARTUP_VALIDATION_CASE_DIR"] = str(case_dir)
            env["VLLM_STARTUP_GENERATED_REPORT"] = str(startup_report_path)
            if isolate_compile_cache:
                env["VLLM_CACHE_ROOT"] = str(case_dir / "cache")
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    list(command), env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=timeout_s,
                    check=False)
                return_code = completed.returncode
                output = completed.stdout
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                return_code = -1
                output = str(exc.stdout or "")
                timed_out = True
            elapsed = time.perf_counter() - started
            (case_dir / "run.log").write_text(output, encoding="utf-8")
            data = _load_result(result_path)
            startup_report = _load_result(startup_report_path)
            fingerprint = str(data.get("fingerprint", "")) or None
            if baseline_fingerprint is None and fingerprint and return_code == 0:
                baseline_fingerprint = fingerprint
            attempts.append({
                "return_code": return_code,
                "timed_out": timed_out,
                "elapsed_s": elapsed,
                "fingerprint": fingerprint,
                "matches_baseline": (
                    fingerprint == baseline_fingerprint
                    if baseline_fingerprint and fingerprint else None),
                "edge_exercised": _edge_exercised(
                    startup_report, perturbation),
                "log": str(case_dir / "run.log"),
                "result": str(result_path) if result_path.exists() else None,
                "startup_report": (
                    str(startup_report_path)
                    if startup_report_path.exists() else None),
            })
        elapsed_values = [item["elapsed_s"] for item in attempts]
        all_success = all(item["return_code"] == 0 for item in attempts)
        all_matching = all(item["matches_baseline"] is True for item in attempts)
        all_exercised = all(item["edge_exercised"] is True for item in attempts)
        regression = None
        if baseline_times:
            regression = (
                statistics.median(elapsed_values)
                > statistics.median(baseline_times)
                * (1.0 + maximum_regression_fraction)
                or max(elapsed_values) > max(baseline_times)
                * (1.0 + maximum_regression_fraction))
        if not all_success or any(
                item["matches_baseline"] is False for item in attempts):
            classification = "required"
        elif (not all_matching or not all_exercised or regression is None):
            classification = "inconclusive"
        elif regression:
            classification = "required"
        else:
            classification = "safe_remove"
        results.append({
            "contract": str(contract),
            "perturbation": perturbation,
            "classification": classification,
            "attempts": attempts,
            "median_elapsed_s": statistics.median(elapsed_values),
            "maximum_elapsed_s": max(elapsed_values),
            "performance_regression": regression,
        })
    report = {
        "schema": "igniteserve-startup-contract-validation-v2",
        "command": list(command),
        "baseline_fingerprint": baseline_fingerprint,
        "baseline_elapsed_s": baseline_times,
        "repeats": repeats,
        "maximum_regression_fraction": maximum_regression_fraction,
        "results": results,
        "summary": {
            "candidate_count": len(results),
            "safe_remove": sum(item["classification"] == "safe_remove"
                               for item in results),
            "required": sum(item["classification"] == "required"
                            for item in results),
            "inconclusive": sum(item["classification"] == "inconclusive"
                                for item in results),
            "total_elapsed_s": sum(
                attempt["elapsed_s"] for item in results
                for attempt in item["attempts"]),
        },
    }
    _write_json(output_dir / "validation_report.json", report)
    return report
