# SPDX-License-Identifier: Apache-2.0
"""Generate and execute isolated perturbations of candidate startup edges."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def generate_edge_removal_candidates(
    contract_path: Path,
    output_dir: Path,
    *,
    maximum_confidence: float = 0.89,
    protected_edges: Iterable[tuple[str, str]] = (),
) -> list[Path]:
    """Create one isolated contract for each reviewable conservative edge."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protected = set(protected_edges)
    candidates: list[Path] = []
    edge_details = {
        (str(edge["source"]), str(edge["target"])): edge
        for edge in contract.get("edges", [])
    }
    for node in contract.get("nodes", []):
        target = str(node.get("name", ""))
        for source in list(node.get("dependencies", [])):
            edge = (str(source), target)
            details = edge_details.get(edge, {})
            confidence = float(details.get("confidence", 0.5))
            if edge in protected or confidence > maximum_confidence:
                continue
            payload = json.loads(json.dumps(contract))
            for candidate_node in payload["nodes"]:
                if str(candidate_node.get("name")) == target:
                    candidate_node["dependencies"] = [
                        item for item in candidate_node.get("dependencies", [])
                        if str(item) != str(source)
                    ]
                    break
            payload["perturbation"] = {
                "kind": "remove_edge",
                "source": source,
                "target": target,
                "original_confidence": confidence,
            }
            digest = hashlib.sha256(
                f"{source}->{target}".encode("utf-8")).hexdigest()[:10]
            path = output_dir / f"remove_{digest}_{source}_to_{target}.json"
            _write_json(path, payload)
            candidates.append(path)
    return candidates


def run_contract_validation(
    command: Sequence[str],
    candidates: Iterable[Path],
    output_dir: Path,
    *,
    timeout_s: float,
    base_env: Optional[dict[str, str]] = None,
    expected_fingerprint: Optional[str] = None,
    isolate_compile_cache: bool = False,
) -> dict[str, Any]:
    """Run each candidate in a fresh process and compare result fingerprints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    baseline_fingerprint = expected_fingerprint
    for index, contract in enumerate(candidates):
        case_dir = output_dir / f"case_{index:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.json"
        env = os.environ.copy()
        env.update(base_env or {})
        env["VLLM_STARTUP_CONTRACT_PATH"] = str(contract)
        env["VLLM_STARTUP_VALIDATION_RESULT"] = str(result_path)
        env["VLLM_STARTUP_VALIDATION_CASE_DIR"] = str(case_dir)
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
        fingerprint = None
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            fingerprint = str(data.get("fingerprint", "")) or None
        if baseline_fingerprint is None and fingerprint and return_code == 0:
            baseline_fingerprint = fingerprint
        results.append({
            "contract": str(contract),
            "return_code": return_code,
            "timed_out": timed_out,
            "elapsed_s": elapsed,
            "fingerprint": fingerprint,
            "matches_baseline": (
                fingerprint == baseline_fingerprint
                if baseline_fingerprint and fingerprint else None),
            "log": str(case_dir / "run.log"),
        })
    report = {
        "schema": "igniteserve-startup-contract-validation-v1",
        "command": list(command),
        "baseline_fingerprint": baseline_fingerprint,
        "results": results,
        "summary": {
            "candidate_count": len(results),
            "successful": sum(item["return_code"] == 0 for item in results),
            "matching": sum(item["matches_baseline"] is True for item in results),
            "failed": sum(item["return_code"] != 0 for item in results),
            "total_elapsed_s": sum(item["elapsed_s"] for item in results),
        },
    }
    _write_json(output_dir / "validation_report.json", report)
    return report
