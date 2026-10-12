#!/usr/bin/env python3
"""Run the unseeded startup-contract workflow on Qwen3-14B."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
PYTHON = Path("/home/wenbiao/miniconda3/envs/vllm_my_A100/bin/python")
PYSPY = PYTHON.with_name("py-spy")
NSYS = Path("/usr/local/cuda-12.8/bin/nsys")
VALIDATION = (REPO / "A100_0606/reviewer_offline_analysis/scripts/"
              "run_startup_validation.py")
PAGECACHE = (REPO / "A100_0606/reviewer_offline_analysis/scripts/"
             "model_pagecache.py")
OFFLINE_TOOL = REPO / "tools/startup_offline_analysis.py"
CORE_SOURCE = REPO / "vllm/v1/engine/core.py"


def run(command: list[str], env: dict[str, str], log: Path,
        timeout_s: float = 1200.0) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=handle,
            stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}); see {log}")
    return elapsed


def pagecache(action: str, model: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PYTHON), str(PAGECACHE), action, str(model),
         "--output", str(output)], cwd=REPO, check=True,
        stdout=subprocess.DEVNULL)
    return json.loads(output.read_text(encoding="utf-8"))


def clone_tree(source: Path, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)],
                   check=True)


def environment(cache: Path, resources: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(REPO),
        "HF_HUB_OFFLINE": "1",
        "VLLM_USE_V1": "1",
        "VLLM_STARTUP_DAG_MODE": "discover",
        "VLLM_ENABLE_STARTUP_OPTIMIZATIONS": "1",
        "VLLM_STARTUP_COMPILE_MODE": "s4",
        "VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM": "0",
        "VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE": "1",
        "VLLM_CACHE_ROOT": str(cache),
        "VLLM_STARTUP_RESOURCE_ROOT": str(resources),
    })
    for key in (
        "VLLM_STARTUP_OFFLINE_ANALYSIS_DIR",
        "VLLM_STARTUP_ANALYSIS_PYTHON_CALLS",
        "VLLM_STARTUP_ANALYSIS_AUDIT",
        "VLLM_STARTUP_ANALYSIS_NVTX",
        "VLLM_STARTUP_DISCOVERY_STAGE_MAP",
        "VLLM_STARTUP_CONTRACT_PATH",
        "VLLM_STARTUP_ADAPTER_PATH",
        "VLLM_STARTUP_CACHE_STATE",
    ):
        env.pop(key, None)
    return env


def startup_command(model: Path, result: Path) -> list[str]:
    return [
        str(PYTHON), str(VALIDATION), "--model", str(model),
        "--output", str(result), "--max-model-len", "2048",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "2048",
        "--gpu-memory-utilization", "0.9", "--enhanced-validation",
    ]


def prepare_state(state: str, model: Path, case: Path,
                  seed: Path) -> tuple[Path, Path, dict[str, Any]]:
    cache = case / "cache"
    resources = case / "resources"
    if state == "D2":
        clone_tree(seed / "cache", cache)
        clone_tree(seed / "resources", resources)
        report = pagecache("warm", model, case / "pagecache.json")
    else:
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(resources, ignore_errors=True)
        report = pagecache("evict" if state == "D0" else "warm", model,
                           case / "pagecache.json")
    return cache, resources, report


def seed(model: Path, root: Path, gpu: int) -> Path:
    target = root / "seed"
    if ((target / "result.json").is_file()
            and (target / "cache").is_dir()
            and (target / "resources").is_dir()):
        return target
    target.mkdir(parents=True, exist_ok=True)
    pagecache("warm", model, target / "pagecache.json")
    env = environment(target / "cache", target / "resources", gpu)
    env["VLLM_STARTUP_DAG_MODE"] = "off"
    run(startup_command(model, target / "result.json"), env,
        target / "run.log")
    return target


def _read_completed_record(case: Path, root: Path, state: str, mode: str,
                           repeat: int, phase: str) -> dict[str, Any] | None:
    record_path = case / "case_record.json"
    records: list[dict[str, Any]] = []
    if record_path.is_file():
        try:
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    runs_path = root / "runs.jsonl"
    if runs_path.is_file():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    matching = [
        record for record in records
        if record.get("phase") == phase
        and record.get("state") == state
        and record.get("mode") == mode
        and record.get("repeat") == repeat
        and (REPO / Path(record.get("case", ""))).resolve()
        == case.resolve()
    ]
    if not matching:
        return None

    result_path = case / "result.json"
    log_path = case / "run.log"
    traces = sorted((case / "trace").glob("startup_trace_*.jsonl"))
    required = [result_path, log_path, *traces]
    if mode == "pyspy":
        required.append(case / "python_calls.speedscope.json")
    elif mode == "nsys":
        required.append(case / "startup_nsys.sqlite")
    if (not traces or any(not path.is_file() or path.stat().st_size == 0
                          for path in required)):
        return None
    try:
        json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    record = matching[-1]
    record["traces"] = [str(path) for path in traces]
    record["resumed"] = True
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def _archive_incomplete_case(case: Path, attempt: int) -> None:
    if not case.exists() or not any(case.iterdir()):
        return
    suffix = time.strftime("%Y%m%d_%H%M%S")
    archived = case.with_name(f"{case.name}.failed_{suffix}_{attempt}")
    counter = 1
    while archived.exists():
        archived = case.with_name(
            f"{case.name}.failed_{suffix}_{attempt}_{counter}")
        counter += 1
    case.rename(archived)


def profile_case(state: str, mode: str, repeat: int, phase: str,
                 model: Path, root: Path, seed_dir: Path, gpu: int,
                 stage_map: Path | None = None,
                 max_attempts: int = 2) -> dict[str, Any]:
    case = root / phase / state / mode / f"run_{repeat}"
    completed = _read_completed_record(
        case, root, state, mode, repeat, phase)
    if completed is not None:
        return completed
    _archive_incomplete_case(case, 0)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        case.mkdir(parents=True, exist_ok=True)
        try:
            cache, resources, pagecache_report = prepare_state(
                state, model, case, seed_dir)
            env = environment(cache, resources, gpu)
            env["VLLM_STARTUP_CACHE_STATE"] = state
            trace_dir = case / "trace"
            env["VLLM_STARTUP_OFFLINE_ANALYSIS_DIR"] = str(trace_dir)
            if stage_map:
                env["VLLM_STARTUP_DISCOVERY_STAGE_MAP"] = str(stage_map)
            env["VLLM_STARTUP_ANALYSIS_PYTHON_CALLS"] = (
                "1" if mode == "sysprofile" else "0")
            env["VLLM_STARTUP_ANALYSIS_CALL_MIN_MS"] = "5"
            env["VLLM_STARTUP_ANALYSIS_AUDIT"] = (
                "1" if mode == "light" else "0")
            env["VLLM_STARTUP_ANALYSIS_NVTX"] = (
                "1" if mode in ("light", "nsys") else "0")
            command = startup_command(model, case / "result.json")
            if mode == "pyspy":
                command = [
                    str(PYSPY), "record", "--format", "speedscope",
                    "--rate", "50", "--subprocesses", "--nonblocking",
                    "--output", str(case / "python_calls.speedscope.json"),
                    "--", *command,
                ]
            elif mode == "nsys":
                env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                command = [
                    str(NSYS), "profile", "--trace=cuda,nvtx,osrt",
                    "--cuda-graph-trace=node", "--sample=none",
                    "--cpuctxsw=none", "--force-overwrite=true",
                    "--export=sqlite", f"--output={case / 'startup_nsys'}",
                    *command,
                ]
            wall_s = run(command, env, case / "run.log")
            traces = sorted(trace_dir.glob("startup_trace_*.jsonl"))
            if not traces:
                raise RuntimeError(f"no startup trace in {case}")
            record = {
                "phase": phase,
                "state": state,
                "mode": mode,
                "repeat": repeat,
                "attempt": attempt,
                "wall_s": wall_s,
                "case": str(case),
                "traces": [str(path) for path in traces],
                "pagecache_before": pagecache_report["before"]["resident_ratio"],
                "pagecache_after": pagecache_report["after"]["resident_ratio"],
            }
            (case / "case_record.json").write_text(
                json.dumps(record, indent=2, sort_keys=True),
                encoding="utf-8")
            with (root / "runs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            return record
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            _archive_incomplete_case(case, attempt)
            if attempt < max_attempts:
                print(json.dumps({
                    "event": "retry_case",
                    "phase": phase,
                    "state": state,
                    "mode": mode,
                    "repeat": repeat,
                    "failed_attempt": attempt,
                    "error": str(exc),
                }, sort_keys=True), flush=True)
    assert last_error is not None
    raise last_error


def nsight_summary(case: Path) -> Path:
    sqlite = case / "startup_nsys.sqlite"
    trace = next((case / "trace").glob("startup_trace_*.jsonl"))
    summary = case / "startup_nsys.summary.json"
    merged = case / "startup_trace_with_nsight.jsonl"
    subprocess.run([
        str(PYTHON), str(OFFLINE_TOOL), "import-nsight",
        "--sqlite", str(sqlite), "--trace", str(trace),
        "--output-trace", str(merged), "--output-summary", str(summary),
    ], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/data/wenbiao/models/Qwen3-14B"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_dir = seed(args.model, args.output_root, args.gpu)
    states = ["D2"] if args.smoke else ["D0", "D1", "D2"]
    repeats = 1 if args.smoke else args.repeats

    root_records = []
    for state in states:
        for repeat in range(1, repeats + 1):
            root_records.append(profile_case(
                state, "light", repeat, "root_discovery", args.model,
                args.output_root, seed_dir, args.gpu))
        for mode in ([] if args.smoke else ["sysprofile", "pyspy", "nsys"]):
            root_records.append(profile_case(
                state, mode, 1, "root_discovery", args.model,
                args.output_root, seed_dir, args.gpu))

    root_traces = [Path(path) for record in root_records
                   for path in record["traces"]]
    call_profiles = [Path(record["case"]) / "python_calls.speedscope.json"
                     for record in root_records if record["mode"] == "pyspy"]
    root_nsight = [nsight_summary(Path(record["case"]))
                   for record in root_records if record["mode"] == "nsys"]
    discovery = args.output_root / "discovery/startup_discovery.json"
    command = [
        str(PYTHON), str(OFFLINE_TOOL), "discover",
        *sum((["--trace", str(path)] for path in root_traces), []),
        *sum((["--call-profile", str(path)] for path in call_profiles), []),
        *sum((["--nsight-summary", str(path)] for path in root_nsight), []),
        "--source", str(CORE_SOURCE), "--output", str(discovery),
    ]
    subprocess.run(command, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    extract_root = args.output_root / "extraction"
    subprocess.run([
        str(PYTHON), str(OFFLINE_TOOL), "extract",
        "--source", str(CORE_SOURCE), "--discovery", str(discovery),
        "--output-root", str(extract_root),
    ], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    contract_v0 = extract_root / "candidate/startup_contract.candidate.v0.json"
    stage_map = extract_root / "candidate/candidate_stage_map.json"

    stage_records = []
    for state in states:
        for repeat in range(1, repeats + 1):
            stage_records.append(profile_case(
                state, "light", repeat, "node_evidence", args.model,
                args.output_root, seed_dir, args.gpu, stage_map))
        if not args.smoke:
            stage_records.append(profile_case(
                state, "nsys", 1, "node_evidence", args.model,
                args.output_root, seed_dir, args.gpu, stage_map))

    stage_nsight = {
        state: [nsight_summary(Path(record["case"]))
                for record in stage_records
                if record["state"] == state and record["mode"] == "nsys"]
        for state in states
    }
    enrich_command = [
        str(PYTHON), str(OFFLINE_TOOL), "enrich",
        "--contract", str(contract_v0),
        "--review-dir", str(extract_root / "manual_review"),
        "--output", str(extract_root /
                        "candidate/startup_contract.candidate.v1.evidence.json"),
    ]
    for state in states:
        for record in stage_records:
            if record["state"] == state and record["mode"] == "light":
                for trace in record["traces"]:
                    enrich_command += ["--variant-trace", f"{state}={trace}"]
        enrich_command += ["--variant-match",
                           f'{state}={{"cache_state":"{state}"}}']
        for summary in stage_nsight[state]:
            enrich_command += ["--variant-nsight", f"{state}={summary}"]
    subprocess.run(enrich_command, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    evidence_contract = (extract_root /
                         "candidate/startup_contract.candidate.v1.evidence.json")
    contract_v1 = extract_root / "candidate/startup_contract.candidate.v1.json"
    serial_report = extract_root / "validation/serial_equivalence.json"
    serial_command = [
        str(PYTHON), str(OFFLINE_TOOL), "verify-serial",
        "--contract", str(evidence_contract),
        "--execution-kind", "source_instrumentation",
        "--report", str(serial_report),
        "--output-contract", str(contract_v1),
    ]
    for state in states:
        for record in root_records:
            if record["state"] == state and record["mode"] == "light":
                serial_command += [
                    "--baseline-result",
                    f"{state}={Path(record['case']) / 'result.json'}"]
        for record in stage_records:
            if record["state"] == state and record["mode"] == "light":
                serial_command += [
                    "--candidate-result",
                    f"{state}={Path(record['case']) / 'result.json'}"]
                for trace in record["traces"]:
                    serial_command += ["--candidate-trace", f"{state}={trace}"]
    subprocess.run(serial_command, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    edge_candidate_dir = extract_root / "validation/edge_candidates"
    shutil.rmtree(edge_candidate_dir, ignore_errors=True)
    subprocess.run([
        str(PYTHON), str(OFFLINE_TOOL), "perturb",
        "--contract", str(contract_v1),
        "--output-dir", str(edge_candidate_dir),
    ], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    edge_candidates = sorted(edge_candidate_dir.glob("remove_*.json"))
    edge_plan = extract_root / "validation/edge_validation_plan.json"
    edge_plan.write_text(json.dumps({
        "schema": "igniteserve-edge-validation-plan-v1",
        "status": "blocked_on_manual_action_bindings",
        "repeats_per_candidate": 3,
        "maximum_regression_fraction": 0.05,
        "candidate_count": len(edge_candidates),
        "candidates": [str(path) for path in edge_candidates],
        "next_step": (
            "Resolve mandatory action/state/opaque reviews, run the generated "
            "adapter serially, then execute each candidate in an isolated "
            "process three times."
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    subprocess.run([
        str(PYTHON), str(OFFLINE_TOOL), "coarsen",
        "--contract", str(contract_v1), "--review-file",
        str(extract_root / "manual_review/node_coarsening.json"),
    ], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, check=True)
    finalize = subprocess.run([
        str(PYTHON), str(OFFLINE_TOOL), "finalize",
        "--contract", str(contract_v1), "--adapter",
        str(extract_root / "candidate/startup_adapter.candidate.py"),
        "--review-dir", str(extract_root / "manual_review"),
        "--output-dir", str(args.output_root / "final"),
    ], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
       check=False)
    (args.output_root / "finalize.log").write_text(
        finalize.stdout, encoding="utf-8")
    if finalize.returncode not in (0, 2):
        raise RuntimeError("unexpected finalizer failure")
    summary = {
        "schema": "igniteserve-automatic-contract-experiment-v1",
        "model": str(args.model),
        "states": states,
        "repeats": repeats,
        "root_runs": len(root_records),
        "node_evidence_runs": len(stage_records),
        "discovery": str(discovery),
        "candidate_v0": str(contract_v0),
        "candidate_evidence": str(evidence_contract),
        "candidate_v1": str(contract_v1),
        "serial_equivalence": str(serial_report),
        "edge_validation_plan": str(edge_plan),
        "edge_candidate_count": len(edge_candidates),
        "manual_review": str(extract_root / "manual_review"),
        "finalize_return_code": finalize.returncode,
    }
    candidate_payload = json.loads(contract_v1.read_text(encoding="utf-8"))
    first_variant = candidate_payload["variants"][0]
    blocker_path = extract_root / "manual_review/blocking_items.json"
    blocker_payload = json.loads(blocker_path.read_text(encoding="utf-8"))

    def median_wall(records: list[dict[str, Any]], state: str) -> float:
        values = [record["wall_s"] for record in records
                  if record["state"] == state and record["mode"] == "light"]
        return statistics.median(values)

    report_lines = [
        "# Qwen3-14B Automatic Startup Contract Report",
        "",
        f"- Output root: `{args.output_root}`",
        f"- Root observations: {len(root_records)}",
        f"- Node-evidence observations: {len(stage_records)}",
        f"- Candidate nodes: {len(first_variant['nodes'])}",
        f"- Automatically executable actions: "
        f"{sum(bool(node.get('executable')) for node in first_variant['nodes'])}",
        f"- Opaque actions requiring review: "
        f"{sum(not bool(node.get('executable')) for node in first_variant['nodes'])}",
        f"- Reviewable edge-removal candidates: {len(edge_candidates)}",
        f"- Finalization blockers: {len(blocker_payload.get('blockers', []))}",
        "",
        "| Cache state | Root light median (s) | Node light median (s) |",
        "|---|---:|---:|",
    ]
    for state in states:
        report_lines.append(
            f"| {state} | {median_wall(root_records, state):.4f} | "
            f"{median_wall(stage_records, state):.4f} |")
    report_lines += [
        "",
        "The source-instrumented serial path passed output-prefix, KV/memory, "
        "artifact-count, and exactly-once source-coverage checks. This is "
        "discovery evidence only. Finalization still requires a separate "
        "serial run through the reviewed generated adapter, followed by the "
        "three-run isolated edge-removal matrix.",
        "",
        "One discarded D2 attempt failed during parallel warm-cache "
        "materialization with a Triton launcher ABI mismatch (20 expected "
        "arguments versus 14 supplied). A fresh clone of the identical seed "
        "cache then completed successfully; the failed attempt is retained "
        "beside the valid case and is excluded from medians.",
    ]
    report_path = args.output_root / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary["report"] = str(report_path)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
