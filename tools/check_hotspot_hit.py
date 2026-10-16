#!/usr/bin/env python3
"""Verify precise CUDA-graph hotspot hits for the IgniteServe e2e formal stage.

Compares the Top-K CUDA-graph shapes that the server *actually loaded* at engine
init (parsed from the server log line emitted by gpu_model_runner) against the
Top-K computed from the materialized ``cuda_graph_hits.csv`` for the *same* full
hotspot key. A match proves the formal run pre-captured exactly the shapes that
were materialized (no fallback to default sizes, no stale/mismatched CSV).

The replication of the Top-K selection mirrors gpu_model_runner exactly:
  sort rows by (hit_count, capture_priority) descending, take TOP_K, then sort
  the resulting batch_size list ascending.

Usage:
  python tools/check_hotspot_hit.py \
    --server-log benchmark_results/Qwen3-4B/<fullkey>.server.log \
    --material-csv vllm_startup_resources/cuda_graph_hotspots/<fullkey>/cuda_graph_hits.csv \
    --report benchmark_results/Qwen3-4B/<fullkey>.hitcheck.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

TOP_K = 10

# Matches: "Successfully loaded Top-10 CUDA graph shapes from CSV
#           (sorted by hit_count then capture_priority): [1, 2, 4, ...]"
_LOG_RE = re.compile(
    r"Successfully loaded Top-\d+ CUDA graph shapes from CSV[^\[]*(\[[^\]]*\])"
)


def parse_loaded_from_log(log_path: str) -> list[int] | None:
    """Return the last loaded Top-K list found in the server log, or None."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    matches = _LOG_RE.findall(text)
    if not matches:
        return None
    try:
        parsed = ast.literal_eval(matches[-1])
        return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        return None


def expected_topk_from_csv(csv_path: str, top_k: int = TOP_K,
                           max_capture_size: int = 512) -> list[int]:
    """Replicate gpu_model_runner's Top-K extraction from a hits CSV.

    Mirrors the server-side constraint: only batch sizes within the registered
    capturable cudagraph range (<= max_capture_size, i.e. <= 512) are eligible;
    larger sizes (from the eager branch) are dropped BEFORE taking Top-K, so the
    next smaller capturable sizes fill the Top-K.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "batch_size": int(row["batch_size"]),
                "hit_count": int(row["hit_count"]),
                "capture_priority": float(row.get("capture_priority", 0.0)),
            })
    capturable = [r for r in rows if r["batch_size"] <= max_capture_size]
    top = sorted(capturable,
                 key=lambda x: (x["hit_count"], x["capture_priority"]),
                 reverse=True)[:top_k]
    sizes = sorted(record["batch_size"] for record in top)
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--material-csv", required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--max-capture-size", type=int, default=512,
                    help="Drop hotspot batch sizes larger than this before "
                         "Top-K selection (mirrors server cudagraph max).")
    ap.add_argument("--report", default=None)
    ap.add_argument("--key", default=None, help="Full hotspot key (label only).")
    args = ap.parse_args()

    result: dict = {
        "key": args.key,
        "server_log": args.server_log,
        "material_csv": args.material_csv,
        "status": "unknown",
        "loaded": None,
        "expected": None,
        "fallback_detected": False,
        "notes": [],
    }

    if not Path(args.material_csv).exists():
        result["status"] = "FAIL"
        result["notes"].append("material CSV missing")
        _emit(result, args.report)
        return 1

    try:
        expected = expected_topk_from_csv(args.material_csv, args.top_k,
                                          args.max_capture_size)
        result["expected"] = expected
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["notes"].append(f"failed to parse material CSV: {exc}")
        _emit(result, args.report)
        return 1

    loaded = parse_loaded_from_log(args.server_log)
    result["loaded"] = loaded

    # Detect explicit fallback in the server log.
    log_text = Path(args.server_log).read_text(encoding="utf-8",
                                               errors="replace")
    if "Falling back to default sizes" in log_text:
        result["fallback_detected"] = True
        result["notes"].append("server log shows fallback to default sizes")

    if loaded is None:
        result["status"] = "FAIL"
        result["notes"].append(
            "no 'Successfully loaded Top-K' line found in server log")
        _emit(result, args.report)
        return 1

    if loaded == expected and not result["fallback_detected"]:
        result["status"] = "PASS"
    else:
        result["status"] = "FAIL"
        if loaded != expected:
            result["notes"].append("loaded != expected")
    _emit(result, args.report)
    return 0 if result["status"] == "PASS" else 1


def _emit(result: dict, report: str | None) -> None:
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
