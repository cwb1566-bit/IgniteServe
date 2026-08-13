#!/usr/bin/env python3
"""Populate persisted KV profile resources for explicitly selected models.

This script launches the repository's offline correctness harness once per
model. On a cache miss, vLLM runs the real memory profile and persists the
available KV-cache bytes. Existing compatible entries skip profile_run().
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="Local model directories to profile")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resource-root", default=None)
    parser.add_argument("--cache-root-base", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    harness = repo / "tools" / "validate_coldstart_correctness.py"
    resource_root = Path(args.resource_root or
                         repo / "vllm_startup_resources").resolve()
    cache_root_base = Path(args.cache_root_base or
                           repo / "benchmark_results" /
                           "kv_profile_compile_cache").resolve()
    prompt_spec = resource_root / "kv_profile" / "empty_prompts.json"
    prompt_spec.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for model in args.models:
        model_path = Path(model)
        if not (model_path / "config.json").exists():
            print(f"[skip] {model}: config.json not found", flush=True)
            continue
        label = model_path.name.replace("/", "_")
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_STARTUP_OPTIMIZATIONS": "1",
            "VLLM_ENABLE_KV_PROFILE_CACHE": "1",
            "VLLM_STARTUP_RESOURCE_ROOT": str(resource_root),
            "VLLM_STARTUP_COMPILE_MODE": "precompile_skip_profile",
            "VLLM_CACHE_ROOT": f"{cache_root_base}_{label}",
        })
        env.pop("VLLM_CACHED_AVAILABLE_KV_CACHE_BYTES", None)
        prompt_spec.write_text(
            json.dumps({
                "model": str(model_path),
                "max_tokens": 1,
                "prompts": [],
            }),
            encoding="utf-8",
        )
        cmd = [
            args.python,
            str(harness),
            "gen",
            "--prompts",
            str(prompt_spec),
            "--out",
            str(resource_root / "kv_profile" / f"{label}_probe.json"),
            "--dtype",
            args.dtype,
            "--tp",
            str(args.tp),
            "--max-tokens",
            "1",
        ]
        print(f"[run] {model_path}", flush=True)
        proc = subprocess.run(cmd, cwd=str(repo), env=env)
        if proc.returncode != 0:
            failures.append((str(model_path), proc.returncode))
            print(f"[fail] {model_path}: exit={proc.returncode}", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"[ok] {model_path}", flush=True)
    if failures:
        print("Failures:", failures, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
