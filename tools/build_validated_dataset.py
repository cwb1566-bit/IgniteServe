#!/usr/bin/env python3
"""Build a 2000-prompt validated dataset for the IgniteServe e2e test.

Samples real ShareGPT prompts (same extraction the benchmark clients use),
sends each candidate through the running multi_model_server chat path for BOTH
models, and keeps only prompts that succeed on *both* (HTTP 200, no error
stream, output tokens > 0). Writes the intersection (first --target passing) as
a ShareGPT-format JSON the existing clients can load directly.

Usage:
  python tools/build_validated_dataset.py \
    --base-url http://127.0.0.1:8089/v1 \
    --models /models/Qwen3-4B /models/Llama2-7B \
    --sharegpt benchmarks/igniteserve/ShareGPT_V3_unfiltered_cleaned_split.json \
    --target 2000 --out benchmarks/igniteserve/ShareGPT_validated_2000.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


def extract_candidates(sharegpt_path: str, limit: int) -> list[str]:
    """Mirror the clients' extraction: first human turn, strip, len>=10, [:1000]."""
    with open(sharegpt_path, encoding="utf-8") as f:
        data = json.load(f)
    seen: set[str] = set()
    out: list[str] = []
    for item in data:
        convs = item.get("conversations", [])
        human = next((c.get("value", "").strip() for c in convs
                      if c.get("from") == "human"), None)
        if not human or len(human) < 10:
            continue
        prompt = human[:1000]
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(prompt)
        if len(out) >= limit:
            break
    return out


async def send_one(session, base_url, model, prompt, max_tokens):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    out_tokens = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                return False, f"HTTP {resp.status}: {err[:120]}"
            async for line in resp.content:
                s = line.decode("utf-8", errors="replace").strip()
                if not s.startswith("data: ") or "[DONE]" in s:
                    continue
                body = s[6:]
                try:
                    j = json.loads(body)
                    if isinstance(j, dict) and "error" in j:
                        return False, f"ServerError: {str(j['error'])[:120]}"
                except json.JSONDecodeError:
                    pass
                out_tokens += 1
            if out_tokens == 0:
                return False, "0 output tokens"
            return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"Exc: {str(e)[:120]}"


async def validate_model(base_url, model, prompts, max_workers, max_tokens):
    """Return dict idx -> (ok, err) for all prompts on one model."""
    sem = asyncio.Semaphore(max_workers)
    results: dict[int, tuple[bool, str | None]] = {}
    done = 0
    total = len(prompts)

    async def worker(session, idx, prompt):
        nonlocal done
        async with sem:
            ok, err = await send_one(session, base_url, model, prompt, max_tokens)
        results[idx] = (ok, err)
        done += 1
        if done % 200 == 0 or done == total:
            n_ok = sum(1 for v in results.values() if v[0])
            print(f"  [{Path(model).name}] {done}/{total} ok={n_ok}", flush=True)

    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(worker(session, i, p)
                               for i, p in enumerate(prompts)))
    return results


async def main_async(args) -> int:
    candidates = extract_candidates(args.sharegpt, args.max_candidates)
    print(f"[candidates] extracted {len(candidates)} unique prompts "
          f"(cap {args.max_candidates})", flush=True)

    per_model_results: dict[str, dict[int, tuple[bool, str | None]]] = {}
    for model in args.models:
        print(f"[validate] model={model} n={len(candidates)} "
              f"workers={args.max_workers}", flush=True)
        per_model_results[model] = await validate_model(
            args.base_url, model, candidates, args.max_workers, args.max_tokens)

    # Intersection: keep candidates that pass on every model, in order.
    kept_idx: list[int] = []
    for i in range(len(candidates)):
        if all(per_model_results[m].get(i, (False, "missing"))[0]
               for m in args.models):
            kept_idx.append(i)
        if len(kept_idx) >= args.target:
            break

    kept = [candidates[i] for i in kept_idx]
    dataset = [{"id": f"validated_{k:05d}",
                "conversations": [{"from": "human", "value": p}]}
               for k, p in enumerate(kept)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dataset, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # Report: per-model pass counts and failure-reason histogram.
    report = {"candidates": len(candidates), "target": args.target,
              "kept": len(kept), "models": {}}
    for m in args.models:
        res = per_model_results[m]
        n_ok = sum(1 for v in res.values() if v[0])
        fails: dict[str, int] = {}
        for ok, err in res.values():
            if not ok:
                key = (err or "unknown").split(":")[0]
                fails[key] = fails.get(key, 0) + 1
        report["models"][Path(m).name] = {"ok": n_ok, "fail": len(res) - n_ok,
                                           "fail_reasons": fails}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] wrote {len(kept)} validated prompts -> {args.out}")
    if len(kept) < args.target:
        print(f"[WARN] only {len(kept)} < target {args.target}. "
              f"Increase --max-candidates and rerun.")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8089/v1")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sharegpt", default=(
        "benchmarks/igniteserve/ShareGPT_V3_unfiltered_cleaned_split.json"))
    ap.add_argument("--max-candidates", type=int, default=3500)
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="benchmarks/igniteserve/ShareGPT_validated_2000.json")
    ap.add_argument("--report", default=(
        "benchmark_results/ShareGPT_validated_2000_report.json"))
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
