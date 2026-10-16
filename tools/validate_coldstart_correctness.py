#!/usr/bin/env python3
"""Validate cold-start startup optimizations do not change inference outputs.

Subcommands (run as separate processes so VLLM_ENABLE_STARTUP_OPTIMIZATIONS is
honored at import time):
  sample  -- build a fixed prompt set (Random + ShareGPT) -> prompts.json
  gen     -- greedy-decode prompts with offline LLM -> outputs.json
  compare -- diff two gen outputs and report exact-match / first divergence
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _sample_random(tok, num_requests, input_len, seed):
    """Mirror benchmarks/benchmark_dataset.py RandomDataset.sample (range_ratio=0)."""
    np.random.seed(seed)
    vocab_size = tok.vocab_size
    num_special = tok.num_special_tokens_to_add()
    real_input_len = input_len - num_special
    offsets = np.random.randint(0, vocab_size, size=num_requests)
    out = []
    for i in range(num_requests):
        inner = ((offsets[i] + i + np.arange(real_input_len)) % vocab_size).tolist()
        prompt = tok.decode(inner)
        re_enc = tok.encode(prompt, add_special_tokens=False)[:real_input_len]
        prompt = tok.decode(re_enc)
        out.append((prompt, len(re_enc)))
    return out


def _sample_sharegpt(path, num_requests, seed, tok, max_prompt_len):
    """Mirror benchmarks/benchmark_dataset.py ShareGPTDataset (first human turn)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data = [e for e in data
            if "conversations" in e and len(e["conversations"]) >= 2]
    random.seed(seed)
    random.shuffle(data)
    out = []
    for entry in data:
        if len(out) >= num_requests:
            break
        prompt = entry["conversations"][0]["value"]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        plen = len(tok(prompt).input_ids)
        if max_prompt_len and plen > max_prompt_len:
            continue
        out.append((prompt, plen))
    return out


def cmd_sample(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tok = _tokenizer(args.model)
    prompts: list[dict[str, Any]] = []
    if args.num_random > 0:
        for i, (p, plen) in enumerate(
                _sample_random(tok, args.num_random, args.input_len, args.seed)):
            prompts.append({"id": f"random_{i:04d}", "source": "random",
                            "prompt": p, "prompt_len": plen})
    if args.num_sharegpt > 0:
        for i, (p, plen) in enumerate(
                _sample_sharegpt(args.sharegpt_path, args.num_sharegpt,
                                 args.seed, tok, args.max_prompt_len)):
            prompts.append({"id": f"sharegpt_{i:04d}", "source": "sharegpt",
                            "prompt": p, "prompt_len": plen})
    out = {"model": args.model, "seed": args.seed, "max_tokens": args.max_tokens,
           "input_len": args.input_len, "output_len": args.output_len,
           "prompts": prompts}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    nr = sum(p["source"] == "random" for p in prompts)
    ns = sum(p["source"] == "sharegpt" for p in prompts)
    print(f"[sample] {len(prompts)} prompts (random={nr}, sharegpt={ns}) -> {args.out}")
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams
    spec = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    prompts = spec["prompts"]
    max_tokens = args.max_tokens or spec.get("max_tokens", 128)
    flag = os.environ.get("VLLM_ENABLE_STARTUP_OPTIMIZATIONS", "0")
    print(f"[gen] model={spec['model']} n={len(prompts)} max_tokens={max_tokens} "
          f"OPT={flag}", flush=True)
    llm = LLM(model=spec["model"], dtype=args.dtype, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              seed=args.seed, enforce_eager=args.enforce_eager,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                        seed=args.seed, ignore_eos=args.ignore_eos)
    results = llm.generate([p["prompt"] for p in prompts], sp)
    records = []
    for p, r in zip(prompts, results):
        o = r.outputs[0]
        records.append({"id": p["id"], "source": p["source"],
                        "output_text": o.text,
                        "token_ids": list(o.token_ids),
                        "finish_reason": o.finish_reason})
    out = {"model": spec["model"], "opt_flag": flag, "max_tokens": max_tokens,
           "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"[gen] wrote {len(records)} completions -> {args.out}")
    return 0


def _first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def cmd_compare(args: argparse.Namespace) -> int:
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    cmp = json.loads(Path(args.cmp).read_text(encoding="utf-8"))
    rmap = {r["id"]: r for r in ref["records"]}
    cmap = {r["id"]: r for r in cmp["records"]}
    ids = [i for i in rmap if i in cmap]
    per_src: dict[str, dict[str, int]] = {}
    mismatches = []
    for i in ids:
        src = rmap[i]["source"]
        st = per_src.setdefault(src, {"total": 0, "exact": 0})
        st["total"] += 1
        ra, ca = rmap[i]["token_ids"], cmap[i]["token_ids"]
        if ra == ca:
            st["exact"] += 1
        else:
            d = _first_divergence(ra, ca)
            mismatches.append({"id": i, "source": src, "first_divergence": d,
                               "ref_len": len(ra), "cmp_len": len(ca),
                               "ref_text": rmap[i]["output_text"][:200],
                               "cmp_text": cmap[i]["output_text"][:200]})
    total = len(ids)
    exact = sum(s["exact"] for s in per_src.values())
    # Divergence-position stats: where the first differing token appears.
    # Late divergence => benign float-accumulation nondeterminism.
    # Early divergence (<=4) => candidate real corruption.
    divs = sorted(m["first_divergence"] for m in mismatches
                  if m["first_divergence"] >= 0)
    early = sum(1 for d in divs if d <= 4)
    div_stats = {
        "num_diverged": len(divs),
        "early_divergence_le4": early,
        "min_divergence": divs[0] if divs else None,
        "median_divergence": divs[len(divs) // 2] if divs else None,
        "max_divergence": divs[-1] if divs else None,
    }
    summary = {"ref": args.ref, "cmp": args.cmp,
               "ref_opt": ref.get("opt_flag"), "cmp_opt": cmp.get("opt_flag"),
               "total": total, "exact_match": exact,
               "exact_match_rate": round(exact / total, 4) if total else None,
               "per_source": {k: {**v,
                   "rate": round(v["exact"] / v["total"], 4) if v["total"] else None}
                   for k, v in per_src.items()},
               "divergence_stats": div_stats,
               "num_mismatches": len(mismatches),
               "mismatches": mismatches[:args.max_show]}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
        ("ref_opt", "cmp_opt", "total", "exact_match", "exact_match_rate",
         "per_source", "divergence_stats", "num_mismatches")},
        ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--model", required=True)
    s.add_argument("--sharegpt-path", default=str(
        REPO / "benchmarks/igniteserve/ShareGPT_V3_unfiltered_cleaned_split.json"))
    s.add_argument("--num-random", type=int, default=64)
    s.add_argument("--num-sharegpt", type=int, default=64)
    s.add_argument("--input-len", type=int, default=256)
    s.add_argument("--output-len", type=int, default=128)
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--max-prompt-len", type=int, default=0,
                   help="Skip prompts longer than this many tokens (0=off)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sample)

    g = sub.add_parser("gen")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-tokens", type=int, default=0)
    g.add_argument("--dtype", default="float16")
    g.add_argument("--tp", type=int, default=1)
    g.add_argument("--gpu-mem-util", type=float, default=0.85)
    g.add_argument("--max-num-seqs", type=int, default=64)
    g.add_argument("--max-model-len", type=int, default=2048)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--ignore-eos", action="store_true")
    g.add_argument("--enforce-eager", action="store_true")
    g.set_defaults(func=cmd_gen)

    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cmp", required=True)
    c.add_argument("--out", default="")
    c.add_argument("--max-show", type=int, default=20)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
