#!/usr/bin/env python3
"""Exercise real-request warmup and concurrent piecewise CUDA graph capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

from vllm import LLM, SamplingParams


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--first-tokens", type=int, default=128)
    parser.add_argument("--second-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    init_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=64,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    init_seconds = time.perf_counter() - init_started

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.first_tokens,
        ignore_eos=True,
    )
    first_started = time.perf_counter()
    first = llm.generate(
        ["Explain why deterministic systems experiments need careful controls."],
        sampling,
        use_tqdm=False,
    )[0].outputs[0]
    first_seconds = time.perf_counter() - first_started

    # Give the serial capture worker a bounded opportunity to finish before
    # the replay-validation request. The capture itself should have overlapped
    # the still-running first request.
    time.sleep(1.0)
    second_started = time.perf_counter()
    second = llm.generate(
        ["Explain why deterministic systems experiments need careful controls."],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.second_tokens,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )[0].outputs[0]
    second_seconds = time.perf_counter() - second_started

    prefix_matches = first.text.startswith(second.text)
    result = {
        "model": args.model,
        "init_seconds": init_seconds,
        "first_generate_seconds": first_seconds,
        "second_generate_seconds": second_seconds,
        "first_token_count": len(first.token_ids),
        "second_token_count": len(second.token_ids),
        "first_digest": _digest(first.text),
        "second_digest": _digest(second.text),
        "second_is_first_prefix": prefix_matches,
    }
    print("CONCURRENT_CAPTURE_VALIDATION=" + json.dumps(result, sort_keys=True))
    if not prefix_matches:
        raise AssertionError("post-capture deterministic replay changed output")


if __name__ == "__main__":
    main()
