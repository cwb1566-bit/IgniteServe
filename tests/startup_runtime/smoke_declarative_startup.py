"""Run in an isolated process/GPU; compare JSON outputs across startup modes."""
import argparse
import json
import time
from pathlib import Path


def main():
    from vllm import LLM, SamplingParams
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compile-cache-dir")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--timing-output")
    parser.add_argument("--bounded-capture", action="store_true")
    args = parser.parse_args()
    llm = None
    try:
        options = ({"compilation_config": {"cache_dir": args.compile_cache_dir}}
                   if args.compile_cache_dir else {})
        if args.bounded_capture:
            options.setdefault("compilation_config", {})["cudagraph_capture_sizes"] = [1, 2]
        started = time.perf_counter()
        llm = LLM(model=args.model, max_model_len=512 if args.bounded_capture else 2048,
                  tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
                  disable_custom_all_reduce=args.tp > 1,
                  enforce_eager=args.eager,
                  max_num_batched_tokens=64 if args.bounded_capture else 512,
                  max_num_seqs=8 if args.bounded_capture else 16,
                  enable_chunked_prefill=True,
                  gpu_memory_utilization=0.45 if args.bounded_capture else 0.5,
                  seed=123, dtype="float16", **options)
        ready = time.perf_counter()
        result = llm.generate(["The capital of France is", "Explain a startup DAG in one sentence:"],
                              SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
        Path(args.output).write_text(json.dumps([
            {"tokens": r.outputs[0].token_ids, "text": r.outputs[0].text}
            for r in result], indent=2))
        if args.timing_output:
            Path(args.timing_output).write_text(json.dumps({
                "llm_init_s": ready - started,
                "first_generate_s": time.perf_counter() - ready}))
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
