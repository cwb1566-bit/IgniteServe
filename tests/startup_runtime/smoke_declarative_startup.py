"""Run in an isolated process/GPU; compare JSON outputs across startup modes."""
import argparse
import json
from pathlib import Path


def main():
    from vllm import LLM, SamplingParams
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compile-cache-dir")
    args = parser.parse_args()
    llm = None
    try:
        options = ({"compilation_config": {"cache_dir": args.compile_cache_dir}}
                   if args.compile_cache_dir else {})
        llm = LLM(model=args.model, max_model_len=2048,
                  max_num_batched_tokens=512, max_num_seqs=16,
                  gpu_memory_utilization=0.5, seed=123, dtype="float16", **options)
        result = llm.generate(["The capital of France is", "Explain a startup DAG in one sentence:"],
                              SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
        Path(args.output).write_text(json.dumps([
            {"tokens": r.outputs[0].token_ids, "text": r.outputs[0].text}
            for r in result], indent=2))
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
