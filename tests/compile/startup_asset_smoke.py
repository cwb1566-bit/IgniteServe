# SPDX-License-Identifier: Apache-2.0
"""Short separate-process smoke; the caller owns GPU selection and cache root."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dag")
    parser.add_argument("--memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=128)
    args = parser.parse_args()
    import os
    if args.dag is not None:
        os.environ["VLLM_STARTUP_DAG_MODE"] = args.dag
    from vllm import LLM, SamplingParams
    from vllm.compilation.startup_resources import resource_root
    started = time.perf_counter()
    llm = LLM(model=args.model, max_model_len=args.max_model_len, max_num_batched_tokens=128,
              max_num_seqs=8, gpu_memory_utilization=args.memory_utilization, seed=42,
              compilation_config={"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]})
    startup = time.perf_counter() - started
    outputs = llm.generate(["The capital of France is", "Explain a compiler in one sentence."],
                           SamplingParams(temperature=0, max_tokens=16))
    result = {"model": args.model, "dag": os.getenv("VLLM_STARTUP_DAG_MODE", "off"),
              "root": str(resource_root()),
              "startup_s": startup,
              "tokens": [o.outputs[0].token_ids for o in outputs],
              "texts": [o.outputs[0].text for o in outputs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
