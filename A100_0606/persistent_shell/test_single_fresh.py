import argparse
import json
import time

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.compiled,
        trust_remote_code=True,
    )
    init_s = time.perf_counter() - started
    output = llm.generate(
        ["Hello"], SamplingParams(temperature=0, max_tokens=8)
    )[0].outputs[0].text
    print(json.dumps({
        "model": args.model,
        "compiled": args.compiled,
        "init_s": init_s,
        "output": output,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
