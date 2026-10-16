import argparse
import glob
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--evict-before-rebind", action="store_true")
    args = parser.parse_args()

    common = {
        "dtype": "float16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": not args.compiled,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "disable_custom_all_reduce": args.tp > 1,
    }
    if args.compiled:
        common["compilation_config"] = {
            "level": 3,
            "cudagraph_capture_sizes": [1, 2, 4],
        }
    sampling = SamplingParams(temperature=0, max_tokens=8)

    started = time.perf_counter()
    llm = LLM(model=args.model_a, **common)
    first_init_s = time.perf_counter() - started
    output_a = llm.generate(["Hello"], sampling)[0].outputs[0].text

    if args.evict_before_rebind:
        for path in glob.glob(os.path.join(args.model_b, "*.safetensors")):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    engine_common = common.copy()
    if args.compiled:
        engine_common.pop("compilation_config")
    engine_args = EngineArgs(model=args.model_b, **engine_common)
    if args.compiled:
        engine_args.compilation_config = CompilationConfig(
            level=3, cudagraph_capture_sizes=[1, 2, 4])
    config_b = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    started = time.perf_counter()
    rebind = llm.llm_engine.rebind_model(config_b)
    rebind_wall_s = time.perf_counter() - started
    llm.default_sampling_params = None
    output_b = llm.generate(["Hello"], sampling)[0].outputs[0].text

    print(json.dumps({
        "model_a": args.model_a,
        "model_b": args.model_b,
        "compiled": args.compiled,
        "tp": args.tp,
        "pp": args.pp,
        "first_init_s": first_init_s,
        "rebind_wall_s": rebind_wall_s,
        "rebind": rebind,
        "output_a": output_a,
        "output_b": output_b,
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "1")
    main()
