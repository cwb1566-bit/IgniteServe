#!/usr/bin/env python3
"""Validate a persistent engine shell across A -> B -> A model switches."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext


def evict_weights(model: Path) -> None:
    for filename in glob.glob(str(model / "*.safetensors")):
        with open(filename, "rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0,
                             os.POSIX_FADV_DONTNEED)


def clear_compile_cache(cache_root: Path) -> None:
    for name in ("torch_compile_cache", "torch_extensions"):
        path = cache_root / name
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--hotspot-root", type=Path, required=True)
    parser.add_argument("--cache-state", choices=("cold", "warm"),
                        required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    def make_compilation() -> CompilationConfig:
        return CompilationConfig(
            level=3, cudagraph_capture_sizes=[1, 2, 4])
    common = {
        "dtype": "float16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "disable_custom_all_reduce": args.tp > 1,
    }

    def make_config(model: Path):
        engine_args = EngineArgs(model=str(model), **common)
        # CompilationConfig is mutated with model-specific cache and runtime
        # fields. A general shell rebind must receive a fresh instance.
        engine_args.compilation_config = make_compilation()
        return engine_args.create_engine_config(UsageContext.LLM_CLASS)

    sampling = SamplingParams(temperature=0, max_tokens=8)
    args.hotspot_root.mkdir(parents=True, exist_ok=True)

    def select_hotspot_dir(model: Path) -> str:
        path = args.hotspot_root / model.name
        path.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_CUDA_GRAPH_HOTSPOT_DIR"] = str(path)
        return str(path)

    select_hotspot_dir(args.model_a)
    init_started = time.perf_counter()
    llm = LLM(model=str(args.model_a), compilation_config=make_compilation(),
              **common)
    init_s = time.perf_counter() - init_started
    outputs = {
        "a_before": llm.generate(["Hello"], sampling)[0].outputs[0].text
    }

    switches = []
    previous_worker_pids = None
    for label, model in (("a_to_b", args.model_b),
                         ("b_to_a", args.model_a)):
        expected_hotspot_dir = select_hotspot_dir(model)
        if args.cache_state == "cold":
            evict_weights(model)
            clear_compile_cache(args.cache_root)
        started = time.perf_counter()
        result = llm.llm_engine.rebind_model(make_config(model))
        wall_s = time.perf_counter() - started
        llm.default_sampling_params = None
        output = llm.generate(["Hello"], sampling)[0].outputs[0].text

        worker_pids = [int(item["worker_pid"])
                       for item in result["runtime_validation"]]
        old_worker_pids = [int(item["old_runtime"]["worker_pid"])
                           for item in result["worker_results"]]
        assert result["mode"] == "general_shell_rebind", result
        assert worker_pids == old_worker_pids, result
        if previous_worker_pids is not None:
            assert worker_pids == previous_worker_pids, result
        assert result["kv_s"] > 0.0, result
        assert result["cuda_graph_s"] > 0.0, result
        assert all(item["kv_cache_tensors"] > 0
                   for item in result["runtime_validation"]), result
        assert all(item["cuda_graph_shapes"]
                   for item in result["runtime_validation"]), result
        assert all(item["rebind_epoch"] == result["rebind_epoch"]
                   for item in result["runtime_validation"]), result
        assert all(item["hotspot_dir"] == expected_hotspot_dir
                   for item in result["runtime_validation"]), result

        previous_worker_pids = worker_pids
        outputs[label] = output
        switches.append({
            "label": label,
            "model": str(model),
            "wall_s": wall_s,
            "result": result,
        })

    print("GENERAL_SHELL_SWITCH_RESULT=" + json.dumps({
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "cache_state": args.cache_state,
        "tp": args.tp,
        "pp": args.pp,
        "init_s": init_s,
        "outputs": outputs,
        "switches": switches,
    }, default=str), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "1")
    os.environ.setdefault("VLLM_PERSISTENT_SHELL_REBIND_MODE", "general")
    main()
