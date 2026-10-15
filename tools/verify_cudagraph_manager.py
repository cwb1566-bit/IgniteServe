"""Real-request capture/replay acceptance; no repeated writes to a live KV batch.

Run each phase in a separate process on an otherwise free GPU. This is a
correctness test, not a serving-performance benchmark.
"""

import argparse
from collections import Counter
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--phase", choices=("decode", "prefill", "mixed"), required=True
    )
    args = parser.parse_args()
    if subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip():
        raise RuntimeError("Selected GPU is occupied")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        VLLM_ENABLE_STARTUP_OPTIMIZATIONS="1",
        VLLM_STARTUP_DAG_MODE="off",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_ENABLE_ADAPTIVE_CUDAGRAPH="1",
        VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE="1",
        VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS="1",
        VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE="1",
        VLLM_CUDAGRAPH_DYNAMIC_CROSS_PHASE="1",
        VLLM_ENABLE_CUDAGRAPH_SPLIT="0",
    )
    import torch
    from vllm import LLM, SamplingParams
    from vllm.v1.worker.CudaGraphManager import (
        CUDAGraphManager,
        RuntimeDecision,
    )

    original_precapture = CUDAGraphManager.precapture
    original_select = CUDAGraphManager.select
    original_replay = torch.cuda.CUDAGraph.replay
    managers, replays = [], Counter()
    baseline, current_phase = True, None

    def precapture(manager):
        registered = manager.runner.cudagraph_batch_sizes
        manager.runner.cudagraph_batch_sizes = [1, 2]
        try:
            original_precapture(manager)
        finally:
            manager.runner.cudagraph_batch_sizes = registered
        manager.capture_phases = frozenset({args.phase})
        managers.append(manager)

    def select(manager, phase, raw, *a, **kw):
        nonlocal current_phase
        current_phase = phase
        if baseline:
            return RuntimeDecision(
                phase, raw, raw, raw, True, False, "test-eager"
            ), ()
        return original_select(manager, phase, raw, *a, **kw)

    def replay(graph):
        # Count actual launches of the first piece, not mere policy admission.
        for manager in managers:
            backends = manager.registry.backends()
            if backends and any(
                backends[0].concrete_size_entries[n].cudagraph is graph
                for n in manager.registry.dynamic_shapes
            ):
                replays[current_phase] += 1
        return original_replay(graph)

    CUDAGraphManager.precapture = precapture
    CUDAGraphManager.select = select
    torch.cuda.CUDAGraph.replay = replay
    llm = None
    try:
        llm = LLM(
            model=args.model,
            dtype="float16",
            max_model_len=512,
            max_num_seqs=8,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.45,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            seed=123,
            compilation_config={
                "cudagraph_capture_sizes": [1, 2, 4] + list(range(8, 65, 8))
            },
        )
        manager = managers[0]
        assert manager.capture.enabled
        base = llm.get_tokenizer().encode(
            "Explain parallel computing.", add_special_tokens=False
        )
        prompts = [
            {"prompt_token_ids": (base * (n // len(base) + 1))[:n]}
            for n in (6, 6, 6, 12, 14, 36, 36, 36)
        ]
        params = SamplingParams(
            temperature=0, max_tokens=16, ignore_eos=True, logprobs=1
        )
        reference = llm.generate(prompts, params, use_tqdm=False)
        baseline = False
        maximum = 0.0
        for _ in range(4):
            result = llm.generate(prompts, params, use_tqdm=False)
            for actual, expected in zip(result, reference):
                a, e = actual.outputs[0], expected.outputs[0]
                assert a.token_ids == e.token_ids
                maximum = max(
                    maximum,
                    max(
                        abs(x[t].logprob - y[t].logprob)
                        for t, x, y in zip(a.token_ids, a.logprobs, e.logprobs)
                    ),
                )
            manager.capture._queue.join()
        records = manager.snapshot()
        assert not any(r.failure for r in records)
        captures = [
            r.shape
            for r in records
            if r.phase == args.phase and r.capture_attempts > 0
        ]
        assert captures and replays[args.phase] > 0, (captures, replays)
        assert maximum < 0.05, maximum
        print(
            "GRAPH_MANAGER_VERIFIED",
            json.dumps(
                dict(
                    phase=args.phase,
                    captures=captures,
                    actual_dynamic_replays=dict(replays),
                    tokens_equal=True,
                    max_logprob_error=maximum,
                )
            ),
            flush=True,
        )
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            assert manager._closed
            assert (
                manager.capture._thread is None
                or not manager.capture._thread.is_alive()
            )
        CUDAGraphManager.precapture = original_precapture
        CUDAGraphManager.select = original_select
        torch.cuda.CUDAGraph.replay = original_replay
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
