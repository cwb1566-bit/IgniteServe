# SPDX-License-Identifier: Apache-2.0
"""Real TP/PP requests: eager reference, background capture, and actual replay.

Instrumentation is confined to this test worker extension, not serving code.
Run only on explicitly selected free GPUs. This is not a performance test.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


class CaptureProbe:
    def capture_probe_install(self, phase):
        import torch
        from vllm.v1.worker.CudaGraphManager import RuntimeDecision
        manager = self.model_runner.cuda_graph_manager
        assert manager.capture.enabled, manager.capture.disabled_reason
        manager.capture_phases = frozenset({phase})
        self._probe_baseline = True
        self._probe_replays = 0
        select = manager.select
        replay = torch.cuda.CUDAGraph.replay

        def select_probe(phase, raw, *a, **kw):
            if self._probe_baseline:
                r = self.model_runner
                target = (r.vllm_config.pad_for_cudagraph(raw)
                          if raw <= r.cudagraph_batch_sizes[-1] else raw)
                return RuntimeDecision(phase, raw, target, target, True, False, 'test-eager'), ()
            return select(phase, raw, *a, **kw)

        def replay_probe(graph):
            backends = manager.registry.backends()
            if backends and any(backends[0].concrete_size_entries[s].cudagraph is graph
                                for s in manager.registry.dynamic_shapes):
                self._probe_replays += 1
            return replay(graph)

        manager.select = select_probe
        torch.cuda.CUDAGraph.replay = replay_probe

    def capture_probe_enable(self):
        import torch
        self._probe_memory_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self._probe_baseline = False

    def capture_probe_results(self):
        import torch
        manager = self.model_runner.cuda_graph_manager
        return {'rank': self.rank, 'replays': self._probe_replays,
                'allocated_before_bytes': self._probe_memory_before,
                'allocated_after_bytes': torch.cuda.memory_allocated(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'reserved_after_bytes': torch.cuda.memory_reserved(),
                'shadow_checks': getattr(self.model_runner, 'capture_shadow_checks', 0),
                'published': sorted(manager.registry.dynamic_shapes),
                'failures': [r.failure for r in manager.snapshot() if r.failure]}


# Imported by spawned workers before GPUModelRunner construction.
if os.environ.get('IGNITESERVE_DISTRIBUTED_CAPTURE_PROBE') == '1':
    from vllm.v1.worker.CudaGraphManager import CUDAGraphManager
    _precapture = CUDAGraphManager.precapture

    def _minimal_precapture(manager):
        sizes = manager.runner.cudagraph_batch_sizes
        manager.runner.cudagraph_batch_sizes = [1, 2]
        try:
            _precapture(manager)
        finally:
            manager.runner.cudagraph_batch_sizes = sizes

    CUDAGraphManager.precapture = _minimal_precapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--phase', choices=('decode', 'prefill', 'mixed'), default='mixed')
    args = parser.parse_args()
    if args.tp * args.pp != len(args.gpus.split(',')):
        parser.error('GPU count must equal TP * PP')
    for gpu in args.gpus.split(','):
        if subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError(f'GPU {gpu} is occupied')
    root = Path(__file__).resolve().parents[1]
    contract = root / 'vllm/v1/engine/startup_contract.distributed.json'
    candidate = json.loads(contract.read_text())['status'] == 'candidate'
    with tempfile.TemporaryDirectory(prefix='distributed-capture-') as tmp:
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpus, VLLM_WORKER_MULTIPROC_METHOD='spawn',
            VLLM_ENABLE_STARTUP_OPTIMIZATIONS='1', VLLM_STARTUP_DAG_MODE='execute',
            VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION='1' if candidate else '0',
            VLLM_STARTUP_CONTRACT_PATH=str(contract),
            VLLM_ENABLE_ADAPTIVE_CUDAGRAPH='1', VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS='1',
            VLLM_ENABLE_DISTRIBUTED_BACKGROUND_CAPTURE='1',
            VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE='1',
            VLLM_CUDAGRAPH_DYNAMIC_CROSS_PHASE='1', VLLM_ENABLE_CUDAGRAPH_SPLIT='0',
            VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE='1',
            VLLM_STARTUP_RESOURCE_ROOT=tmp, TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS='2',
            IGNITESERVE_DISTRIBUTED_CAPTURE_PROBE='1')
        from vllm import LLM, SamplingParams
        llm = None
        try:
            llm = LLM(model=args.model, tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
                disable_custom_all_reduce=args.tp > 1,
                worker_extension_cls='tools.verify_distributed_capture.CaptureProbe',
                dtype='float16', max_model_len=512, max_num_seqs=8, max_num_batched_tokens=64,
                gpu_memory_utilization=0.45, enable_chunked_prefill=True,
                enable_prefix_caching=False, seed=123,
                compilation_config={'cudagraph_capture_sizes': [1, 2, 4] + list(range(8, 65, 8))})
            llm.collective_rpc('capture_probe_install', args=(args.phase,))
            base = llm.get_tokenizer().encode('Explain parallel computing.', add_special_tokens=False)
            prompts = [{'prompt_token_ids': (base * (n // len(base) + 1))[:n]}
                       for n in (6, 6, 6, 12, 14, 36, 36, 36)]
            params = SamplingParams(temperature=0, max_tokens=96, ignore_eos=True, logprobs=1)
            reference = llm.generate(prompts, params, use_tqdm=False)
            repeated = llm.generate(prompts, params, use_tqdm=False)
            assert [o.outputs[0].token_ids for o in repeated] == [
                o.outputs[0].token_ids for o in reference], 'Eager reference is not repeatable'
            llm.collective_rpc('capture_probe_enable')
            max_error = 0.
            for _ in range(4):
                outputs = llm.generate(prompts, params, use_tqdm=False)
                for actual, expected in zip(outputs, reference):
                    a, e = actual.outputs[0], expected.outputs[0]
                    assert a.token_ids == e.token_ids, ('Token mismatch', next(
                        (i, x, y) for i, (x, y) in enumerate(zip(a.token_ids, e.token_ids)) if x != y))
                    errors = [abs(x[t].logprob-y[t].logprob)
                              for t, x, y in zip(a.token_ids, a.logprobs, e.logprobs)]
                    assert all(math.isfinite(value) for value in errors), errors
                    max_error = max(max_error, max(errors))
            results = llm.collective_rpc('capture_probe_results')
            assert all(r['replays'] > 0 and r['published'] and not r['failures'] for r in results), results
            if os.environ.get('VLLM_MIXED_CAPTURE_SHADOW_VALIDATE') == '1':
                assert all(r['shadow_checks'] > 0 for r in results), results
            # Cross-run pipeline scheduling can change kernel shapes. Report
            # its log-prob difference; numerical acceptance uses same-step
            # shadow assertions, not an arbitrary cross-run error threshold.
            print('DISTRIBUTED_CAPTURE_VERIFIED', json.dumps({'ranks': results, 'max_logprob_error': max_error}), flush=True)
        finally:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
