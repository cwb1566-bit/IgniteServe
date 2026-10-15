"""Calibrate sequential graph adaptation; publish only verified forward costs.

The collect subprocess deliberately exercises otherwise cost-gated splits.
These admission estimates are never exported as measurements. A final fresh
process runs the actual policy with measured costs. Greedy tokens must agree;
FP16 cross-shape logprobs are bounded and equal-shape replay is checked separately.
"""
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def conservative_p90(values):
    return sorted(values)[math.ceil(.9 * len(values)) - 1]


def calibrate(args):
    from vllm.compilation.startup_assets import digest, publish_bundle, write_json
    from vllm.compilation.startup_resources import resource_dir
    from vllm.v1.worker.cudagraph_profile import MANIFEST_FILENAME
    with tempfile.TemporaryDirectory(prefix='graph-adaptation-') as temp:
        temp = Path(temp)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                   VLLM_ENABLE_STARTUP_OPTIMIZATIONS='1', VLLM_STARTUP_DAG_MODE='off',
                   VLLM_STARTUP_RESOURCE_ROOT=str(temp / 'assets'),
                   VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE='1',
                   VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS='1000000',
                   VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE='0',
                   VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE='1',
                   VLLM_ENABLE_CUDAGRAPH_SPLIT='1', VLLM_ENABLE_V1_MULTIPROCESSING='0')
        env.pop('VLLM_CUDA_GRAPH_HOTSPOT_DIR', None)
        env.pop('VLLM_CUDAGRAPH_COST_TABLE', None)
        results = {}
        costs = temp / 'cuda_graph_cost_table.csv'
        for mode in ('reference', 'split-reference', 'collect', 'padding', 'policy'):
            if mode == 'policy':
                env['VLLM_CUDAGRAPH_COST_TABLE'] = str(costs)
            occupied = subprocess.check_output([
                'nvidia-smi', f'--id={args.gpu}', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip()
            if occupied:
                raise RuntimeError(f'GPU {args.gpu} is occupied; refusing concurrent calibration')
            output = temp / f'{mode}.json'
            command = [sys.executable, __file__, '--worker', '--mode', mode,
                       '--output', str(output), '--model', args.model,
                       '--repeats', str(args.repeats), '--costs', str(costs),
                       '--max-model-len', str(args.max_model_len),
                       '--max-num-batched-tokens', str(args.max_num_batched_tokens),
                       '--max-num-seqs', str(args.max_num_seqs),
                       '--capture-sizes', args.capture_sizes]
            if args.prompts:
                command += ['--prompts', args.prompts]
            with (temp / f'{mode}.log').open('w+') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=900)
                    if code:
                        log.seek(0)
                        raise RuntimeError(log.read()[-12000:])
                finally:
                    import signal
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            results[mode] = json.loads(output.read_text())
            print(mode, 'actions:', results[mode]['splits'], flush=True)
            if mode != 'reference':
                assert results[mode]['tokens'] == results['reference']['tokens'], 'greedy token mismatch'
                assert results[mode]['manifest'] == results['reference']['manifest'], 'profile identity changed'
                differences = [abs(a - b) for a, b in zip(
                    results[mode]['token_logprobs'], results['reference']['token_logprobs'])]
                assert len(results[mode]['token_logprobs']) == len(results['reference']['token_logprobs'])
                print(mode, 'max token-logprob difference:', max(differences, default=0.), flush=True)
                # FP16 GEMM arithmetic changes with batch shape, including ordinary
                # padding. Bound this difference, then isolate replay at equal shapes.
                assert max(differences, default=0.) < args.max_logprob_error, 'cross-shape logprob tolerance exceeded'
                if mode == 'collect':
                    control = results['split-reference']
                    assert results[mode]['splits'] == control['splits'], 'split control coverage differs'
                    assert results[mode]['tokens'] == control['tokens'], 'split replay token mismatch'
                    replay_error = max(abs(a - b) for a, b in zip(
                        results[mode]['token_logprobs'], control['token_logprobs']))
                    print('same-shape split replay logprob difference:', replay_error, flush=True)
                    assert replay_error < 1e-5, 'split replay differs from equal-shape eager control'
            if mode == 'padding':
                assert sum(results['collect']['splits'].values()), 'no real split replay was exercised'
                rows = [row for phase in ('reference', 'collect', 'padding')
                        for row in results[phase]['profile']]
                with costs.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                write_json(temp / MANIFEST_FILENAME, results[mode]['manifest'])
        manifest = results['reference']['manifest']
        target = Path(args.output) if args.output else resource_dir('hotspots') / digest(manifest)
        if (target / 'cuda_graph_cost_table.csv').exists():
            raise FileExistsError(f'Profile already exists: {target}')
        from vllm.v1.worker.cudagraph_profile import ensure_cudagraph_profile_manifest
        valid, reason = ensure_cudagraph_profile_manifest(target, manifest)
        if not valid:
            raise ValueError(reason)
        from vllm.compilation.startup_assets import atomic_write
        atomic_write(target / costs.name, costs.read_bytes())
        write_json(target / MANIFEST_FILENAME, manifest)
        files = [p.name for p in target.glob('*.csv')] + [MANIFEST_FILENAME]
        publish_bundle(target, manifest, files)
        print('Verified cost profile:', target)
        print('Policy rejection of all splits is valid; calibration is not an end-to-end benchmark.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['reference', 'split-reference', 'collect', 'padding', 'policy'], default='policy')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output')
    parser.add_argument('--costs')
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-logprob-error', type=float, default=.01,
                        help='Explicit cross-shape selected-token logprob tolerance; never relaxed automatically')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-model-len', type=int, default=512)
    parser.add_argument('--max-num-batched-tokens', type=int, default=64)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--capture-sizes', default='1,2,4,8,16,32,64')
    parser.add_argument('--prompts', help='JSON list of representative text prompts')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least three calibration samples are required')
    if not 0 < args.max_logprob_error < float('inf'):
        parser.error('Logprob tolerance must be positive and finite')
    if not args.worker:
        calibrate(args)
        return
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    import torch
    from vllm import LLM, SamplingParams
    from vllm.v1.worker.cudagraph_adaptation import GraphAdapter, MatchResult, workload_key
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.forward_context import get_forward_context
    if args.mode == 'split-reference':
        # Offline control only: identical partitions and buffers, but no replay.
        import vllm.v1.worker.cudagraph_adaptation as adaptation
        from contextlib import contextmanager
        original_context = adaptation.set_forward_context
        @contextmanager
        def eager_context(*a, **kw):
            with original_context(*a, **kw):
                get_forward_context().skip_cuda_graphs = True
                yield
        adaptation.set_forward_context = eager_context

    original_execute = GPUModelRunner.execute_model
    original_plan = GraphAdapter.plan
    original_split = GraphAdapter.execute
    original_match = GraphAdapter.find_best_match
    events, counts = [], defaultdict(int)
    manifest = None

    def stamp():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def instrument(self, scheduler_output, *a, **kw):
        nonlocal manifest
        manifest = self.cuda_graph_manager.profile_manifest
        if args.mode == 'policy' and not getattr(self, '_profile_loaded', False):
            from vllm.v1.worker.cudagraph_profile import validate_cudagraph_profile_manifest
            valid, reason = validate_cudagraph_profile_manifest(Path(args.costs).parent, manifest)
            assert valid, reason
            assert self.cuda_graph_manager.adapter.cost_table._workloads, 'runtime did not load qualified profile'
            self._profile_loaded = True
        model = self.model
        runner = self
        class Proxy:
            def __getattr__(self, name):
                return getattr(model, name)
            def __call__(self, *p, **k):
                context = get_forward_context()
                if context.cudagraph_adaptation_stats is not None:
                    return model(*p, **k)
                ids = runner.input_batch.req_ids
                q = [scheduler_output.num_scheduled_tokens[i] for i in ids]
                n = len(q)
                s = runner.seq_lens_np[:n].tolist()
                f = (runner.input_batch.num_computed_tokens_cpu[:n]
                     < runner.input_batch.num_prompt_tokens[:n]).tolist()
                key = workload_key(q, s, f)
                stats = {}
                context.cudagraph_adaptation_stats = stats
                start = stamp()
                out = model(*p, **k)
                end = stamp()
                context.cudagraph_adaptation_stats = None
                if context.skip_cuda_graphs:
                    events.append(('eager', sum(q), key, 1, start, end, ()))
                elif stats.get('pieces', 0) and stats.get('pieces') == stats.get('replays'):
                    size = len(k['input_ids'])
                    events.append(('graph', size, key, 1, start, end, ()))
                    counts['padding' if size > sum(q) else 'exact'] += 1
                return out
        self.model = Proxy()
        try:
            return original_execute(self, scheduler_output, *a, **kw)
        finally:
            self.model = model

    def match(self, num_tokens, *a, **kw):
        if args.mode in ('reference', 'split-reference', 'collect'):
            return self._eager(num_tokens, 'probe-eager-reference')
        if args.mode == 'padding':
            size = next((s for s in self.captured_sizes if s >= num_tokens), None)
            if size is not None and (size - num_tokens) / num_tokens <= self.max_pad_ratio:
                return MatchResult('pad_up', num_tokens, size, graph_size=size,
                                   reason='offline-padding-measurement')
            return self._eager(num_tokens, 'probe-no-padding-candidate')
        return original_match(self, num_tokens, *a, **kw)

    def plan(self, metadata, q, s, f):
        if args.mode in ('reference', 'padding'):
            return ()
        if args.mode in ('split-reference', 'collect'):
            # Deliberately admit plans for coverage/calibration; measurements,
            # not these synthetic admission costs, are exported below.
            table = self.runner.cuda_graph_manager.adapter.cost_table
            table._workloads[('eager', sum(q), workload_key(q, s, f), 1)] = 100.
            for count in (2, 3):
                table._workloads[('split_overhead', sum(q), workload_key(q, s, f), count)] = 1.
            for i in range(len(q)):
                for j in range(i + 1, len(q) + 1):
                    key = workload_key(q[i:j], s[i:j], f[i:j])
                    for shape in self.runner.cuda_graph_manager.adapter.captured_sizes:
                        if shape >= sum(q[i:j]):
                            table._workloads[('graph', shape, key, 1)] = 1.
        return original_plan(self, metadata, q, s, f)

    def split(self, parts, metadata, q, s, **kw):
        r = self.runner
        f = (r.input_batch.num_computed_tokens_cpu[:len(q)]
             < r.input_batch.num_prompt_tokens[:len(q)]).tolist()
        phase = 'prefill' if all(f) else 'mixed' if any(f) else 'decode'
        intervals = []
        class Timer:
            def before(self, part):
                self.start = stamp()
            def after(self, part):
                end = stamp()
                key = workload_key(q[part.first_request:part.last_request],
                                   s[part.first_request:part.last_request],
                                   f[part.first_request:part.last_request])
                intervals.append((self.start, end))
                events.append(('graph', part.graph_size, key, 1, self.start, end, ()))
        start = stamp()
        out = original_split(self, parts, metadata, q, s, profile=Timer())
        assert out is not None, 'allocation declined before split; no calibration evidence'
        end = stamp()
        # Keep the directly measured whole split, not only the sum of separately
        # profiled pieces. CUDA-event spans include GPU gaps during CPU dispatch.
        events.append(('split_forward', sum(q), workload_key(q, s, f), len(parts),
                       start, end, ()))
        events.append(('split_overhead', sum(q), workload_key(q, s, f), len(parts),
                       start, end, tuple(intervals)))
        counts[phase] += 1
        if args.mode == 'split-reference':
            self.fallbacks = 0
        elif self.fallbacks:
            raise AssertionError('Split reached an eager fallback instead of full replay')
        return out

    GPUModelRunner.execute_model = instrument
    GraphAdapter.find_best_match = match
    GraphAdapter.plan = plan
    GraphAdapter.execute = split
    engine = None
    try:
        engine = LLM(model=args.model, dtype='float16',
                     max_model_len=args.max_model_len,
                     max_num_batched_tokens=args.max_num_batched_tokens,
                     max_num_seqs=args.max_num_seqs,
                     enable_chunked_prefill=True, gpu_memory_utilization=.45,
                     seed=123, compilation_config={'cudagraph_capture_sizes':
                                                  [int(x) for x in args.capture_sizes.split(',')]})
        answers = []
        token_logprobs = []
        for _ in range(args.repeats):
            prompts = ['The capital of France is', 'Two plus two equals',
                       'The color of the sky is',
                       'Write a short definition of a computer.',
                       'Explain the purpose of a compiler in one sentence.',
                       ' '.join(['Describe memory.'] * 25),
                       ' '.join(['Describe a computer.'] * 28)]
            if args.prompts:
                prompts = json.loads(Path(args.prompts).read_text())
            outputs = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=12,
                                                               ignore_eos=True, logprobs=1), use_tqdm=False)
            answers.append([o.outputs[0].token_ids for o in outputs])
            for o in outputs:
                result = o.outputs[0]
                token_logprobs.extend(row[token].logprob for token, row in
                                     zip(result.token_ids, result.logprobs))
        torch.cuda.synchronize()
        rows = defaultdict(list)
        for mode, shape, key, parts, start, end, inner in events:
            value = start.elapsed_time(end) - sum(a.elapsed_time(b) for a, b in inner)
            rows[(mode, shape, key, parts)].append(max(.000001, value))
        output = Path(args.output)
        output.write_text(json.dumps({'tokens': answers, 'splits': counts,
                                     'token_logprobs': token_logprobs,
                                     'profile': [dict(mode=k[0], execution_shape=k[1], workload=k[2],
                                                      parts=k[3], samples=len(v), scope='forward',
                                                      mean_gpu_ms=statistics.mean(v),
                                                      p50_gpu_ms=statistics.median(v),
                                                      p90_gpu_ms=conservative_p90(v))
                                                 for k, v in rows.items()],
                                     'manifest': manifest}, indent=2))
        print('PROBE', args.mode, dict(counts), 'cost keys', len(rows), flush=True)
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
