"""Isolated real-request calibration and online composite-state acceptance.

Calibration forces admission only; policy validation uses measured prices.
No live KV writes are repeated for shadow checks. Separate identical requests
compare whole eager, equal-partition eager and composite replay.
"""
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--batch', type=int, choices=(7, 23), default=7)
parser.add_argument('--cycles', type=int, default=4)
parser.add_argument('--publish', action='store_true')
parser.add_argument('--policy-only', action='store_true')
parser.add_argument('--costs', help='Qualified CSV from an earlier independent calibration')
parser.add_argument('--profile-window', action='store_true')
args = parser.parse_args()
if args.cycles < 4:
    parser.error('Use at least one warm cycle and three measured cycles')
if args.policy_only and not args.costs:
    parser.error('--policy-only requires --costs')
if args.costs:
    os.environ['VLLM_CUDAGRAPH_COST_TABLE'] = args.costs
if subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
    raise RuntimeError('Selected GPU is occupied')
os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),
    VLLM_ENABLE_STARTUP_OPTIMIZATIONS='1', VLLM_STARTUP_DAG_MODE='off',
    VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE='1',
    VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS='1000000',
    VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE='1',
    VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE='1', VLLM_ENABLE_CUDAGRAPH_SPLIT='1')

import torch
from vllm import LLM, SamplingParams
from vllm.forward_context import set_forward_context
from vllm.v1.worker.cudagraph_adaptation import (
    GraphAdapter, GraphPartition, workload_key, composite_workload)
from vllm.v1.worker.composite_cudagraph import partition_metadata, contract, CompositeGraph
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

B = args.batch
sizes = [4, 4] if B == 7 else [16, 8]
boundary = sizes[0]
parts = (GraphPartition(0, boundary, 0, boundary, sizes[0]),
         GraphPartition(boundary, B, boundary, B, sizes[1]))
original_precapture = GPUModelRunner.pre_capture_model
original_plan = GraphAdapter.plan
original_execute = GraphAdapter.execute
mode = 'eager'
cycle = 0
records, states, generations = [], [], []
runners = []

def precapture(self):
    registered = self.cudagraph_batch_sizes
    self.cudagraph_batch_sizes = [s for s in registered if s <= max(sizes)]
    try:
        return original_precapture(self)
    finally:
        self.cudagraph_batch_sizes = registered
        runners.append(self)

def plan(self, metadata, q, s, f):
    if q != [1] * B or any(f):
        return ()
    if mode == 'policy':
        return original_plan(self, metadata, q, s, f)
    self._composite_parts = parts
    return parts

def stamp():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e

def execute(self, selected, metadata, q, s, **kw):
    r = self.runner
    begin = stamp()
    started = time.perf_counter()
    used = mode
    if mode == 'eager':
        with set_forward_context(metadata, r.vllm_config, num_tokens=B, skip_cuda_graphs=True):
            out = r.model(input_ids=r.input_ids[:B], positions=r.positions[:B],
                          intermediate_tensors=None, inputs_embeds=None)
    elif mode == 'split_reference':
        lanes = partition_metadata(metadata, selected, q, s, r.model_config.max_model_len)
        p = self.composite.plans[contract(selected, lanes)]
        assert p.ready.is_set() and not p.failed
        p.stage(lanes)
        for part, lane in zip(selected, p.lanes):
            with set_forward_context(lane['metadata'], r.vllm_config,
                    num_tokens=part.graph_size, skip_cuda_graphs=True):
                value = r.model(input_ids=lane['inputs'], positions=lane['positions'],
                                intermediate_tensors=None, inputs_embeds=None)
                p.output[part.first_token:part.last_token].copy_(value[:part.last_token-part.first_token])
        out = p.output
    else:
        before = self.composite.replays if self.composite else 0
        out = original_execute(self, selected, metadata, q, s, **kw)
        if out is None:
            used = 'pending_eager'
            with set_forward_context(metadata, r.vllm_config, num_tokens=B, skip_cuda_graphs=True):
                out = r.model(input_ids=r.input_ids[:B], positions=r.positions[:B],
                              intermediate_tensors=None, inputs_embeds=None)
        else:
            assert self.composite.replays == before + 1
            if mode == 'policy':
                target = next(n for n in r.cudagraph_batch_sizes if n >= B)
                state = next(x.state.value for x in r.adaptive_cudagraph_controller.snapshot()
                             if x.phase == 'decode' and x.shape == target)
                states.append(state)
    submit_ms = (time.perf_counter() - started) * 1000
    end = stamp()
    records.append((cycle, used, workload_key(q, s, [False]*B),
                    composite_workload(selected, q, s, [False]*B),
                    begin, end, submit_ms))
    return out

GPUModelRunner.pre_capture_model = precapture
GraphAdapter.plan = plan
GraphAdapter.execute = execute
llm = None
try:
    llm = LLM(model=args.model, dtype='float16', max_model_len=512,
        max_num_seqs=32, max_num_batched_tokens=256, gpu_memory_utilization=.45,
        enable_prefix_caching=False, seed=123,
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]})
    r = runners[0]
    assert r.concurrent_piecewise_cudagraph_manager.enabled
    variants = ['The capital of France is', 'The capital of China is',
                'The capital of Japan is', 'The capital of Italy is']
    lengths = [len(llm.get_tokenizer().encode(text)) for text in variants]
    assert len(set(lengths)) == 1, lengths
    prompts = [variants[i % len(variants)] for i in range(B)]
    params = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True, logprobs=1)
    if args.policy_only:
        from vllm.v1.worker.cudagraph_profile import validate_cudagraph_profile_manifest
        valid, reason = validate_cudagraph_profile_manifest(Path(args.costs).parent, r.cudagraph_profile_manifest)
        assert valid, reason
        assert r.graphMatcher.cost_table._workloads
    for cycle in range(1 if args.policy_only else args.cycles):
        for mode in (('eager',) if args.policy_only else ('eager', 'composite', 'split_reference')):
            start = time.perf_counter()
            answer = llm.generate(prompts, params, use_tqdm=False)
            generations.append((mode, answer, time.perf_counter()-start))
    torch.cuda.synchronize()
    grouped = defaultdict(list)
    for c, name, eager_key, composite_key, start, end, submit in records:
        if c > 0 and name in ('eager', 'composite'):
            grouped[name, eager_key if name == 'eager' else composite_key].append(start.elapsed_time(end))
    table = r.graphMatcher.cost_table
    if not args.policy_only:
        table._workloads.clear()
    rows = []
    for (name, key), values in grouped.items():
        if len(values) < 3:
            continue
        p50 = statistics.median(values)
        p90 = sorted(values)[math.ceil(.9*len(values))-1]
        count = 1 if name == 'eager' else 2
        table._workloads[name, B, key, count] = p50 if name == 'eager' else p90
        rows.append(dict(mode=name, execution_shape=B, workload=key, parts=count,
                         samples=len(values), scope='forward', mean_gpu_ms=statistics.mean(values),
                         p50_gpu_ms=p50, p90_gpu_ms=p90))
    if not args.policy_only:
        assert rows and r.graph_adapter.composite.replays > 0
    # An already available composite spans both requested online states. The
    # controller is reset, not delayed; ordinary whole-shape capture runs normally.
    r.adaptive_cudagraph_controller.reset()
    r.adaptive_cudagraph_controller.min_hits = 24
    r.adaptive_cudagraph_controller.capture_phases = frozenset({'decode'})
    mode = 'policy'
    if args.profile_window:
        original_replay = CompositeGraph.replay
        def traced_replay(self, metadata):
            with torch.cuda.nvtx.range('online_composite.replay'):
                return original_replay(self, metadata)
        CompositeGraph.replay = traced_replay
        torch.cuda.profiler.start()
    start = time.perf_counter()
    answer = llm.generate(prompts, params, use_tqdm=False)
    generations.append((mode, answer, time.perf_counter()-start))
    if args.profile_window:
        torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    reference = generations[0][1]
    def tokens(outputs):
        return [o.outputs[0].token_ids for o in outputs]
    assert all(tokens(out) == tokens(reference) for _, out, _ in generations)
    def probs(outputs):
        return [p[t].logprob for o in outputs for t,p in zip(
            o.outputs[0].token_ids, o.outputs[0].logprobs)]
    cross_error = max(abs(a-b) for _,out,_ in generations for a,b in zip(probs(reference),probs(out)))
    assert cross_error < .05, cross_error
    # The first generation may contain whole-eager pending requests. Check warmed generations separately.
    same_error = None
    if not args.policy_only:
        equal_reference = probs(next(out for name,out,_ in generations if name == 'split_reference'))
        warmed_composite = [out for name,out,_ in generations if name == 'composite'][1:]
        same_error = max(abs(a-b) for out in warmed_composite for a,b in zip(equal_reference,probs(out)))
        assert same_error < 1e-5, same_error
    target = next(n for n in r.cudagraph_batch_sizes if n >= B)
    assert target in r.cuda_graph_captured
    timings = {}
    for name in ('eager', 'composite', 'split_reference', 'policy', 'pending_eager'):
        values = [a.elapsed_time(b) for c,n,_,_,a,b,_ in records if n == name and (c > 0 or args.policy_only or name == 'pending_eager')]
        if values:
            timings[name] = dict(n=len(values), p50_ms=statistics.median(values),
                                p99_ms=sorted(values)[math.ceil(.99*len(values))-1])
    result = dict(batch=B, timings=timings, states=states, tokens_equal=True,
                  cross_shape_logprob_error=cross_error, same_shape_logprob_error=same_error,
                  builds=[p.build_seconds for p in r.graph_adapter.composite.plans.values()],
                  replays=r.graph_adapter.composite.replays,
                  generation_seconds=[(name, seconds) for name,_,seconds in generations],
                  scope='forward incl staging/merge; excluding logits/sampling',
                  one_native_launch=True)
    print('ONLINE_COMPOSITE_RESULT', json.dumps(result), flush=True)
    assert 'observed' in states, states
    assert 'capturing' in states or 'queued' in states, states
    if args.publish and not args.policy_only:
        from vllm.compilation.startup_assets import digest, atomic_write, write_json, publish_bundle
        from vllm.compilation.startup_resources import resource_dir
        from vllm.v1.worker.cudagraph_profile import MANIFEST_FILENAME
        import io
        manifest = r.cudagraph_profile_manifest
        target_dir = resource_dir('hotspots') / digest(manifest)
        if (target_dir / 'cuda_graph_cost_table.csv').exists():
            from vllm.v1.worker.cudagraph_profile import validate_cudagraph_profile_manifest
            valid, reason = validate_cudagraph_profile_manifest(target_dir, manifest)
            assert valid, reason
            with (target_dir / 'cuda_graph_cost_table.csv').open() as existing:
                old = list(csv.DictReader(existing))
            rows = old + rows
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(target_dir / 'cuda_graph_cost_table.csv', buf.getvalue().encode())
        write_json(target_dir / MANIFEST_FILENAME, manifest)
        publish_bundle(target_dir, manifest, [p.name for p in target_dir.glob('*.csv')]
                       + [MANIFEST_FILENAME])
        print('Published measured profile', target_dir, flush=True)
finally:
    if llm is not None:
        llm.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
