"""Offline diagnostic: CUDA-native fork/join, not Python threaded dispatch.

Run on an idle GPU, e.g. --model /models/Qwen3-4B --gpu 4 --batch-size 7.
Compares piecewise split, full serial/parallel split graphs, whole eager,
whole padded piecewise replay, and a new whole-forward graph. The latter three
new graph paths capture attention too: they are NOT free reuse of old graphs.
First-use allocation/warmup/capture and hidden-state shadow checks are excluded
from steady-state samples; construction costs are reported separately.
Unified sampling is untouched.

This is deliberately not a serving path: synchronous capture and same-token KV
write repetition are only suitable for this isolated, deterministic experiment.
It neither enables a production flag nor publishes a cost-table asset.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--batch-size', type=int, choices=(7, 23), default=7)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--profile-window', action='store_true',
                    help='Bracket one warmed cycle for Nsight --capture-range=cudaProfilerApi')
args = parser.parse_args()
if args.repeats < 3:
    parser.error('At least three alternating cycles are required')
occupied = subprocess.check_output([
    'nvidia-smi', f'--id={args.gpu}', '--query-compute-apps=pid',
    '--format=csv,noheader,nounits'], text=True).strip()
if occupied:
    raise RuntimeError('Selected GPU is occupied; refusing the experiment')
os.environ.update(
    CUDA_VISIBLE_DEVICES=str(args.gpu), VLLM_ENABLE_STARTUP_OPTIMIZATIONS='1',
    VLLM_STARTUP_DAG_MODE='off', VLLM_ENABLE_V1_MULTIPROCESSING='0',
    VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE='1',
    VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS='1000000',
    VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE='0',
    VLLM_ENABLE_CUDAGRAPH_SPLIT='1')

import torch
from vllm import LLM
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.worker.cudagraph_adaptation import GraphAdapter, GraphPartition

original_split = GraphAdapter.execute
original_generate = LLM.generate
original_init = LLM.__init__
batch_size = args.batch_size
variants = ('baseline', 'full_serial', 'full_parallel', 'eager', 'padding', 'full_whole')
iteration = 0
mode = 'baseline'
plans = {}
samples = []
builds = []
checks = []
warmup_cycles = 1


def initialize(self, *args, **kwargs):
    kwargs['enable_prefix_caching'] = False
    original_init(self, *args, **kwargs)


def clone_metadata(meta, maximum):
    values = {f.name: (getattr(meta, f.name).clone()
                      if isinstance(getattr(meta, f.name), torch.Tensor)
                      else getattr(meta, f.name)) for f in dataclasses.fields(meta)}
    values['max_seq_len'] = maximum
    return dataclasses.replace(meta, **values)


def split(self, parts, metadata, q, s, *, profile=None):
    if q != [1] * batch_size:
        return original_split(self, parts, metadata, q, s, profile=profile)
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    if mode == 'baseline':
        # Equal instrumentation: all variants have only the outer timing pair.
        out = original_split(self, parts, metadata, q, s, profile=None)
        end.record()
        if iteration > len(variants) * warmup_cycles:
            samples.append((mode, start, end))
        return out
    r = self.runner
    if mode in ('eager', 'padding'):
        size = (batch_size if mode == 'eager' else
                next(n for n in r.cuda_graph_manager.adapter.captured_sizes if n >= batch_size))
        r.input_ids[batch_size:size].zero_()
        r.positions[batch_size:size].zero_()
        stats = {}
        with set_forward_context(metadata, r.vllm_config, num_tokens=size,
                                 cudagraph_adaptation_stats=stats):
            get_forward_context().skip_cuda_graphs = mode == 'eager'
            out = r.model(input_ids=r.input_ids[:size], positions=r.positions[:size],
                          intermediate_tensors=None, inputs_embeds=None)
        if mode == 'padding':
            assert stats.get('pieces', 0) > 0 and stats['pieces'] == stats.get('replays')
        end.record()
        if iteration > len(variants) * warmup_cycles:
            samples.append((mode, start, end))
        return out[:batch_size]
    if mode == 'full_whole':
        parts = (GraphPartition(0, batch_size, 0, batch_size, batch_size),)
    key = (mode, tuple(parts))
    fresh = key not in plans
    maximum = r.model_config.max_model_len
    allocated_before = torch.cuda.memory_allocated()
    created = time.perf_counter()
    if fresh:
        assert self.supported(metadata), 'unsupported model/attention contract'
        lanes = []
        for part in parts:
            by_id = {}
            lane_meta = {}
            for name, meta in metadata.items():
                if id(meta) not in by_id:
                    by_id[id(meta)] = clone_metadata(meta.split_for_graph(part, q, s), maximum)
                lane_meta[name] = by_id[id(meta)]
            lanes.append(dict(inputs=torch.empty_like(r.input_ids[:part.graph_size]),
                              positions=torch.empty_like(r.positions[:part.graph_size]),
                              metadata=lane_meta, stream=torch.cuda.Stream()))
        plan = dict(lanes=lanes, merged=torch.empty((sum(q), r.hidden_size),
                    dtype=r.dtype, device=r.device), graph=torch.cuda.CUDAGraph(),
                    stream=torch.cuda.Stream())
        plans[key] = plan
    plan = plans[key]
    for part, lane in zip(parts, plan['lanes']):
        lo, hi = part.first_token, part.last_token
        actual = hi-lo
        lane['inputs'][:actual].copy_(r.input_ids[lo:hi])
        lane['positions'][:actual].copy_(r.positions[lo:hi])
        lane['inputs'][actual:].zero_()
        lane['positions'][actual:].zero_()
        seen = set()
        for name, meta in metadata.items():
            if id(meta) in seen:
                continue
            seen.add(id(meta))
            current = meta.split_for_graph(part, q, s)
            target = lane['metadata'][name]
            for f in dataclasses.fields(current):
                value = getattr(current, f.name)
                expected = getattr(target, f.name)
                if isinstance(value, torch.Tensor):
                    assert value.shape == expected.shape and value.dtype == expected.dtype
                    expected.copy_(value)
                elif f.name == 'max_seq_len':
                    assert value <= maximum
                else:
                    assert value == expected, f.name

    def forward(part, lane):
        with set_forward_context(lane['metadata'], r.vllm_config,
                                 num_tokens=part.graph_size,
                                 cudagraph_adaptation_stats={}):
            get_forward_context().skip_cuda_graphs = True
            result = r.model(input_ids=lane['inputs'], positions=lane['positions'],
                             intermediate_tensors=None, inputs_embeds=None)
            plan['merged'][part.first_token:part.last_token].copy_(
                result[:part.last_token-part.first_token])

    if fresh:
        torch.cuda.synchronize()
        began = time.perf_counter()
        for part, lane in zip(parts, plan['lanes']):
            with torch.cuda.stream(lane['stream']):
                forward(part, lane)
            lane['stream'].synchronize()
        fork = torch.cuda.Event()
        joins = [torch.cuda.Event() for _ in parts]
        with torch.cuda.graph(plan['graph'], stream=plan['stream']):
            fork.record()
            for part, lane, join in zip(parts, plan['lanes'], joins):
                if mode == 'full_parallel':
                    lane['stream'].wait_event(fork)
                    with torch.cuda.stream(lane['stream']):
                        forward(part, lane)
                        join.record()
                else:
                    forward(part, lane)
            if mode == 'full_parallel':
                for join in joins:
                    plan['stream'].wait_event(join)
        builds.append(dict(mode=mode, warmup_capture_ms=1000*(time.perf_counter()-began),
                           setup_ms=1000*(time.perf_counter()-created),
                           extra_allocated_bytes=torch.cuda.memory_allocated()-allocated_before,
                           partitions=[p.graph_size for p in parts]))
    plan['graph'].replay()
    end.record()
    if not fresh and iteration > len(variants) * warmup_cycles:
        samples.append((mode, start, end))
    # Offline validation only. Repeat the same-token KV writes, using the same
    # partition shapes without replay, and compare every hidden-state element.
    if fresh or sum(c['mode'] == mode for c in checks) < 3:
        captured = plan['merged'].clone()
        for part, lane in zip(parts, plan['lanes']):
            forward(part, lane)
        error = (captured-plan['merged']).abs().max().item()
        torch.testing.assert_close(captured, plan['merged'], atol=1e-3, rtol=1e-3)
        checks.append(dict(mode=mode, max_hidden_error=error, seq_lens=s))
        plan['merged'].copy_(captured)
    self.executions += 1
    return plan['merged']


def generate(self, *args, **kw):
    global iteration, mode
    mode = variants[iteration % len(variants)]
    iteration += 1
    if profile_window and iteration == len(variants) + 1:
        torch.cuda.profiler.start()
    try:
        return original_generate(self, *args, **kw)
    finally:
        if profile_window and iteration == 2 * len(variants):
            torch.cuda.profiler.stop()


def traced_split(*a, **kw):
    with torch.cuda.nvtx.range(f'composite_probe.{mode}'):
        return split(*a, **kw)


GraphAdapter.execute = traced_split
LLM.generate = generate
LLM.__init__ = initialize
temporary = tempfile.TemporaryDirectory(prefix='igniteserve-composite-')
profile_window = args.profile_window
output = Path(temporary.name) / 'result.json'
worker = Path(__file__).with_name('profile_cudagraph_adaptation.py')
sys.argv = [str(worker), '--worker', '--mode', 'collect', '--model', args.model,
            '--repeats', str((args.repeats + warmup_cycles) * len(variants)),
            '--output', str(output)]
if batch_size == 23:
    prompts = [
        'The capital of France is', 'Two plus two equals', 'The color of the sky is',
        'Write a definition of a computer.', 'Explain the purpose of a compiler.',
        'Describe memory.', 'What is a compiler?', 'The capital of China is',
        'Explain the purpose of an operating system.', 'Count from one to ten.',
        'Explain a virtual machine.', 'Write a short sentence about the ocean.',
        'What does a GPU do?', 'Describe a database.', 'Explain parallel computing.',
        'What does CPU mean?', 'The opposite of hot is', 'Explain a tensor.',
        'What is a programming language?', 'What is an integer?', 'Describe Python.',
        'Explain the difference between memory and storage.', 'What is a matrix?']
    prompt_file = Path(temporary.name) / 'prompts.json'
    prompt_file.write_text(json.dumps(prompts))
    sys.argv += ['--max-num-seqs', '32', '--max-num-batched-tokens', '128',
                 '--capture-sizes', '1,2,4,8,16,32,64,128', '--prompts', str(prompt_file)]
try:
    runpy.run_path(str(worker), run_name='__main__')
    result = json.loads(output.read_text())
    answers, values = result['tokens'], result['token_logprobs']
    assert len(answers) == (args.repeats + warmup_cycles) * len(variants)
    assert all(answer == answers[0] for answer in answers), 'greedy tokens differ'
    width = batch_size * 12
    assert len(values) == len(answers) * width
    differences = {}
    for i, variant in enumerate(variants):
        # Whole-forward capture must agree with whole eager, while split capture
        # must agree with equal-partition piecewise replay. Cross-shape arithmetic
        # differences are reported, never silently treated as replay corruption.
        reference = variants.index('eager') if variant == 'full_whole' else 0
        error = max(abs(values[step*width+j] - values[reference*width+j])
                    for step in range(i, len(answers), len(variants)) for j in range(width))
        differences[variant] = error
        if variant in ('baseline', 'full_serial', 'full_parallel', 'full_whole'):
            assert error < 1e-5, f'{variant}: equal-shape replay differs: {error}'
    torch.cuda.synchronize()
    rows = defaultdict(list)
    for name, a, b in samples:
        rows[name].append(a.elapsed_time(b))
    assert set(rows) == set(variants), 'missing execution-path coverage'
    print('COMPOSITE_PROBE', json.dumps(dict(
        batch_size=batch_size, scope='forward_including_staging_excluding_sampling',
        warmup_cycles=warmup_cycles, builds=builds, checks=checks,
        tokens_equal=True, logprob_errors=differences,
        timings={k: dict(count=len(v), p50_ms=statistics.median(v),
                        minimum_ms=min(v), maximum_ms=max(v)) for k,v in rows.items()})), flush=True)
finally:
    GraphAdapter.execute = original_split
    LLM.generate = original_generate
    LLM.__init__ = original_init
    plans.clear()
    temporary.cleanup()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
