# SPDX-License-Identifier: Apache-2.0
"""Private full-forward fork/join graphs for cost-qualified batch partitions.

Recording performs no model execution and never warms by repeating live KV
writes. A single capture worker serializes this with on-demand piece capture.
Until publication, requests stay eager. Replay includes attention, so both the
CPU metadata contract and all tensor buffers are private and checked on reuse.
"""
import dataclasses
import threading
import time

import torch

from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


def runtime_binding(runner):
    return (id(runner.model), runner.input_ids.data_ptr(), runner.positions.data_ptr(),
            tuple((cache.data_ptr(), tuple(cache.shape), str(cache.dtype))
                  for cache in runner.kv_caches))


def metadata_signature(metadata):
    fields = []
    for field in dataclasses.fields(metadata):
        value = getattr(metadata, field.name)
        if isinstance(value, torch.Tensor):
            value = (tuple(value.shape), tuple(value.stride()), str(value.dtype),
                     str(value.device))
        elif not isinstance(value, (int, float, bool, str, type(None))):
            raise ValueError(f'unsupported composite metadata: {field.name}')
        fields.append((field.name, value))
    return tuple(fields)


def partition_metadata(metadata, parts, query_lens, seq_lens, maximum):
    lanes = []
    for part in parts:
        shared = {}
        lane = {}
        for name, meta in metadata.items():
            if id(meta) not in shared:
                value = meta.split_for_graph(part, query_lens, seq_lens)
                # Captured FA launch arguments stay fixed within a context bucket.
                bound = min(maximum, 1 << (max(value.max_seq_len, 1) - 1).bit_length())
                shared[id(meta)] = dataclasses.replace(value, max_seq_len=bound)
            lane[name] = shared[id(meta)]
        lanes.append(lane)
    return lanes


def contract(parts, lanes):
    result = []
    for lane in lanes:
        indices, signatures, names = {}, [], []
        for name, meta in sorted(lane.items()):
            if id(meta) not in indices:
                indices[id(meta)] = len(signatures)
                signatures.append(metadata_signature(meta))
            names.append((name, indices[id(meta)]))
        result.append((tuple(names), tuple(signatures)))
    return tuple(parts), tuple(result)


class CompositeGraph:
    def __init__(self, runner, parts, metadata):
        self.runner = runner
        self.parts = tuple(parts)
        self.ready = threading.Event()
        self.failed = False
        self.graph = None
        self.lanes = []
        self.replays = 0
        self.build_seconds = None
        self.created_at = time.perf_counter()
        for part, metas in zip(parts, metadata):
            copies = {}
            private = {}
            for name, meta in metas.items():
                if id(meta) not in copies:
                    values = {f.name: (getattr(meta, f.name).clone()
                              if isinstance(getattr(meta, f.name), torch.Tensor)
                              else getattr(meta, f.name)) for f in dataclasses.fields(meta)}
                    copies[id(meta)] = dataclasses.replace(meta, **values)
                private[name] = copies[id(meta)]
            self.lanes.append(dict(
                inputs=torch.empty_like(runner.input_ids[:part.graph_size]),
                positions=torch.empty_like(runner.positions[:part.graph_size]),
                metadata=private, stream=torch.cuda.Stream(device=runner.device)))
        total = parts[-1].last_token
        self.output = torch.empty((total, runner.hidden_size), dtype=runner.dtype,
                                  device=runner.device)
        self.stream = torch.cuda.Stream(device=runner.device)
        self.staged = torch.cuda.Event()

    def stage(self, metadata):
        r = self.runner
        for part, lane, metas in zip(self.parts, self.lanes, metadata):
            lo, hi = part.first_token, part.last_token
            actual = hi - lo
            lane['inputs'][:actual].copy_(r.input_ids[lo:hi])
            lane['positions'][:actual].copy_(r.positions[lo:hi])
            lane['inputs'][actual:].zero_()
            lane['positions'][actual:].zero_()
            seen = set()
            for name, source in metas.items():
                target = lane['metadata'][name]
                if id(target) in seen:
                    continue
                seen.add(id(target))
                for field in dataclasses.fields(source):
                    value = getattr(source, field.name)
                    if isinstance(value, torch.Tensor):
                        getattr(target, field.name).copy_(value)

    def capture(self):
        r = self.runner
        started = time.perf_counter()
        try:
            manager = r.cuda_graph_manager.capture
            manager._initialize_worker_runtime()
            # Only wait for this immutable snapshot, never the whole device.
            self.staged.synchronize()
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            fork = torch.cuda.Event()
            joins = [torch.cuda.Event() for _ in self.parts]
            with torch.inference_mode(), torch.cuda.graph(
                    graph, pool=pool, stream=self.stream,
                    capture_error_mode='thread_local',
                    synchronize_before_capture=False, collect_before_capture=False,
                    empty_cache_before_capture=False, manage_default_generator=False):
                fork.record()
                for part, lane, join in zip(self.parts, self.lanes, joins):
                    lane['stream'].wait_event(fork)
                    with torch.cuda.stream(lane['stream']), set_forward_context(
                            lane['metadata'], r.vllm_config, num_tokens=part.graph_size,
                            skip_cuda_graphs=True, cudagraph_adaptation_stats={}):
                        out = r.model(input_ids=lane['inputs'], positions=lane['positions'],
                                      intermediate_tensors=None, inputs_embeds=None)
                        self.output[part.first_token:part.last_token].copy_(
                            out[:part.last_token - part.first_token])
                        join.record()
                for join in joins:
                    self.stream.wait_event(join)
            self.graph = graph
            self.build_seconds = time.perf_counter() - started
            self.ready.set()
            logger.info('Composite graph published: parts=%s build_s=%.6f',
                        [p.graph_size for p in self.parts], self.build_seconds)
        except Exception:
            self.failed = True
            logger.exception('Composite recording failed; requests remain eager')

    def replay(self, metadata):
        self.stage(metadata)
        # One native launch; fork/join and copies are graph nodes. Foreground
        # stream order also prevents a later request from overwriting live inputs.
        self.graph.replay()
        self.replays += 1
        if self.replays == 1:
            logger.info('Composite graph replay activated: parts=%s native_launches=1',
                        [p.graph_size for p in self.parts])
        return self.output


class CompositeGraphStore:
    def __init__(self, runner):
        self.runner = runner
        self.binding = runtime_binding(runner)
        self.plans = {}
        self.builds = self.replays = self.pending = self.declines = 0

    def execute(self, parts, metadata, query_lens, seq_lens):
        r = self.runner
        manager = r.cuda_graph_manager.capture
        # Model/KV rebind invalidates captured addresses, just as a new engine
        # does. Never replay a previous model's full-forward graph.
        if not manager.enabled or runtime_binding(r) != self.binding:
            self.declines += 1
            return None
        try:
            lanes = partition_metadata(metadata, parts, query_lens, seq_lens,
                                       r.model_config.max_model_len)
            key = contract(parts, lanes)
            plan = self.plans.get(key)
            if plan is None:
                # Bounded residency, no eviction of a graph still used by CUDA.
                if len(self.plans) >= 4:
                    self.declines += 1
                    return None
                plan = CompositeGraph(r, parts, lanes)
                plan.stage(lanes)
                plan.staged.record(torch.cuda.current_stream(r.device))
                self.plans[key] = plan
                self.builds += 1
                if not manager.enqueue_auxiliary(plan.capture):
                    plan.failed = True
                return None
            if not plan.ready.is_set() or plan.failed:
                self.pending += 1
                return None
        except (ValueError, torch.OutOfMemoryError):
            self.declines += 1
            return None
        # After replay submission, never retry the batch: KV writes may have run.
        output = plan.replay(lanes)
        self.replays += 1
        return output
