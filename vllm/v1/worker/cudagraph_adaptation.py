# SPDX-License-Identifier: Apache-2.0
"""Cost-gated whole-request partitioning and private composite graph replay."""

import bisect
import json
import os
import time
from collections import Counter
from dataclasses import dataclass

import torch

from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


def workload_key(query_lens, seq_lens, is_prefill):
    # Exact query sizes and a context-length histogram distinguish equal-token
    # batches with very different attention work. Never extrapolate a profile.
    if not len(query_lens) == len(seq_lens) == len(is_prefill):
        raise ValueError("workload vectors must describe the same requests")
    rows = Counter((int(q), 1 << (max(int(s), 1) - 1).bit_length(), bool(p))
                   for q, s, p in zip(query_lens, seq_lens, is_prefill))
    return json.dumps(sorted((*key, n) for key, n in rows.items()),
                      separators=(",", ":"))


@dataclass(frozen=True)
class GraphPartition:
    first_request: int
    last_request: int
    first_token: int
    last_token: int
    graph_size: int


def composite_workload(parts, query_lens, seq_lens, is_prefill):
    return json.dumps([
        (p.graph_size, workload_key(query_lens[p.first_request:p.last_request],
                                   seq_lens[p.first_request:p.last_request],
                                   is_prefill[p.first_request:p.last_request]))
        for p in parts], separators=(",", ":"))


def plan_partitions(matcher, query_lens, seq_lens, is_prefill, *,
                    max_parts=3, max_pad_ratio=.2, gamma=.9, captured_sizes=None,
                    composite=False):
    """Bounded beam search with a padding budget for the entire batch."""
    count = len(query_lens)
    if count < 2 or not 0 < gamma < 1 or any(int(q) <= 0 for q in query_lens):
        return ()
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + int(q))
    total = starts[-1]
    sizes = tuple(matcher.captured_sizes if captured_sizes is None else captured_sizes)
    table = matcher.cost_table
    workload = workload_key(query_lens, seq_lens, is_prefill)
    eager = table.estimate_workload("eager", total, workload)
    if eager is None:
        return ()
    started = time.perf_counter()
    budget_ms = min(1., .05 * eager)

    def accepted():
        elapsed_ms = (time.perf_counter() - started) * 1000
        return best[1] if best and best[0] + elapsed_ms < gamma * eager else ()

    frontier = [(0., ())]
    best = None
    for n in range(1, max_parts + 1):
        following = []
        for cost, parts in frontier:
            first = parts[-1].last_request if parts else 0
            padded = sum(p.graph_size for p in parts) - starts[first]
            for size in reversed(sizes):
                if (time.perf_counter() - started) * 1000 > budget_ms:
                    return accepted()
                last = bisect.bisect_right(starts, starts[first] + size) - 1
                if last <= first:
                    continue
                actual = starts[last] - starts[first]
                # A small tail may exceed the ratio locally while adding very
                # little work overall (7 -> 4+4 instead of 4+2+1). Charge every
                # part against one batch-wide budget, never separate allowances.
                if padded + size - actual > max_pad_ratio * total:
                    continue
                key = workload_key(query_lens[first:last], seq_lens[first:last],
                                   is_prefill[first:last])
                elapsed = 0. if composite else table.estimate_workload("graph", size, key)
                if elapsed is None:
                    continue
                if cost + elapsed >= min(gamma * eager, best[0] if best else float('inf')):
                    continue
                plan = parts + (GraphPartition(first, last, starts[first], starts[last], size),)
                if last == count:
                    overhead = table.estimate_workload(
                        'composite' if composite else 'split_overhead', total,
                        composite_workload(plan, query_lens, seq_lens, is_prefill)
                        if composite else workload, n)
                    if n >= 2 and overhead is not None:
                        predicted = cost + elapsed + overhead
                        if predicted < gamma * eager and (best is None or predicted < best[0]):
                            best = (predicted, plan)
                            if composite:
                                # Larger ready parts first; stop once a measured
                                # whole-composite cost proves a conservative win.
                                return accepted()
                else:
                    following.append((cost + elapsed, plan))
        frontier = sorted(following, key=lambda item: item[0])[:16]
    return accepted()


class GraphAdapter:
    def __init__(self, runner):
        self.runner = runner
        self.enabled = os.getenv("VLLM_ENABLE_CUDAGRAPH_SPLIT",
                                 os.getenv("VLLM_ENABLE_STARTUP_OPTIMIZATIONS", "0")) == "1"
        self._memory_checked = 0.
        self._headroom = 0
        self._pad_ratio = runner.graphMatcher.max_pad_ratio
        self._pressure = 0
        self.executions = 0
        self.fallbacks = 0
        self.declines = 0
        self.composite = None
        self._composite_parts = ()

    def ready_sizes(self, candidates=None):
        """Only complete, published chains; a matcher entry alone is not proof."""
        registry = getattr(self.runner.compilation_config,
                           'cuda_piecewise_backends_list', {})
        backends = tuple(registry.values())
        if not backends or any(b.total_piecewise_compiles != len(backends)
                               for b in backends):
            return ()
        ready = []
        for size in tuple(self.runner.graphMatcher.captured_sizes):
            if candidates is not None and size not in candidates:
                continue
            entries = [b.concrete_size_entries.get(size) for b in backends]
            if all(e is not None and e.use_cudagraph and e.cudagraph is not None
                   and not e.concurrent_capture_in_progress
                   and (e.concurrent_capture_gate is None
                        or e.concurrent_capture_gate.is_set()) for e in entries):
                ready.append(size)
        return tuple(ready)

    def pad_limit(self):
        now = time.monotonic()
        if now - self._memory_checked >= 1.:
            free, total = torch.cuda.mem_get_info(self.runner.device)
            # Reserved-but-unused bytes may belong to live graph private pools;
            # they are not necessarily reclaimable by this allocation.
            self._headroom = max(0, free - max(256 << 20, total // 50))
            fraction = self._headroom / total
            if fraction < .05:
                self._pressure = 2
            elif ((fraction < .1 and self._pressure < 2)
                  or (self._pressure == 2 and fraction >= .07)):
                self._pressure = 1
            if fraction >= .12:
                self._pressure = 0
            self._pad_ratio = min(self.runner.graphMatcher.max_pad_ratio,
                                  (0.2, 0.1, 0.)[self._pressure])
            self._memory_checked = now
        return self._pad_ratio

    def supported(self, metadata):
        r = self.runner
        p = r.parallel_config
        return (self.enabled and self.fallbacks == 0 and r.use_cuda_graph and not r.full_cuda_graph
                and p.tensor_parallel_size == p.pipeline_parallel_size == p.data_parallel_size == 1
                and not (r.is_multimodal_model or r.uses_mrope or r.lora_config
                         or r.speculative_config or r.use_aux_hidden_state_outputs
                         or getattr(r.model_config, "is_hybrid", False)
                         or getattr(r.model_config, "is_encoder_decoder", False)
                         or r.input_batch.pooling_params)
                and bool(metadata) and all(
                    callable(getattr(m, "split_for_graph", None))
                    and not m.use_cascade and m.local_attn_metadata is None
                    for m in metadata.values()))

    def plan(self, metadata, query_lens, seq_lens, is_prefill):
        self._composite_parts = ()
        if not self.supported(metadata):
            return ()
        limit = self.pad_limit()
        r = self.runner
        total = sum(query_lens)
        # Full hidden-state aggregation preserves prompt-logprob consumers too.
        required = total * (r.hidden_size * torch.empty((), dtype=r.dtype).element_size() + 32)
        if required + (1 << 20) > self._headroom:
            return ()
        layers = getattr(r.model_config.hf_config, "num_hidden_layers", 0)
        manager = getattr(r, 'concurrent_piecewise_cudagraph_manager', None)
        if manager is not None and manager.enabled:
            parts = plan_partitions(
                r.graphMatcher, query_lens, seq_lens, is_prefill,
                max_parts=2 if layers > 40 else 3, max_pad_ratio=limit,
                captured_sizes=self.ready_sizes(), composite=True)
            if parts:
                self._composite_parts = parts
                return parts
        return plan_partitions(r.graphMatcher, query_lens, seq_lens, is_prefill,
                               max_parts=2 if layers > 40 else 3,
                               max_pad_ratio=limit, captured_sizes=self.ready_sizes())

    def execute(self, parts, metadata, query_lens, seq_lens, *, profile=None):
        """No whole-batch retry after KV writes. Same-stream copies own outputs."""
        r = self.runner
        total = sum(query_lens)
        # Decline before staging or KV writes if publication changed since plan.
        # Existing ready chains are not evicted/rearmed by the capture manager.
        ready = self.ready_sizes({p.graph_size for p in parts})
        if any(p.graph_size not in ready for p in parts):
            self.declines += 1
            return None
        if parts == self._composite_parts:
            if self.composite is None:
                from vllm.v1.worker.composite_cudagraph import CompositeGraphs
                self.composite = CompositeGraphs(r)
            output = self.composite.execute(parts, metadata, query_lens, seq_lens)
            if output is not None:
                self.executions += 1
            return output
        # Build every metadata object and reserve memory before the first KV write.
        try:
            lanes = []
            for part in parts:
                by_identity = {}
                lane = {}
                for name, meta in metadata.items():
                    if id(meta) not in by_identity:
                        by_identity[id(meta)] = meta.split_for_graph(part, query_lens, seq_lens)
                    lane[name] = by_identity[id(meta)]
                lanes.append(lane)
            inputs = r.input_ids[:total].clone()
            positions = r.positions[:total].clone()
            merged = torch.empty((total, r.hidden_size), dtype=r.dtype, device=r.device)
        except torch.OutOfMemoryError:
            self.fallbacks += 1
            logger.warning("Graph adaptation allocation declined before any KV write")
            return None
        stats = {}
        try:
            for part, lane in zip(parts, lanes):
                start, end, size = part.first_token, part.last_token, part.graph_size
                actual = end - start
                r.input_ids[:actual].copy_(inputs[start:end])
                r.positions[:actual].copy_(positions[start:end])
                r.input_ids[actual:size].zero_()
                r.positions[actual:size].zero_()
                with set_forward_context(lane, r.vllm_config, num_tokens=size,
                                         cudagraph_adaptation_stats=stats):
                    if profile:
                        profile.before(part)
                    output = r.model(input_ids=r.input_ids[:size],
                                     positions=r.positions[:size],
                                     intermediate_tensors=None, inputs_embeds=None)
                    if profile:
                        profile.after(part)
                    if (not isinstance(output, torch.Tensor) or output.dtype != merged.dtype
                            or output.ndim != 2 or output.shape[1] != r.hidden_size
                            or output.shape[0] < actual):
                        raise RuntimeError("Split forward returned an unsupported hidden-state contract")
                    # Saving views is unsafe: replay may overwrite the same output.
                    merged[start:end].copy_(output[:actual])
        finally:
            r.input_ids[:total].copy_(inputs)
            r.positions[:total].copy_(positions)
        self.executions += 1
        replays, pieces = stats.get('replays', 0), stats.get('pieces', 0)
        if not pieces or replays != pieces:
            self.fallbacks += 1
            logger.warning("Graph adaptation disabled after split fallback: replays=%d pieces=%d",
                           replays, pieces)
        if self.executions <= 8:
            logger.info("Graph adaptation split: requests=%d tokens=%d graphs=%s replays=%d pieces=%d",
                        len(query_lens), total, [p.graph_size for p in parts], replays, pieces)
        return merged
