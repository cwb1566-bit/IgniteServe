# SPDX-License-Identifier: Apache-2.0
"""Cost-gated whole-request partitioning and private composite graph replay."""

import bisect
import csv
import math
from typing import Iterable, Optional
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class MatchResult:
    strategy: str
    raw_size: int
    execution_size: int
    graph_size: int = 0
    padding_ratio: float = 0.0
    estimated_eager_ms: Optional[float] = None
    estimated_graph_ms: Optional[float] = None
    expected_speedup: float = 1.0
    reason: str = ""


class GraphCostTable:
    """Offline GPU-time samples used by the runtime matcher.

    Qualified runtime costs match workload keys exactly. Legacy phase/shape
    rows remain interpolatable for diagnostic plans, never runtime admission.
    """

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self._samples: dict[tuple[str, str], list[tuple[int, float]]] = {}
        self._workloads: dict[tuple, float] = {}
        if csv_path:
            self.load(csv_path)

    def load(self, csv_path: str) -> None:
        path = Path(csv_path)
        if not path.is_file():
            return
        samples: dict[tuple[str, str], dict[int, list[float]]] = {}
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    elapsed = float(row["mean_gpu_ms"])
                    shape = int(row["execution_shape"])
                    mode = row["mode"]
                    count = int(row.get("samples") or 0)
                    parts = int(row.get("parts") or 1)
                    quantile = "p50_gpu_ms" if mode == "eager" else "p90_gpu_ms"
                    qualified_elapsed = float(row.get(quantile) or elapsed)
                except (KeyError, TypeError, ValueError):
                    continue
                if shape <= 0 or parts <= 0:
                    continue
                if not math.isfinite(elapsed) or elapsed <= 0:
                    continue
                if row.get("workload"):
                    # Qualified rows measure forward or preparation/merge only.
                    # Logits and sampling remain identical on both paths.
                    if row.get("scope") != "forward" or count < 3:
                        continue
                    elapsed = qualified_elapsed
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        continue
                    key = (mode, shape, row["workload"], parts)
                    previous = self._workloads.get(key, elapsed)
                    self._workloads[key] = (
                        min(previous, elapsed)
                        if mode == "eager"
                        else max(previous, elapsed)
                    )
                    continue
                key = (row.get("phase", "unknown"), mode)
                samples.setdefault(key, {}).setdefault(shape, []).append(
                    elapsed
                )
        for key, values_by_shape in samples.items():
            self._samples[key] = sorted(
                (shape, sum(values) / len(values))
                for shape, values in values_by_shape.items()
            )

    def estimate_workload(
        self, mode: str, shape: int, workload: str, parts: int = 1
    ) -> Optional[float]:
        return self._workloads.get((mode, shape, workload, parts))

    def estimate(self, phase: str, mode: str, shape: int) -> Optional[float]:
        values = self._samples.get((phase, mode))
        if not values:
            return None
        # Interpolation is valid only inside the measured range. Clamping an
        # unseen large prefill to the largest sample can fabricate a profitable
        # split decision by severely underestimating its cost.
        if shape < values[0][0] or shape > values[-1][0]:
            return None
        if shape == values[0][0]:
            return values[0][1]
        if shape == values[-1][0]:
            return values[-1][1]
        for (left_shape, left_ms), (right_shape, right_ms) in zip(
            values, values[1:]
        ):
            if left_shape <= shape <= right_shape:
                if left_shape == right_shape:
                    return left_ms
                weight = (shape - left_shape) / (right_shape - left_shape)
                return left_ms + weight * (right_ms - left_ms)
        return None


def workload_key(query_lens, seq_lens, is_prefill):
    # Exact query sizes and a context-length histogram distinguish equal-token
    # batches with very different attention work. Never extrapolate a profile.
    if not len(query_lens) == len(seq_lens) == len(is_prefill):
        raise ValueError("workload vectors must describe the same requests")
    rows = Counter(
        (int(q), 1 << (max(int(s), 1) - 1).bit_length(), bool(p))
        for q, s, p in zip(query_lens, seq_lens, is_prefill)
    )
    return json.dumps(
        sorted((*key, n) for key, n in rows.items()), separators=(",", ":")
    )


@dataclass(frozen=True)
class GraphPartition:
    first_request: int
    last_request: int
    first_token: int
    last_token: int
    graph_size: int


def composite_workload(parts, query_lens, seq_lens, is_prefill):
    return json.dumps(
        [
            (
                p.graph_size,
                workload_key(
                    query_lens[p.first_request : p.last_request],
                    seq_lens[p.first_request : p.last_request],
                    is_prefill[p.first_request : p.last_request],
                ),
            )
            for p in parts
        ],
        separators=(",", ":"),
    )


def plan_partitions(
    matcher,
    query_lens,
    seq_lens,
    is_prefill,
    *,
    max_parts=3,
    max_pad_ratio=0.2,
    gamma=0.9,
    captured_sizes=None,
    composite=False,
):
    """Bounded beam search with a padding budget for the entire batch."""
    count = len(query_lens)
    if count < 2 or not 0 < gamma < 1 or any(int(q) <= 0 for q in query_lens):
        return ()
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + int(q))
    total = starts[-1]
    sizes = tuple(
        matcher.captured_sizes if captured_sizes is None else captured_sizes
    )
    table = matcher.cost_table
    workload = workload_key(query_lens, seq_lens, is_prefill)
    eager = table.estimate_workload("eager", total, workload)
    if eager is None:
        return ()
    started = time.perf_counter()
    budget_ms = min(1.0, 0.05 * eager)

    def accepted():
        elapsed_ms = (time.perf_counter() - started) * 1000
        return best[1] if best and best[0] + elapsed_ms < gamma * eager else ()

    frontier = [(0.0, ())]
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
                key = workload_key(
                    query_lens[first:last],
                    seq_lens[first:last],
                    is_prefill[first:last],
                )
                elapsed = (
                    0.0
                    if composite
                    else table.estimate_workload("graph", size, key)
                )
                if elapsed is None:
                    continue
                if cost + elapsed >= min(
                    gamma * eager, best[0] if best else float("inf")
                ):
                    continue
                plan = parts + (
                    GraphPartition(
                        first, last, starts[first], starts[last], size
                    ),
                )
                if last == count:
                    overhead = table.estimate_workload(
                        "composite" if composite else "split_overhead",
                        total,
                        composite_workload(
                            plan, query_lens, seq_lens, is_prefill
                        )
                        if composite
                        else workload,
                        n,
                    )
                    if n >= 2 and overhead is not None:
                        predicted = cost + elapsed + overhead
                        if predicted < gamma * eager and (
                            best is None or predicted < best[0]
                        ):
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
    def __init__(
        self,
        runner=None,
        *,
        registry=None,
        captured_sizes=(),
        max_pad_ratio=0.2,
        cost_table_path=None,
        min_estimated_saving_ratio=0.0,
    ):
        from vllm.v1.worker.piecewise_graph_registry import (
            PiecewiseGraphRegistry,
        )

        self.runner = runner
        self.registry = (
            registry
            if registry is not None
            else PiecewiseGraphRegistry(
                getattr(runner, "compilation_config", None)
            )
        )
        for shape in captured_sizes:
            self.registry.publish(int(shape))
        self.max_pad_ratio = max_pad_ratio
        self.min_estimated_saving_ratio = min_estimated_saving_ratio
        self.cost_table = GraphCostTable(cost_table_path)
        self.enabled = (
            os.getenv(
                "VLLM_ENABLE_CUDAGRAPH_SPLIT",
                os.getenv("VLLM_ENABLE_STARTUP_OPTIMIZATIONS", "0"),
            )
            == "1"
        )
        self._memory_checked = 0.0
        self._headroom = 0
        self._pad_ratio = max_pad_ratio
        self._pressure = 0
        self.executions = 0
        self.fallbacks = 0
        self.declines = 0
        self._composite_parts = ()

    def ready_sizes(self, candidates=None):
        """Only complete, published chains; a matcher entry alone is not proof."""
        return self.registry.ready_sizes(candidates)

    @property
    def captured_sizes(self):
        return self.registry.published

    def pad_limit(self):
        now = time.monotonic()
        if now - self._memory_checked >= 1.0:
            free, total = torch.cuda.mem_get_info(self.runner.device)
            # Reserved-but-unused bytes may belong to live graph private pools;
            # they are not necessarily reclaimable by this allocation.
            self._headroom = max(0, free - max(256 << 20, total // 50))
            fraction = self._headroom / total
            if fraction < 0.05:
                self._pressure = 2
            elif (fraction < 0.1 and self._pressure < 2) or (
                self._pressure == 2 and fraction >= 0.07
            ):
                self._pressure = 1
            if fraction >= 0.12:
                self._pressure = 0
            self._pad_ratio = min(
                self.max_pad_ratio, (0.2, 0.1, 0.0)[self._pressure]
            )
            self._memory_checked = now
        return self._pad_ratio

    def supported(self, metadata):
        r = self.runner
        p = r.parallel_config
        return (
            self.enabled
            and self.fallbacks == 0
            and r.use_cuda_graph
            and not r.full_cuda_graph
            and p.tensor_parallel_size
            == p.pipeline_parallel_size
            == p.data_parallel_size
            == 1
            and not (
                r.is_multimodal_model
                or r.uses_mrope
                or r.lora_config
                or r.speculative_config
                or r.use_aux_hidden_state_outputs
                or getattr(r.model_config, "is_hybrid", False)
                or getattr(r.model_config, "is_encoder_decoder", False)
                or r.input_batch.pooling_params
            )
            and bool(metadata)
            and all(
                callable(getattr(m, "split_for_graph", None))
                and not m.use_cascade
                and m.local_attn_metadata is None
                for m in metadata.values()
            )
        )

    def plan(self, metadata, query_lens, seq_lens, is_prefill):
        self._composite_parts = ()
        if not self.supported(metadata):
            return ()
        limit = self.pad_limit()
        r = self.runner
        total = sum(query_lens)
        # Full hidden-state aggregation preserves prompt-logprob consumers too.
        required = total * (
            r.hidden_size * torch.empty((), dtype=r.dtype).element_size() + 32
        )
        if required + (1 << 20) > self._headroom:
            return ()
        layers = getattr(r.model_config.hf_config, "num_hidden_layers", 0)
        manager = getattr(r.cuda_graph_manager, "capture", None)
        if manager is not None and manager.enabled:
            parts = plan_partitions(
                self,
                query_lens,
                seq_lens,
                is_prefill,
                max_parts=2 if layers > 40 else 3,
                max_pad_ratio=limit,
                captured_sizes=self.ready_sizes(),
                composite=True,
            )
            if parts:
                self._composite_parts = parts
                return parts
        return plan_partitions(
            self,
            query_lens,
            seq_lens,
            is_prefill,
            max_parts=2 if layers > 40 else 3,
            max_pad_ratio=limit,
            captured_sizes=self.ready_sizes(),
        )

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
            if r.cuda_graph_manager.composites is None:
                from vllm.v1.worker.composite_cudagraph import (
                    CompositeGraphStore,
                )

                r.cuda_graph_manager.composites = CompositeGraphStore(r)
            output = r.cuda_graph_manager.composites.execute(
                parts, metadata, query_lens, seq_lens
            )
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
                        by_identity[id(meta)] = meta.split_for_graph(
                            part, query_lens, seq_lens
                        )
                    lane[name] = by_identity[id(meta)]
                lanes.append(lane)
            inputs = r.input_ids[:total].clone()
            positions = r.positions[:total].clone()
            merged = torch.empty(
                (total, r.hidden_size), dtype=r.dtype, device=r.device
            )
        except torch.OutOfMemoryError:
            self.fallbacks += 1
            logger.warning(
                "Graph adaptation allocation declined before any KV write"
            )
            return None
        stats = {}
        try:
            for part, lane in zip(parts, lanes):
                start, end, size = (
                    part.first_token,
                    part.last_token,
                    part.graph_size,
                )
                actual = end - start
                r.input_ids[:actual].copy_(inputs[start:end])
                r.positions[:actual].copy_(positions[start:end])
                r.input_ids[actual:size].zero_()
                r.positions[actual:size].zero_()
                with set_forward_context(
                    lane,
                    r.vllm_config,
                    num_tokens=size,
                    cudagraph_adaptation_stats=stats,
                ):
                    if profile:
                        profile.before(part)
                    output = r.model(
                        input_ids=r.input_ids[:size],
                        positions=r.positions[:size],
                        intermediate_tensors=None,
                        inputs_embeds=None,
                    )
                    if profile:
                        profile.after(part)
                    if (
                        not isinstance(output, torch.Tensor)
                        or output.dtype != merged.dtype
                        or output.ndim != 2
                        or output.shape[1] != r.hidden_size
                        or output.shape[0] < actual
                    ):
                        raise RuntimeError(
                            "Split forward returned an unsupported hidden-state contract"
                        )
                    # Saving views is unsafe: replay may overwrite the same output.
                    merged[start:end].copy_(output[:actual])
        finally:
            r.input_ids[:total].copy_(inputs)
            r.positions[:total].copy_(positions)
        self.executions += 1
        replays, pieces = stats.get("replays", 0), stats.get("pieces", 0)
        if not pieces or replays != pieces:
            self.fallbacks += 1
            logger.warning(
                "Graph adaptation disabled after split fallback: replays=%d pieces=%d",
                replays,
                pieces,
            )
        if self.executions <= 8:
            logger.info(
                "Graph adaptation split: requests=%d tokens=%d graphs=%s replays=%d pieces=%d",
                len(query_lens),
                total,
                [p.graph_size for p in parts],
                replays,
                pieces,
            )
        return merged

    @staticmethod
    def _eager(
        raw_size: int, reason: str, eager_ms: Optional[float] = None
    ) -> MatchResult:
        return MatchResult(
            strategy="eager",
            raw_size=raw_size,
            execution_size=raw_size,
            estimated_eager_ms=eager_ms,
            reason=reason,
        )

    def find_best_match(
        self,
        num_tokens: int,
        *,
        phase: str = "decode",
        default_graph_size: Optional[int] = None,
        dynamic_sizes: Iterable[int] = (),
        allow_dynamic_cross_phase: bool = False,
        workload: Optional[str] = None,
        max_pad_ratio: Optional[float] = None,
    ) -> MatchResult:
        raw_size = int(num_tokens)
        dynamic = set(int(size) for size in dynamic_sizes)
        eligible = [
            size
            for size in self.captured_sizes
            if size >= raw_size
            and (
                phase == "decode"
                or size not in dynamic
                or allow_dynamic_cross_phase
            )
        ]

        def estimate(mode, shape):
            if workload is not None:
                return self.cost_table.estimate_workload(mode, shape, workload)
            return self.cost_table.estimate(phase, mode, shape)

        eager_ms = estimate("eager", raw_size)
        if not eligible:
            return self._eager(raw_size, "no-captured-graph", eager_ms)

        candidate = eligible[0]
        padding_ratio = (candidate - raw_size) / max(raw_size, 1)
        native_candidate = default_graph_size is not None and candidate == int(
            default_graph_size
        )

        limit = self.max_pad_ratio if max_pad_ratio is None else max_pad_ratio
        if padding_ratio > limit:
            return self._eager(
                raw_size,
                f"pad-ratio={padding_ratio:.4f}>max={limit:.4f}",
                eager_ms,
            )

        graph_ms = estimate("graph", candidate)
        if candidate != raw_size and (eager_ms is None or graph_ms is None):
            return MatchResult(
                strategy="eager",
                raw_size=raw_size,
                execution_size=raw_size,
                graph_size=candidate,
                padding_ratio=padding_ratio,
                estimated_eager_ms=eager_ms,
                estimated_graph_ms=graph_ms,
                reason="cost-unavailable-for-sparse-pad",
            )
        if eager_ms is not None and graph_ms is not None:
            required = eager_ms * (1.0 - self.min_estimated_saving_ratio)
            if graph_ms >= required:
                return MatchResult(
                    strategy="eager",
                    raw_size=raw_size,
                    execution_size=raw_size,
                    graph_size=candidate,
                    padding_ratio=padding_ratio,
                    estimated_eager_ms=eager_ms,
                    estimated_graph_ms=graph_ms,
                    reason=(
                        f"cost-reject graph={graph_ms:.4f}ms "
                        f"eager={eager_ms:.4f}ms"
                    ),
                )
            speedup = eager_ms / graph_ms if graph_ms else float("inf")
        else:
            speedup = 1.0

        if candidate == raw_size:
            strategy = "exact"
        elif native_candidate:
            strategy = "native_pad"
        else:
            strategy = "pad_up"
        return MatchResult(
            strategy=strategy,
            raw_size=raw_size,
            execution_size=candidate,
            graph_size=candidate,
            padding_ratio=padding_ratio,
            estimated_eager_ms=eager_ms,
            estimated_graph_ms=graph_ms,
            expected_speedup=speedup,
            reason=(
                f"{strategy} {raw_size}->{candidate} "
                f"padding_ratio={padding_ratio:.4f}"
            ),
        )
