# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class MatchResult:
    strategy: str
    raw_size: int
    execution_size: int
    graph_size: int = 0
    split_request_sizes: tuple[int, ...] = field(default_factory=tuple)
    split_sizes: tuple[int, ...] = field(default_factory=tuple)
    match_sizes: tuple[int, ...] = field(default_factory=tuple)
    padding_ratio: float = 0.0
    estimated_eager_ms: Optional[float] = None
    estimated_graph_ms: Optional[float] = None
    expected_speedup: float = 1.0
    reason: str = ""


@dataclass(frozen=True)
class PhaseSeparationResult:
    strategy: str
    prefill_tokens: int
    decode_tokens: int
    decode_graph_size: int = 0
    estimated_mixed_ms: Optional[float] = None
    estimated_prefill_ms: Optional[float] = None
    estimated_decode_ms: Optional[float] = None
    estimated_total_ms: Optional[float] = None
    expected_speedup: float = 1.0
    reason: str = ""


class GraphCostTable:
    """Offline GPU-time samples used by the runtime matcher.

    Expected CSV columns are ``phase``, ``mode``, ``execution_shape`` and
    ``mean_gpu_ms``. Missing entries are interpolated from the nearest two
    samples for the same phase and execution mode.
    """

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self._samples: dict[tuple[str, str], list[tuple[int, float]]] = {}
        if csv_path:
            self.load(csv_path)

    def load(self, csv_path: str) -> None:
        path = Path(csv_path)
        if not path.is_file():
            return
        samples: dict[tuple[str, str], dict[int, list[float]]] = {}
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["phase"], row["mode"])
                shape = int(row["execution_shape"])
                samples.setdefault(key, {}).setdefault(shape, []).append(
                    float(row["mean_gpu_ms"]))
        for key, values_by_shape in samples.items():
            self._samples[key] = sorted(
                (shape, sum(values) / len(values))
                for shape, values in values_by_shape.items())

    def estimate(self, phase: str, mode: str,
                 shape: int) -> Optional[float]:
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
                values, values[1:]):
            if left_shape <= shape <= right_shape:
                if left_shape == right_shape:
                    return left_ms
                weight = ((shape - left_shape) /
                          (right_shape - left_shape))
                return left_ms + weight * (right_ms - left_ms)
        return None


class GraphMatcher:
    """Select an already captured graph without changing request semantics.

    This class is intentionally decision-only. Split plans are never executed
    by the model runner until attention metadata slicing, KV-cache indexing and
    unified sampling have independent correctness coverage.
    """

    def __init__(
        self,
        captured_sizes: Iterable[int],
        max_pad_ratio: float = 0.20,
        split_overhead_ms: float = 0.0,
        max_split_num: int = 3,
        cost_table_path: Optional[str] = None,
        min_estimated_saving_ratio: float = 0.0,
    ) -> None:
        self.captured_sizes = sorted(set(int(size)
                                         for size in captured_sizes))
        self.max_pad_ratio = max(0.0, float(max_pad_ratio))
        self.split_overhead_ms = max(0.0, float(split_overhead_ms))
        self.max_split_num = max(1, int(max_split_num))
        self.min_estimated_saving_ratio = max(
            0.0, float(min_estimated_saving_ratio))
        self.cost_table = GraphCostTable(cost_table_path)

    def update_captured_sizes(self, captured_sizes: Iterable[int]) -> None:
        self.captured_sizes = sorted(set(int(size)
                                         for size in captured_sizes))

    @staticmethod
    def _eager(raw_size: int, reason: str,
               eager_ms: Optional[float] = None) -> MatchResult:
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
        allow_split: bool = False,
        *,
        phase: str = "decode",
        default_graph_size: Optional[int] = None,
        dynamic_sizes: Iterable[int] = (),
        allow_dynamic_cross_phase: bool = False,
    ) -> MatchResult:
        del allow_split  # Split execution is deliberately not enabled yet.
        raw_size = int(num_tokens)
        dynamic = set(int(size) for size in dynamic_sizes)
        eligible = [
            size for size in self.captured_sizes
            if size >= raw_size and (
                phase == "decode" or size not in dynamic
                or allow_dynamic_cross_phase)
        ]
        eager_ms = self.cost_table.estimate(phase, "eager", raw_size)
        if not eligible:
            return self._eager(raw_size, "no-captured-graph", eager_ms)

        candidate = eligible[0]
        padding_ratio = (candidate - raw_size) / max(raw_size, 1)
        native_candidate = (default_graph_size is not None
                            and candidate == int(default_graph_size))

        # Preserve vLLM's native nearest registered graph choice. The ratio
        # guard applies only when a sparse capture set requires padding beyond
        # that native target.
        if not native_candidate and padding_ratio > self.max_pad_ratio:
            return self._eager(
                raw_size,
                f"pad-ratio={padding_ratio:.4f}>max={self.max_pad_ratio:.4f}",
                eager_ms,
            )

        graph_ms = self.cost_table.estimate(phase, "graph", candidate)
        if (not native_candidate and candidate != raw_size
                and (eager_ms is None or graph_ms is None)):
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
                    reason=(f"cost-reject graph={graph_ms:.4f}ms "
                            f"eager={eager_ms:.4f}ms"),
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
            match_sizes=(candidate, ),
            padding_ratio=padding_ratio,
            estimated_eager_ms=eager_ms,
            estimated_graph_ms=graph_ms,
            expected_speedup=speedup,
            reason=(f"{strategy} {raw_size}->{candidate} "
                    f"padding_ratio={padding_ratio:.4f}"),
        )

    def plan_split(
        self,
        phase: str,
        num_tokens: int,
        gamma: float = 0.90,
        execution_mode: str = "sequential",
        parallel_overlap_efficiency: float = 0.0,
    ) -> MatchResult:
        """Return a costed split plan without authorizing its execution.

        ``split_request_sizes`` partitions the real requests, while
        ``split_sizes`` contains the graph sizes used to execute those parts.
        Keeping these separate is essential when the final part is padded.

        ``ideal_parallel`` is an optimistic lower bound: it assumes the graph
        GPU times overlap perfectly and therefore uses the slowest graph as the
        critical path. It is useful for deciding whether isolated buffers and
        multi-stream replay are worth implementing; it never enables replay.
        """
        if execution_mode not in (
                "sequential", "ideal_parallel", "calibrated_parallel"):
            raise ValueError(f"Unsupported split execution mode: "
                             f"{execution_mode}")
        overlap_efficiency = min(
            1.0, max(0.0, float(parallel_overlap_efficiency)))
        raw_size = int(num_tokens)
        eager_ms = self.cost_table.estimate(phase, "eager", raw_size)
        if eager_ms is None:
            return self._eager(raw_size, "split-requires-cost-table")

        candidates = [size for size in self.captured_sizes
                      if size <= raw_size]
        if not candidates:
            return self._eager(raw_size, "no-split-candidates", eager_ms)

        best: Optional[tuple[float, float, float, tuple[int, ...]]] = None

        def consider(parts: tuple[int, ...], costs: tuple[float, ...]) -> None:
            nonlocal best
            if len(parts) < 2 or sum(parts) < raw_size:
                return
            padding_ratio = (sum(parts) - raw_size) / max(raw_size, 1)
            if padding_ratio > self.max_pad_ratio:
                return
            coordination_ms = self.split_overhead_ms * (len(parts) - 1)
            sequential_ms = sum(costs) + coordination_ms
            parallel_lower_bound_ms = max(costs) + coordination_ms
            if execution_mode == "sequential":
                selected_ms = sequential_ms
            elif execution_mode == "ideal_parallel":
                selected_ms = parallel_lower_bound_ms
            else:
                overlapable_ms = sequential_ms - parallel_lower_bound_ms
                selected_ms = (sequential_ms -
                               overlap_efficiency * overlapable_ms)
            candidate = (selected_ms, sequential_ms,
                         parallel_lower_bound_ms, parts)
            if best is None or candidate < best:
                best = candidate

        def search(parts: tuple[int, ...], costs: tuple[float, ...],
                   max_candidate_index: int) -> None:
            if len(parts) >= 2:
                consider(parts, costs)
            if len(parts) >= self.max_split_num or sum(parts) >= raw_size:
                return
            # Graph order does not change the cost. Enumerating non-increasing
            # sizes removes duplicate permutations while retaining all plans.
            for index in range(max_candidate_index, -1, -1):
                size = candidates[index]
                graph_ms = self.cost_table.estimate(phase, "graph", size)
                if graph_ms is None:
                    continue
                search(parts + (size, ), costs + (graph_ms, ), index)

        search((), (), len(candidates) - 1)
        if best is None:
            return self._eager(raw_size, "no-feasible-split", eager_ms)

        graph_sizes = best[3]
        remaining = raw_size
        request_sizes: list[int] = []
        for index, graph_size in enumerate(graph_sizes):
            remaining_parts = len(graph_sizes) - index - 1
            request_size = min(graph_size, remaining - remaining_parts)
            if request_size <= 0:
                return self._eager(raw_size, "invalid-split-partition",
                                   eager_ms)
            request_sizes.append(request_size)
            remaining -= request_size
        if remaining != 0:
            return self._eager(raw_size, "incomplete-split-partition",
                               eager_ms)

        if best[0] >= gamma * eager_ms:
            return MatchResult(
                strategy="eager",
                raw_size=raw_size,
                execution_size=raw_size,
                graph_size=max(graph_sizes),
                split_request_sizes=tuple(request_sizes),
                split_sizes=graph_sizes,
                match_sizes=graph_sizes,
                padding_ratio=(sum(graph_sizes) - raw_size) /
                max(raw_size, 1),
                estimated_eager_ms=eager_ms,
                estimated_graph_ms=best[0],
                expected_speedup=eager_ms / best[0],
                reason=f"split-{execution_mode}-not-profitable",
            )

        return MatchResult(
            strategy={
                "sequential": "split_plan_only",
                "ideal_parallel": "parallel_split_plan_only",
                "calibrated_parallel": "calibrated_parallel_split_plan_only",
            }[execution_mode],
            raw_size=raw_size,
            execution_size=sum(graph_sizes),
            graph_size=max(graph_sizes),
            split_request_sizes=tuple(request_sizes),
            split_sizes=graph_sizes,
            match_sizes=graph_sizes,
            padding_ratio=(sum(graph_sizes) - raw_size) / max(raw_size, 1),
            estimated_eager_ms=eager_ms,
            estimated_graph_ms=best[0],
            expected_speedup=eager_ms / best[0],
            reason=(f"costed {execution_mode} split plan; execution disabled; "
                    f"sequential={best[1]:.4f}ms; "
                    f"ideal_parallel={best[2]:.4f}ms; "
                    f"overlap_efficiency={overlap_efficiency:.4f}"),
        )

    def plan_phase_separation(
        self,
        prefill_tokens: int,
        decode_tokens: int,
        gamma: float = 0.90,
        execution_mode: str = "sequential",
        parallel_overlap_efficiency: float = 0.0,
    ) -> PhaseSeparationResult:
        """Cost a mixed-batch prefill/decode separation without executing it."""
        if execution_mode not in (
                "sequential", "ideal_parallel", "calibrated_parallel"):
            raise ValueError(f"Unsupported phase execution mode: "
                             f"{execution_mode}")
        overlap_efficiency = min(
            1.0, max(0.0, float(parallel_overlap_efficiency)))
        prefill_tokens = int(prefill_tokens)
        decode_tokens = int(decode_tokens)
        if prefill_tokens <= 0 or decode_tokens <= 0:
            return PhaseSeparationResult(
                "eager", prefill_tokens, decode_tokens,
                reason="phase-separation-requires-mixed-batch")

        mixed_tokens = prefill_tokens + decode_tokens
        mixed_ms = self.cost_table.estimate("mixed", "eager", mixed_tokens)
        prefill_ms = self.cost_table.estimate(
            "prefill", "eager", prefill_tokens)
        decode_eager_ms = self.cost_table.estimate(
            "decode", "eager", decode_tokens)
        if mixed_ms is None or prefill_ms is None or decode_eager_ms is None:
            return PhaseSeparationResult(
                "eager",
                prefill_tokens,
                decode_tokens,
                estimated_mixed_ms=mixed_ms,
                estimated_prefill_ms=prefill_ms,
                estimated_decode_ms=decode_eager_ms,
                reason="phase-separation-requires-cost-table",
            )

        decode_match = self.find_best_match(
            decode_tokens,
            phase="decode",
            default_graph_size=None,
        )
        decode_ms = (decode_match.estimated_graph_ms
                     if decode_match.strategy != "eager" else
                     decode_eager_ms)
        if decode_ms is None:
            return PhaseSeparationResult(
                "eager", prefill_tokens, decode_tokens,
                estimated_mixed_ms=mixed_ms,
                estimated_prefill_ms=prefill_ms,
                reason="decode-cost-unavailable")

        sequential_ms = prefill_ms + decode_ms + self.split_overhead_ms
        ideal_ms = max(prefill_ms, decode_ms) + self.split_overhead_ms
        if execution_mode == "sequential":
            total_ms = sequential_ms
        elif execution_mode == "ideal_parallel":
            total_ms = ideal_ms
        else:
            total_ms = (sequential_ms - overlap_efficiency *
                        (sequential_ms - ideal_ms))
        accepted = total_ms < gamma * mixed_ms
        return PhaseSeparationResult(
            strategy=(f"phase_separation_{execution_mode}_plan_only"
                      if accepted else "eager"),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            decode_graph_size=(decode_match.execution_size
                               if decode_match.strategy != "eager" else 0),
            estimated_mixed_ms=mixed_ms,
            estimated_prefill_ms=prefill_ms,
            estimated_decode_ms=decode_ms,
            estimated_total_ms=total_ms,
            expected_speedup=mixed_ms / total_ms,
            reason=(f"costed {execution_mode} phase separation; execution "
                    f"disabled" if accepted else
                    f"phase-separation-{execution_mode}-not-profitable"),
        )
