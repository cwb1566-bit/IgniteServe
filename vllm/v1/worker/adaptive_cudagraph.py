# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class GraphState(str, Enum):
    UNSEEN = "unseen"
    OBSERVED = "observed"
    WARMING = "warming"
    QUEUED = "queued"
    CAPTURING = "capturing"
    PARTIAL = "partial"
    READY = "ready"
    FAILED = "failed"


@dataclass
class GraphRecord:
    phase: str
    shape: int
    state: GraphState = GraphState.UNSEEN
    hit_count: int = 0
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    capture_seconds: float = 0.0
    capture_started_at: float = 0.0
    capture_finished_at: float = 0.0
    capture_context_enter_seconds: float = 0.0
    capture_body_seconds: float = 0.0
    capture_context_exit_seconds: float = 0.0
    capture_attempts: int = 0
    captured_pieces: int = 0
    remaining_pieces: int = 0
    last_capture_chunk_seconds: float = 0.0
    failure: Optional[str] = None


@dataclass(frozen=True)
class RuntimeDecision:
    phase: str
    raw_shape: int
    target_shape: int
    execution_shape: int
    skip_cudagraph: bool
    mark_runtime_warmup: bool
    reason: str


@dataclass(frozen=True)
class CaptureMaintenancePlan:
    """One non-preemptive capture unit admitted by an SLO slack budget."""

    phase: str
    shape: int
    piece_budget: int
    predicted_ms: float
    fixed_cost_p90_ms: float
    piece_cost_p90_ms: float
    hit_count: int
    remaining_pieces: int
    value_cost_score: float
    reason: str = "hotness-per-capture-cost"


class AdaptiveCUDAGraphController:
    """Owns adaptive CUDA graph state and capture authorization.

    Idle capture remains decode-only. A supported concurrent piecewise runner
    explicitly enables prefill and mixed triggers after its capability checks.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        min_hits: Optional[int] = None,
    ) -> None:
        self.enabled = (_env_flag("VLLM_ENABLE_ADAPTIVE_CUDAGRAPH", True)
                        if enabled is None else enabled)
        self.min_hits = (int(
            os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS", "1"))
                         if min_hits is None else min_hits)
        self.extra_warmups = int(
            os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_EXTRA_WARMUPS", "0"))
        self.max_captures_per_idle = max(
            1,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_MAX_CAPTURES_PER_IDLE", "1")))
        self._records: dict[tuple[str, int], GraphRecord] = {}
        self._lock = threading.Lock()
        self.telemetry_enabled = _env_flag(
            "VLLM_ADAPTIVE_CUDAGRAPH_TELEMETRY", True)
        self.profile_gpu = _env_flag(
            "VLLM_ADAPTIVE_CUDAGRAPH_PROFILE_GPU", False)
        self.max_gpu_samples_per_key = max(
            1,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_GPU_SAMPLES_PER_KEY", "16")))
        self._decision_counts: dict[tuple[str, int, int, int, str], int] = {}
        self._split_shadow_seen: set[tuple[str, int]] = set()
        self._split_shadow_rows: list[dict[str, object]] = []
        self._phase_separation_seen: set[tuple[int, int]] = set()
        self._phase_separation_rows: list[dict[str, object]] = []
        self._cross_phase_replay_counts: dict[tuple[str, int, int], int] = {}
        self._pending_gpu_samples: list[tuple[str, str, int, object,
                                              object]] = []
        self._reserved_gpu_samples: dict[tuple[str, str, int], int] = {}
        self._gpu_samples: dict[tuple[str, str, int], list[float]] = {}
        self._capture_chunk_rows: list[dict[str, object]] = []
        self._piece_capture_history_ms: dict[int, list[float]] = {}
        self._fixed_capture_history_ms: list[float] = []
        self._maintenance_plan_rows: list[dict[str, object]] = []
        self._planned_capture_shape: Optional[int] = None
        self.capture_phases = frozenset({"decode"})

    def _eligible_phase(self, phase: str) -> bool:
        return phase in self.capture_phases

    def settle_shared_capture(self, phase: str, shape: int,
                              error: Optional[BaseException] = None) -> None:
        """Resolve other phase records sharing one piecewise shape capture."""
        with self._lock:
            for (other_phase, other_shape), record in self._records.items():
                if other_shape != shape or other_phase == phase:
                    continue
                if record.state in (GraphState.OBSERVED, GraphState.WARMING,
                                    GraphState.QUEUED, GraphState.CAPTURING):
                    record.state = GraphState.READY if error is None else GraphState.FAILED
                    record.failure = None if error is None else repr(error)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._decision_counts.clear()
            self._split_shadow_seen.clear()
            self._split_shadow_rows.clear()
            self._phase_separation_seen.clear()
            self._phase_separation_rows.clear()
            self._cross_phase_replay_counts.clear()
            self._pending_gpu_samples.clear()
            self._reserved_gpu_samples.clear()
            self._gpu_samples.clear()
            self._capture_chunk_rows.clear()
            self._piece_capture_history_ms.clear()
            self._fixed_capture_history_ms.clear()
            self._maintenance_plan_rows.clear()
            self._planned_capture_shape = None

    def claim_split_shadow_shape(self, phase: str, raw_shape: int) -> bool:
        """Reserve one shadow evaluation per unique runtime shape."""
        key = (phase, int(raw_shape))
        with self._lock:
            if key in self._split_shadow_seen:
                return False
            self._split_shadow_seen.add(key)
            return True

    def record_split_shadow(self, phase: str, raw_shape: int, mode: str,
                            result: object) -> None:
        """Persist a decision-only split estimate without enabling execution."""
        with self._lock:
            self._split_shadow_rows.append({
                "phase": phase,
                "raw_shape": int(raw_shape),
                "mode": mode,
                "strategy": getattr(result, "strategy", "unknown"),
                "request_sizes": "+".join(
                    map(str, getattr(result, "split_request_sizes", ()))),
                "graph_sizes": "+".join(
                    map(str, getattr(result, "split_sizes", ()))),
                "estimated_eager_ms": getattr(
                    result, "estimated_eager_ms", None),
                "estimated_split_ms": getattr(
                    result, "estimated_graph_ms", None),
                "expected_speedup": getattr(result, "expected_speedup", 1.0),
                "reason": getattr(result, "reason", ""),
            })

    def claim_phase_separation(self, prefill_tokens: int,
                               decode_tokens: int) -> bool:
        key = (int(prefill_tokens), int(decode_tokens))
        with self._lock:
            if key in self._phase_separation_seen:
                return False
            self._phase_separation_seen.add(key)
            return True

    def record_phase_separation(self, mode: str, result: object) -> None:
        with self._lock:
            self._phase_separation_rows.append({
                "prefill_tokens": getattr(result, "prefill_tokens", 0),
                "decode_tokens": getattr(result, "decode_tokens", 0),
                "mode": mode,
                "strategy": getattr(result, "strategy", "unknown"),
                "decode_graph_size": getattr(
                    result, "decode_graph_size", 0),
                "estimated_mixed_ms": getattr(
                    result, "estimated_mixed_ms", None),
                "estimated_prefill_ms": getattr(
                    result, "estimated_prefill_ms", None),
                "estimated_decode_ms": getattr(
                    result, "estimated_decode_ms", None),
                "estimated_total_ms": getattr(
                    result, "estimated_total_ms", None),
                "expected_speedup": getattr(result, "expected_speedup", 1.0),
                "reason": getattr(result, "reason", ""),
            })

    def record_cross_phase_replay(self, phase: str, raw_shape: int,
                                  graph_shape: int) -> None:
        key = (phase, int(raw_shape), int(graph_shape))
        with self._lock:
            self._cross_phase_replay_counts[key] = (
                self._cross_phase_replay_counts.get(key, 0) + 1)

    def _record_decision_locked(self, decision: RuntimeDecision) -> None:
        if not self.telemetry_enabled:
            return
        key = (decision.phase, decision.raw_shape, decision.target_shape,
               decision.execution_shape, decision.reason)
        self._decision_counts[key] = self._decision_counts.get(key, 0) + 1

    def register_ready_shapes(
        self,
        shapes: Iterable[int],
        phase: str = "decode",
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            for shape in shapes:
                key = (phase, int(shape))
                record = self._records.setdefault(
                    key, GraphRecord(phase=phase, shape=int(shape)))
                record.state = GraphState.READY
                record.first_seen_at = record.first_seen_at or now
                record.last_seen_at = now

    def plan_runtime(
        self,
        phase: str,
        raw_shape: int,
        target_shape: int,
        capturable: bool,
        already_captured: bool,
        allow_replay: bool = True,
    ) -> RuntimeDecision:
        if not self.enabled:
            return RuntimeDecision(phase, raw_shape, target_shape, target_shape,
                                   False, False, "adaptive-disabled")

        if already_captured:
            self.register_ready_shapes([target_shape], phase)

        key = (phase, target_shape)
        now = time.time()
        with self._lock:
            record = self._records.setdefault(
                key, GraphRecord(phase=phase, shape=target_shape))
            record.hit_count += 1
            record.first_seen_at = record.first_seen_at or now
            record.last_seen_at = now
            if record.state == GraphState.UNSEEN:
                record.state = GraphState.OBSERVED

            if record.state == GraphState.READY:
                if allow_replay:
                    decision = RuntimeDecision(phase, raw_shape, target_shape,
                                               target_shape, False, False,
                                               "ready-replay")
                else:
                    decision = RuntimeDecision(phase, raw_shape, target_shape,
                                               raw_shape, True, False,
                                               "cost-policy-eager")
                self._record_decision_locked(decision)
                return decision

            if not self._eligible_phase(phase):
                decision = RuntimeDecision(phase, raw_shape, target_shape,
                                           raw_shape, True, False,
                                           "phase-not-yet-supported")
                self._record_decision_locked(decision)
                return decision

            if not capturable:
                decision = RuntimeDecision(phase, raw_shape, target_shape,
                                           raw_shape, True, False,
                                           "shape-not-capturable")
                self._record_decision_locked(decision)
                return decision

            if (record.state == GraphState.OBSERVED
                    and record.hit_count >= self.min_hits):
                record.state = GraphState.WARMING
                decision = RuntimeDecision(
                    phase,
                    raw_shape,
                    target_shape,
                    target_shape,
                    True,
                    True,
                    "threshold-runtime-warmup",
                )
                self._record_decision_locked(decision)
                return decision

            decision = RuntimeDecision(phase, raw_shape, target_shape,
                                       raw_shape, True, False,
                                       record.state.value)
            self._record_decision_locked(decision)
            return decision

    def reserve_gpu_sample(self, phase: str, mode: str,
                           execution_shape: int) -> bool:
        if not self.enabled or not self.profile_gpu:
            return False
        key = (phase, mode, int(execution_shape))
        with self._lock:
            completed = len(self._gpu_samples.get(key, ()))
            reserved = self._reserved_gpu_samples.get(key, 0)
            if completed + reserved >= self.max_gpu_samples_per_key:
                return False
            self._reserved_gpu_samples[key] = reserved + 1
            return True

    def add_gpu_sample(self, phase: str, mode: str, execution_shape: int,
                       start_event: object, end_event: object) -> None:
        with self._lock:
            self._pending_gpu_samples.append(
                (phase, mode, int(execution_shape), start_event, end_event))

    def flush_gpu_samples(self) -> None:
        with self._lock:
            pending = self._pending_gpu_samples
            self._pending_gpu_samples = []
        completed: list[tuple[tuple[str, str, int], float]] = []
        resolved_keys: list[tuple[str, str, int]] = []
        for phase, mode, shape, start_event, end_event in pending:
            key = (phase, mode, shape)
            resolved_keys.append(key)
            try:
                end_event.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
                completed.append((key, elapsed_ms))
            except BaseException:
                logger.exception(
                    "Failed to resolve adaptive CUDA graph GPU timing: "
                    "phase=%s mode=%s shape=%d", phase, mode, shape)
        if not resolved_keys:
            return
        with self._lock:
            for key, elapsed_ms in completed:
                self._gpu_samples.setdefault(key, []).append(elapsed_ms)
            for key in resolved_keys:
                self._reserved_gpu_samples[key] = max(
                    0, self._reserved_gpu_samples.get(key, 1) - 1)

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def export_telemetry(self) -> None:
        if not self.telemetry_enabled:
            return
        output_dir_text = (
            os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_TELEMETRY_DIR")
            or os.environ.get("VLLM_CUDA_GRAPH_HOTSPOT_DIR"))
        if not output_dir_text:
            return
        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            decisions = list(self._decision_counts.items())
            gpu_samples = {
                key: list(values) for key, values in self._gpu_samples.items()
            }
            records = [vars(record).copy()
                       for record in self._records.values()]
            split_shadow_rows = [dict(row)
                                 for row in self._split_shadow_rows]
            phase_separation_rows = [dict(row)
                                     for row in self._phase_separation_rows]
            cross_phase_counts = list(self._cross_phase_replay_counts.items())
            capture_chunk_rows = [dict(row)
                                  for row in self._capture_chunk_rows]

        decision_path = output_dir / "adaptive_cudagraph_decisions.csv"
        with decision_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "phase", "raw_shape", "target_shape", "execution_shape",
                "reason", "count", "padding_tokens"
            ])
            for key, count in sorted(decisions):
                phase, raw_shape, target_shape, execution_shape, reason = key
                writer.writerow([
                    phase, raw_shape, target_shape, execution_shape, reason,
                    count, (execution_shape - raw_shape) * count
                ])

        cost_path = output_dir / "adaptive_cudagraph_gpu_costs.csv"
        with cost_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "phase", "mode", "execution_shape", "samples",
                "mean_gpu_ms", "p50_gpu_ms", "p90_gpu_ms"
            ])
            for (phase, mode, shape), values in sorted(gpu_samples.items()):
                writer.writerow([
                    phase, mode, shape, len(values),
                    sum(values) / len(values),
                    self._percentile(values, 0.50),
                    self._percentile(values, 0.90),
                ])

        split_path = output_dir / "adaptive_cudagraph_split_shadow.csv"
        split_fields = [
            "phase", "raw_shape", "mode", "strategy", "request_sizes",
            "graph_sizes", "estimated_eager_ms", "estimated_split_ms",
            "expected_speedup", "reason"
        ]
        with split_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=split_fields)
            writer.writeheader()
            writer.writerows(sorted(
                split_shadow_rows,
                key=lambda row: (str(row["phase"]), int(row["raw_shape"]),
                                 str(row["mode"])),
            ))

        phase_path = output_dir / "adaptive_cudagraph_phase_separation.csv"
        phase_fields = [
            "prefill_tokens", "decode_tokens", "mode", "strategy",
            "decode_graph_size", "estimated_mixed_ms",
            "estimated_prefill_ms", "estimated_decode_ms",
            "estimated_total_ms", "expected_speedup", "reason"
        ]
        with phase_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=phase_fields)
            writer.writeheader()
            writer.writerows(sorted(
                phase_separation_rows,
                key=lambda row: (int(row["prefill_tokens"]),
                                 int(row["decode_tokens"]), str(row["mode"])),
            ))

        cross_phase_path = output_dir / "adaptive_cudagraph_cross_phase.csv"
        with cross_phase_path.open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "phase", "raw_shape", "graph_shape", "count",
                "padding_tokens"
            ])
            for (phase, raw_shape, graph_shape), count in sorted(
                    cross_phase_counts):
                writer.writerow([
                    phase, raw_shape, graph_shape, count,
                    (graph_shape - raw_shape) * count
                ])

        for record in records:
            state = record.get("state")
            if isinstance(state, GraphState):
                record["state"] = state.value
        (output_dir / "adaptive_cudagraph_states.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8")

        chunk_path = output_dir / "adaptive_cudagraph_capture_chunks.csv"
        chunk_fields = [
            "phase", "shape", "mode", "captured_piece_ids",
            "captured_piece_count", "deferred_piece_count", "total_pieces",
            "elapsed_ms", "context_enter_ms", "body_ms", "context_exit_ms",
            "piece_capture_ms", "admitted_deadline_ms"
        ]
        with chunk_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=chunk_fields)
            writer.writeheader()
            writer.writerows(capture_chunk_rows)

        with self._lock:
            maintenance_rows = [dict(row)
                                for row in self._maintenance_plan_rows]
        maintenance_path = output_dir / (
            "adaptive_cudagraph_maintenance_plans.csv")
        maintenance_fields = [
            "phase", "shape", "piece_budget", "predicted_ms",
            "fixed_cost_p90_ms", "piece_cost_p90_ms", "hit_count",
            "remaining_pieces", "value_cost_score", "reason",
            "admitted_slack_ms", "timestamp",
        ]
        with maintenance_path.open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=maintenance_fields)
            writer.writeheader()
            writer.writerows(maintenance_rows)

    def finish_runtime_warmup(self, phase: str, shape: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            record = self._records.get((phase, shape))
            if record is None or record.state != GraphState.WARMING:
                return
            record.state = GraphState.QUEUED
        logger.info(
            "Adaptive CUDA graph shape queued after real-request warmup: "
            "phase=%s shape=%d", phase, shape)

    def take_queued_shapes(
        self,
        phase: str = "decode",
        limit: Optional[int] = None,
        selected_shape: Optional[int] = None,
    ) -> list[int]:
        if not self.enabled:
            return []
        with self._lock:
            queued = [
                record for record in self._records.values()
                if record.phase == phase and record.state in (
                    GraphState.QUEUED, GraphState.PARTIAL)
            ]
            if selected_shape is not None:
                queued = [record for record in queued
                          if record.shape == int(selected_shape)]
            queued.sort(key=lambda record: (record.hit_count, record.shape),
                        reverse=True)
            if self._planned_capture_shape is not None:
                planned_shape = self._planned_capture_shape
                queued.sort(
                    key=lambda record: record.shape == planned_shape,
                    reverse=True,
                )
                self._planned_capture_shape = None
            selected = queued[:limit] if limit is not None else queued
            for record in selected:
                record.state = GraphState.CAPTURING
            return [record.shape for record in selected]

    def pending_shapes(self, phase: str = "decode") -> list[int]:
        """Return queued shapes without mutating their capture state."""
        if not self.enabled:
            return []
        with self._lock:
            queued = [
                record for record in self._records.values()
                if record.phase == phase and record.state in (
                    GraphState.QUEUED, GraphState.PARTIAL)
            ]
            queued.sort(key=lambda record: (record.hit_count, record.shape),
                        reverse=True)
            return [record.shape for record in queued]

    def plan_capture_maintenance(
        self,
        slack_ms: float,
        max_pieces: int,
        default_fixed_ms: float,
        default_piece_ms: float,
        safety_factor: float = 1.05,
    ) -> Optional[CaptureMaintenancePlan]:
        """Select the hottest capture unit whose measured P90 fits slack."""
        if not self.enabled or slack_ms <= 0.0 or max_pieces <= 0:
            return None
        with self._lock:
            candidates = [
                GraphRecord(**vars(record))
                for record in self._records.values()
                if record.phase == "decode" and record.state in (
                    GraphState.QUEUED, GraphState.PARTIAL)
            ]
            piece_values = [
                value for values in self._piece_capture_history_ms.values()
                for value in values
            ]
            fixed_values = list(self._fixed_capture_history_ms)

        fixed_p90 = (self._percentile(fixed_values, 0.90)
                     if fixed_values else max(0.0, default_fixed_ms))
        piece_p90 = (self._percentile(piece_values, 0.90)
                     if piece_values else max(0.01, default_piece_ms))
        fixed_p90 *= max(1.0, safety_factor)
        piece_p90 *= max(1.0, safety_factor)
        usable_ms = slack_ms - fixed_p90
        if usable_ms < piece_p90:
            return None

        affordable_pieces = min(
            max_pieces, max(1, int(usable_ms // piece_p90)))
        plans: list[CaptureMaintenancePlan] = []
        for record in candidates:
            remaining = (record.remaining_pieces
                         if record.remaining_pieces > 0 else max_pieces)
            pieces = min(affordable_pieces, remaining)
            predicted_ms = fixed_p90 + pieces * piece_p90
            if predicted_ms > slack_ms:
                continue
            score = max(1, record.hit_count) / predicted_ms
            plans.append(CaptureMaintenancePlan(
                phase=record.phase,
                shape=record.shape,
                piece_budget=pieces,
                predicted_ms=predicted_ms,
                fixed_cost_p90_ms=fixed_p90,
                piece_cost_p90_ms=piece_p90,
                hit_count=record.hit_count,
                remaining_pieces=remaining,
                value_cost_score=score,
            ))
        if not plans:
            return None
        plan = max(plans, key=lambda item: (
            item.value_cost_score, item.hit_count, item.shape))
        with self._lock:
            self._planned_capture_shape = plan.shape
            self._maintenance_plan_rows.append({
                **vars(plan),
                "admitted_slack_ms": slack_ms,
                "timestamp": time.time(),
            })
        return plan

    def mark_capture_ready(
        self,
        phase: str,
        shape: int,
        capture_seconds: float,
        context_enter_seconds: float = 0.0,
        body_seconds: float = 0.0,
        context_exit_seconds: float = 0.0,
        captured_pieces: int = 0,
    ) -> None:
        with self._lock:
            record = self._records[(phase, shape)]
            record.state = GraphState.READY
            record.capture_seconds += capture_seconds
            record.capture_finished_at = time.time()
            record.capture_context_enter_seconds += context_enter_seconds
            record.capture_body_seconds += body_seconds
            record.capture_context_exit_seconds += context_exit_seconds
            record.capture_attempts += 1
            record.captured_pieces += int(captured_pieces)
            record.last_capture_chunk_seconds = capture_seconds
            record.remaining_pieces = 0
            record.failure = None

    def mark_capture_started(self, phase: str, shape: int) -> None:
        """Record capture start for both idle and concurrent workers."""
        with self._lock:
            record = self._records[(phase, shape)]
            record.capture_started_at = time.time()

    def mark_capture_partial(
        self,
        phase: str,
        shape: int,
        capture_seconds: float,
        captured_pieces: int,
        remaining_pieces: int,
        context_enter_seconds: float = 0.0,
        body_seconds: float = 0.0,
        context_exit_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            record = self._records[(phase, shape)]
            record.state = GraphState.PARTIAL
            record.capture_seconds += capture_seconds
            record.capture_context_enter_seconds += context_enter_seconds
            record.capture_body_seconds += body_seconds
            record.capture_context_exit_seconds += context_exit_seconds
            record.capture_attempts += 1
            record.captured_pieces += int(captured_pieces)
            record.remaining_pieces = int(remaining_pieces)
            record.last_capture_chunk_seconds = capture_seconds
            record.failure = None

    def mark_capture_failed(self, phase: str, shape: int,
                            error: BaseException) -> None:
        with self._lock:
            record = self._records[(phase, shape)]
            record.state = GraphState.FAILED
            record.capture_finished_at = time.time()
            record.failure = repr(error)

    def capture_pending(
        self,
        runner: object,
        max_pieces_per_shape: Optional[int] = None,
        deadline_ms: Optional[float] = None,
        selected_shape: Optional[int] = None,
    ) -> int:
        """Capture queued graphs, optionally in bounded piecewise chunks."""
        if not self.enabled:
            return 0
        if (max_pieces_per_shape is not None
                and getattr(runner, "full_cuda_graph", False)):
            # Full graphs bind attention metadata into one capture unit and
            # cannot use the piecewise compiler's incremental budget.
            return 0
        shapes = self.take_queued_shapes(
            limit=(1 if selected_shape is not None else
                   self.max_captures_per_idle),
            selected_shape=selected_shape)
        if not shapes:
            return 0

        # Import lazily so controller state tests do not initialize CUDA.
        from vllm.distributed.parallel_state import graph_capture
        from vllm.forward_context import (
            CUDAGraphCaptureBudget,
            CUDAGraphCaptureStop,
        )

        logger.info("Adaptive CUDA graph capture started: shapes=%s",
                    shapes)
        progressed = 0
        with runner.manager.fast_capture_scope():
            for shape in shapes:
                started = time.perf_counter()
                with self._lock:
                    self._records[("decode", shape)].capture_started_at = (
                        time.time())
                try:
                    capture_budget = None
                    if max_pieces_per_shape is not None:
                        deadline_ns = None
                        if deadline_ms is not None:
                            deadline_ns = (time.perf_counter_ns() + int(
                                max(0.0, deadline_ms) * 1_000_000))
                        capture_budget = CUDAGraphCaptureBudget(
                            max_pieces=max(1, int(max_pieces_per_shape)),
                            deadline_ns=deadline_ns,
                            stop_after_budget=_env_flag(
                                "VLLM_ADAPTIVE_CUDAGRAPH_EARLY_STOP", True),
                        )
                    with graph_capture(device=runner.device):
                        context_entered = time.perf_counter()
                        if capture_budget is None:
                            for _ in range(self.extra_warmups):
                                runner._dummy_run(
                                    shape,
                                    capture_attn_cudagraph=(
                                        runner.full_cuda_graph),
                                    skip_eplb=True,
                                )
                        try:
                            runner._dummy_run(
                                shape,
                                capture_attn_cudagraph=runner.full_cuda_graph,
                                skip_eplb=True,
                                cudagraph_capture_budget=capture_budget,
                            )
                        except CUDAGraphCaptureStop:
                            logger.debug(
                                "Stopped dummy forward after bounded CUDA "
                                "graph capture: shape=%d pieces=%d", shape,
                                len(capture_budget.captured_piece_ids))
                        body_finished = time.perf_counter()
                    finished = time.perf_counter()
                    elapsed = finished - started
                    with self._lock:
                        if capture_budget is not None:
                            piece_total_ms = sum(
                                seconds * 1000.0 for seconds in
                                capture_budget.piece_capture_seconds.values())
                            fixed_capture_ms = max(
                                0.0,
                                elapsed * 1000.0 - piece_total_ms,
                            )
                            self._fixed_capture_history_ms.append(
                                fixed_capture_ms)
                            del self._fixed_capture_history_ms[:-32]
                            for piece_id, seconds in (
                                    capture_budget.piece_capture_seconds.items()):
                                samples = self._piece_capture_history_ms.setdefault(
                                    int(piece_id), [])
                                samples.append(seconds * 1000.0)
                                del samples[:-16]
                        self._capture_chunk_rows.append({
                            "phase": "decode",
                            "shape": shape,
                            "mode": ("full" if capture_budget is None else
                                     "incremental"),
                            "captured_piece_ids": (
                                "" if capture_budget is None else ";".join(
                                    str(piece_id) for piece_id in sorted(
                                        capture_budget.captured_piece_ids))),
                            "captured_piece_count": (
                                0 if capture_budget is None else len(
                                    capture_budget.captured_piece_ids)),
                            "deferred_piece_count": (
                                0 if capture_budget is None else len(
                                    capture_budget.deferred_piece_ids)),
                            "total_pieces": (
                                0 if capture_budget is None else
                                capture_budget.total_piecewise_compiles),
                            "elapsed_ms": elapsed * 1000.0,
                            "context_enter_ms": (
                                context_entered - started) * 1000.0,
                            "body_ms": (
                                body_finished - context_entered) * 1000.0,
                            "context_exit_ms": (
                                finished - body_finished) * 1000.0,
                            "piece_capture_ms": (
                                "" if capture_budget is None else ";".join(
                                    f"{piece_id}:{seconds * 1000.0:.6f}"
                                    for piece_id, seconds in sorted(
                                        capture_budget.
                                        piece_capture_seconds.items()))),
                            "admitted_deadline_ms": (
                                "" if deadline_ms is None else deadline_ms),
                        })
                    capture_complete = (capture_budget is None
                                        or capture_budget.pass_complete)
                    made_progress = (capture_budget is None
                                     or capture_budget.made_progress)
                    if capture_complete:
                        if shape not in runner.cuda_graph_captured:
                            runner.cuda_graph_captured.append(shape)
                            runner.cuda_graph_captured.sort()
                        graph_matcher = getattr(runner, "graphMatcher", None)
                        if graph_matcher is not None:
                            graph_matcher.update_captured_sizes(
                                runner.cuda_graph_captured)
                        dynamic_shapes = getattr(
                            runner, "dynamic_cuda_graph_captured", None)
                        if dynamic_shapes is not None:
                            dynamic_shapes.add(shape)
                        self.mark_capture_ready(
                            "decode",
                            shape,
                            elapsed,
                            context_enter_seconds=context_entered - started,
                            body_seconds=body_finished - context_entered,
                            context_exit_seconds=finished - body_finished,
                            captured_pieces=(
                                0 if capture_budget is None else len(
                                    capture_budget.captured_piece_ids)),
                        )
                        logger.info(
                            "Adaptive CUDA graph became ready: shape=%d "
                            "capture_seconds=%.6f enter=%.6f body=%.6f "
                            "exit=%.6f", shape, elapsed,
                            context_entered - started,
                            body_finished - context_entered,
                            finished - body_finished)
                    else:
                        assert capture_budget is not None
                        self.mark_capture_partial(
                            "decode",
                            shape,
                            elapsed,
                            captured_pieces=len(
                                capture_budget.captured_piece_ids),
                            remaining_pieces=len(
                                capture_budget.deferred_piece_ids),
                            context_enter_seconds=context_entered - started,
                            body_seconds=body_finished - context_entered,
                            context_exit_seconds=finished - body_finished,
                        )
                        logger.info(
                            "Adaptive CUDA graph capture progressed: "
                            "shape=%d captured_pieces=%d remaining_pieces=%d "
                            "chunk_seconds=%.6f", shape,
                            len(capture_budget.captured_piece_ids),
                            len(capture_budget.deferred_piece_ids), elapsed)
                    if made_progress:
                        progressed += 1
                except BaseException as error:
                    self.mark_capture_failed("decode", shape, error)
                    logger.exception(
                        "Adaptive CUDA graph capture failed: shape=%d", shape)
        return progressed

    def snapshot(self) -> list[GraphRecord]:
        with self._lock:
            return [GraphRecord(**vars(record))
                    for record in self._records.values()]


adaptive_cudagraph_controller = AdaptiveCUDAGraphController()
