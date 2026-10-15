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


class CUDAGraphManager:
    """Per-engine entry point for precapture, policy and capture lifecycle.

    The runner supplies model execution and attention metadata. This manager
    owns policy/state, while its executor and registry own capture and handles.
    A runner is optional for offline lifecycle tests.
    """

    def __init__(
        self,
        runner=None,
        *,
        enabled: Optional[bool] = None,
        min_hits: Optional[int] = None,
    ) -> None:
        self.enabled = (
            _env_flag("VLLM_ENABLE_ADAPTIVE_CUDAGRAPH", True)
            if enabled is None
            else enabled
        )
        from vllm.v1.worker.HotspotAnalyzer import HotspotAnalyzer
        from vllm.v1.worker.piecewise_graph_registry import (
            PiecewiseGraphRegistry,
        )

        self.runner = runner
        self.hotspots = HotspotAnalyzer(
            min_hits=(
                int(os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS", "1"))
                if min_hits is None
                else min_hits
            )
        )
        self.registry = PiecewiseGraphRegistry(
            getattr(runner, "compilation_config", None)
        )
        self.capture = None
        self.adapter = None
        self.composites = None
        self._closed = False
        self.extra_warmups = int(
            os.environ.get("VLLM_ADAPTIVE_CUDAGRAPH_EXTRA_WARMUPS", "0")
        )
        self.max_captures_per_idle = max(
            1,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_MAX_CAPTURES_PER_IDLE", "1"
                )
            ),
        )
        self._records: dict[tuple[str, int], GraphRecord] = {}
        self._lock = threading.Lock()
        self.telemetry_enabled = _env_flag(
            "VLLM_ADAPTIVE_CUDAGRAPH_TELEMETRY", True
        )
        self.profile_gpu = _env_flag(
            "VLLM_ADAPTIVE_CUDAGRAPH_PROFILE_GPU", False
        )
        self.max_gpu_samples_per_key = max(
            1,
            int(
                os.environ.get(
                    "VLLM_ADAPTIVE_CUDAGRAPH_GPU_SAMPLES_PER_KEY", "16"
                )
            ),
        )
        self._decision_counts: dict[tuple[str, int, int, int, str], int] = {}
        self._cross_phase_replay_counts: dict[tuple[str, int, int], int] = {}
        self._pending_gpu_samples: list[
            tuple[str, str, int, object, object]
        ] = []
        self._reserved_gpu_samples: dict[tuple[str, str, int], int] = {}
        self._gpu_samples: dict[tuple[str, str, int], list[float]] = {}
        self._capture_chunk_rows: list[dict[str, object]] = []
        self._piece_capture_history_ms: dict[int, list[float]] = {}
        self._fixed_capture_history_ms: list[float] = []
        self._maintenance_plan_rows: list[dict[str, object]] = []
        self._planned_capture_shape: Optional[int] = None
        self.capture_phases = frozenset({"decode"})
        if runner is not None:
            self._configure()

    @property
    def min_hits(self):
        return self.hotspots.min_hits

    @min_hits.setter
    def min_hits(self, value):
        self.hotspots.min_hits = max(1, int(value))

    def _configure(self):
        from vllm import __version__
        from vllm.compilation.startup_assets import digest
        from vllm.compilation.startup_resources import resource_dir, run_dir
        from vllm.v1.worker.cudagraph_profile import (
            build_cudagraph_profile_manifest,
            ensure_cudagraph_profile_manifest,
            validate_cudagraph_profile_manifest,
        )
        from vllm.v1.worker.cudagraph_adaptation import GraphAdapter
        from vllm.v1.worker.concurrent_piecewise_cudagraph import (
            CaptureExecutor,
        )
        from vllm.v1.worker.LazyFreezeManager import LazyFreezeManager
        import torch

        r = self.runner
        self.gc_scope = LazyFreezeManager.get_instance()
        self.profile_manifest = build_cudagraph_profile_manifest(
            model_config=r.model_config,
            cache_config=r.cache_config,
            scheduler_config=r.scheduler_config,
            parallel_config=r.parallel_config,
            compilation_config=r.compilation_config,
            device_properties=r.device_properties,
            dtype=r.dtype,
            torch_version=torch.__version__,
            vllm_version=__version__,
        )
        self.profile_dir = Path(
            os.getenv(
                "VLLM_CUDA_GRAPH_HOTSPOT_DIR",
                str(resource_dir("hotspots") / digest(self.profile_manifest)),
            )
        )
        ensure_cudagraph_profile_manifest(
            self.profile_dir,
            self.profile_manifest,
            allow_overwrite=_env_flag("VLLM_CUDAGRAPH_PROFILE_ALLOW_OVERWRITE"),
        )
        # Observations are separate from frozen offline inputs.
        self.hotspots.output_dir = run_dir() / "hotspots" / digest(self.profile_manifest)
        self.hotspots.profile_manifest = self.profile_manifest
        ensure_cudagraph_profile_manifest(
            self.hotspots.output_dir, self.profile_manifest
        )
        candidates = [
            self.profile_dir / name
            for name in (
                "cuda_graph_cost_table.csv",
                "adaptive_cudagraph_gpu_costs.csv",
            )
        ]
        if os.getenv("VLLM_CUDAGRAPH_COST_TABLE"):
            candidates.insert(0, Path(os.environ["VLLM_CUDAGRAPH_COST_TABLE"]))
        cost_path = None
        for candidate in candidates:
            if not candidate.is_file():
                continue
            valid, reason = validate_cudagraph_profile_manifest(
                candidate.resolve().parent, self.profile_manifest
            )
            if valid:
                cost_path = str(candidate)
                break
            logger.warning("Ignoring graph costs %s: %s", candidate, reason)
        self.adapter = GraphAdapter(
            r,
            registry=self.registry,
            cost_table_path=cost_path,
            max_pad_ratio=float(
                os.getenv("VLLM_CUDAGRAPH_MAX_PAD_RATIO", ".2")
            ),
            min_estimated_saving_ratio=float(
                os.getenv("VLLM_CUDAGRAPH_MIN_ESTIMATED_SAVING_RATIO", "0")
            ),
        )
        self.cross_phase = not r.full_cuda_graph and _env_flag(
            "VLLM_CUDAGRAPH_DYNAMIC_CROSS_PHASE", self.enabled
        )
        self.capture = CaptureExecutor(self)
        if self.capture.enabled:
            self.capture_phases = frozenset({"decode", "prefill", "mixed"})
        for old in (
            "VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE",
            "VLLM_DYNAMIC_CUDA_GRAPH_CAPTURE_NO_REPLAY",
            "VLLM_CUDAGRAPH_SPLIT_SHADOW",
        ):
            if old in os.environ:
                logger.info(
                    "%s belongs to the retired legacy/shadow path and is ignored; "
                    "VLLM_ENABLE_ADAPTIVE_CUDAGRAPH controls on-demand capture",
                    old,
                )

    def precapture(self):
        if not self.runner.use_cuda_graph:
            return
        shapes = self.hotspots.select_precapture(
            self.runner.cudagraph_batch_sizes,
            self.profile_dir,
            self.profile_manifest,
            adaptive=self.enabled,
        )
        self.capture.precapture(shapes)
        self.register_ready_shapes(shapes)

    def select(
        self,
        phase,
        raw,
        metadata,
        query_lens,
        seq_lens,
        is_prefill,
        *,
        allow_split=True,
    ):
        """One observation and one final execution decision per scheduler batch."""
        from vllm.v1.worker.cudagraph_adaptation import workload_key

        r = self.runner
        capturable = bool(
            r.use_cuda_graph
            and r.cudagraph_batch_sizes
            and raw <= r.cudagraph_batch_sizes[-1]
        )
        target = r.vllm_config.pad_for_cudagraph(raw) if capturable else raw
        self.hotspots.observe(raw, target)
        if not r.use_cuda_graph or not self.enabled:
            replay = capturable and target in self.registry.published
            return RuntimeDecision(
                phase,
                raw,
                target,
                target if replay else raw,
                not replay,
                False,
                "static-replay" if replay else "eager",
            ), ()
        match = self.adapter.find_best_match(
            raw,
            phase=phase,
            default_graph_size=target,
            dynamic_sizes=self.registry.dynamic_shapes,
            workload=workload_key(query_lens, seq_lens, is_prefill),
            max_pad_ratio=self.adapter.pad_limit(),
            allow_dynamic_cross_phase=self.cross_phase,
        )
        parts = ()
        if match.strategy == "eager":
            if allow_split:
                parts = self.adapter.plan(
                    metadata, query_lens, seq_lens, is_prefill
                )
        else:
            target = match.execution_size
        decision = self.plan_runtime(
            phase,
            raw,
            target,
            capturable=capturable
            and not (
                match.reason.startswith("cost-reject")
                and match.graph_size == target
            ),
            already_captured=target in self.registry.published,
            allow_replay=match.strategy != "eager",
            adapted=bool(parts),
        )
        if decision.mark_runtime_warmup or not decision.skip_cudagraph:
            parts = ()
        return decision, parts

    def after_warmup(self, decision):
        self.finish_runtime_warmup(decision.phase, decision.target_shape)
        self.capture.enqueue_after_warmup(decision.phase, decision.target_shape)

    def on_idle(self):
        # Multi-rank capture is admitted collectively by EngineCore, never by
        # an independently idle rank.
        if self.runner.parallel_config.world_size == 1:
            self.capture_pending()

    def capture_pending(self, **budget):
        if self.capture is None or self.capture.enabled or self._closed:
            return 0
        return self.capture.capture_pending(**budget)

    def before_rebind(self):
        self.capture.shutdown()
        self.hotspots.check_and_export(True)
        self.flush_gpu_samples()
        self.export_telemetry()
        # The engine synchronizes before releasing graph-owned storage.
        import torch

        torch.cuda.synchronize(self.runner.device)
        self.composites = None

    def after_rebind(self):
        from vllm.v1.worker.HotspotAnalyzer import HotspotAnalyzer

        old_published = self.registry.published
        self.hotspots = HotspotAnalyzer(
            min_hits=self.min_hits, output_dir=self.hotspots.output_dir
        )
        self.reset()
        # In-place weight rebind retains validated addresses, but profile
        # identity/configuration and observation generation must be refreshed.
        self._configure()
        self.register_ready_shapes(old_published)

    def shutdown(self):
        if self._closed:
            return
        self.before_rebind()
        self.registry.clear()
        self._closed = True

    def _eligible_phase(self, phase: str) -> bool:
        return phase in self.capture_phases

    def settle_shared_capture(
        self, phase: str, shape: int, error: Optional[BaseException] = None
    ) -> None:
        """Resolve other phase records sharing one piecewise shape capture."""
        with self._lock:
            for (other_phase, other_shape), record in self._records.items():
                if other_shape != shape or other_phase == phase:
                    continue
                if record.state in (
                    GraphState.OBSERVED,
                    GraphState.WARMING,
                    GraphState.QUEUED,
                    GraphState.CAPTURING,
                ):
                    record.state = (
                        GraphState.READY if error is None else GraphState.FAILED
                    )
                    record.failure = None if error is None else repr(error)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._decision_counts.clear()
            self._cross_phase_replay_counts.clear()
            self._pending_gpu_samples.clear()
            self._reserved_gpu_samples.clear()
            self._gpu_samples.clear()
            self._capture_chunk_rows.clear()
            self._piece_capture_history_ms.clear()
            self._fixed_capture_history_ms.clear()
            self._maintenance_plan_rows.clear()
            self._planned_capture_shape = None

    def record_cross_phase_replay(
        self, phase: str, raw_shape: int, graph_shape: int
    ) -> None:
        key = (phase, int(raw_shape), int(graph_shape))
        with self._lock:
            self._cross_phase_replay_counts[key] = (
                self._cross_phase_replay_counts.get(key, 0) + 1
            )

    def _record_decision_locked(self, decision: RuntimeDecision) -> None:
        if not self.telemetry_enabled:
            return
        key = (
            decision.phase,
            decision.raw_shape,
            decision.target_shape,
            decision.execution_shape,
            decision.reason,
        )
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
                    key, GraphRecord(phase=phase, shape=int(shape))
                )
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
        adapted: bool = False,
    ) -> RuntimeDecision:
        if not self.enabled:
            return RuntimeDecision(
                phase,
                raw_shape,
                target_shape,
                target_shape,
                False,
                False,
                "adaptive-disabled",
            )

        if already_captured:
            self.register_ready_shapes([target_shape], phase)

        key = (phase, target_shape)
        with self._lock:
            record = self._records.setdefault(
                key, GraphRecord(phase=phase, shape=target_shape)
            )
            self.hotspots.record_target(record)
            if record.state == GraphState.UNSEEN:
                record.state = GraphState.OBSERVED

            if record.state == GraphState.READY:
                if allow_replay:
                    decision = RuntimeDecision(
                        phase,
                        raw_shape,
                        target_shape,
                        target_shape,
                        False,
                        False,
                        "ready-replay",
                    )
                else:
                    decision = RuntimeDecision(
                        phase,
                        raw_shape,
                        target_shape,
                        raw_shape,
                        True,
                        False,
                        "split-reuse" if adapted else "cost-policy-eager",
                    )
                self._record_decision_locked(decision)
                return decision

            if not self._eligible_phase(phase):
                decision = RuntimeDecision(
                    phase,
                    raw_shape,
                    target_shape,
                    raw_shape,
                    True,
                    False,
                    "split-reuse" if adapted else "phase-not-yet-supported",
                )
                self._record_decision_locked(decision)
                return decision

            if not capturable:
                decision = RuntimeDecision(
                    phase,
                    raw_shape,
                    target_shape,
                    raw_shape,
                    True,
                    False,
                    "split-reuse" if adapted else "shape-not-capturable",
                )
                self._record_decision_locked(decision)
                return decision

            if self.capture is not None and self.capture.enabled:
                shared = [
                    r
                    for (p, s), r in self._records.items()
                    if s == target_shape and p != phase
                ]
                if any(
                    r.state
                    in (
                        GraphState.WARMING,
                        GraphState.QUEUED,
                        GraphState.CAPTURING,
                    )
                    for r in shared
                ):
                    decision = RuntimeDecision(
                        phase,
                        raw_shape,
                        target_shape,
                        raw_shape,
                        True,
                        False,
                        "split-reuse" if adapted else "shared-capture-pending",
                    )
                    self._record_decision_locked(decision)
                    return decision

            if (
                record.state == GraphState.OBSERVED
                and self.hotspots.should_capture(record)
            ):
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

            # Reusing smaller graphs must not starve the original shape of its
            # one real warmup. It remains eligible before/while capture runs.
            decision = RuntimeDecision(
                phase,
                raw_shape,
                target_shape,
                raw_shape,
                True,
                False,
                "split-reuse" if adapted else record.state.value,
            )
            self._record_decision_locked(decision)
            return decision

    def reserve_gpu_sample(
        self, phase: str, mode: str, execution_shape: int
    ) -> bool:
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

    def add_gpu_sample(
        self,
        phase: str,
        mode: str,
        execution_shape: int,
        start_event: object,
        end_event: object,
    ) -> None:
        with self._lock:
            self._pending_gpu_samples.append(
                (phase, mode, int(execution_shape), start_event, end_event)
            )

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
                    "phase=%s mode=%s shape=%d",
                    phase,
                    mode,
                    shape,
                )
        if not resolved_keys:
            return
        with self._lock:
            for key, elapsed_ms in completed:
                self._gpu_samples.setdefault(key, []).append(elapsed_ms)
            for key in resolved_keys:
                self._reserved_gpu_samples[key] = max(
                    0, self._reserved_gpu_samples.get(key, 1) - 1
                )

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
            or self.hotspots.output_dir
        )
        if not output_dir_text:
            return
        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            decisions = list(self._decision_counts.items())
            gpu_samples = {
                key: list(values) for key, values in self._gpu_samples.items()
            }
            records = [vars(record).copy() for record in self._records.values()]
            cross_phase_counts = list(self._cross_phase_replay_counts.items())
            capture_chunk_rows = [dict(row) for row in self._capture_chunk_rows]

        decision_path = output_dir / "adaptive_cudagraph_decisions.csv"
        with decision_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "phase",
                    "raw_shape",
                    "target_shape",
                    "execution_shape",
                    "reason",
                    "count",
                    "padding_tokens",
                ]
            )
            for key, count in sorted(decisions):
                phase, raw_shape, target_shape, execution_shape, reason = key
                writer.writerow(
                    [
                        phase,
                        raw_shape,
                        target_shape,
                        execution_shape,
                        reason,
                        count,
                        (execution_shape - raw_shape) * count,
                    ]
                )

        cost_path = output_dir / "adaptive_cudagraph_gpu_costs.csv"
        with cost_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "phase",
                    "mode",
                    "execution_shape",
                    "samples",
                    "mean_gpu_ms",
                    "p50_gpu_ms",
                    "p90_gpu_ms",
                ]
            )
            for (phase, mode, shape), values in sorted(gpu_samples.items()):
                writer.writerow(
                    [
                        phase,
                        mode,
                        shape,
                        len(values),
                        sum(values) / len(values),
                        self._percentile(values, 0.50),
                        self._percentile(values, 0.90),
                    ]
                )

        cross_phase_path = output_dir / "adaptive_cudagraph_cross_phase.csv"
        with cross_phase_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["phase", "raw_shape", "graph_shape", "count", "padding_tokens"]
            )
            for (phase, raw_shape, graph_shape), count in sorted(
                cross_phase_counts
            ):
                writer.writerow(
                    [
                        phase,
                        raw_shape,
                        graph_shape,
                        count,
                        (graph_shape - raw_shape) * count,
                    ]
                )

        for record in records:
            state = record.get("state")
            if isinstance(state, GraphState):
                record["state"] = state.value
        (output_dir / "adaptive_cudagraph_states.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )

        chunk_path = output_dir / "adaptive_cudagraph_capture_chunks.csv"
        chunk_fields = [
            "phase",
            "shape",
            "mode",
            "captured_piece_ids",
            "captured_piece_count",
            "deferred_piece_count",
            "total_pieces",
            "elapsed_ms",
            "context_enter_ms",
            "body_ms",
            "context_exit_ms",
            "piece_capture_ms",
            "admitted_deadline_ms",
        ]
        with chunk_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=chunk_fields)
            writer.writeheader()
            writer.writerows(capture_chunk_rows)

        with self._lock:
            maintenance_rows = [
                dict(row) for row in self._maintenance_plan_rows
            ]
        maintenance_path = output_dir / (
            "adaptive_cudagraph_maintenance_plans.csv"
        )
        maintenance_fields = [
            "phase",
            "shape",
            "piece_budget",
            "predicted_ms",
            "fixed_cost_p90_ms",
            "piece_cost_p90_ms",
            "hit_count",
            "remaining_pieces",
            "value_cost_score",
            "reason",
            "admitted_slack_ms",
            "timestamp",
        ]
        with maintenance_path.open("w", newline="", encoding="utf-8") as stream:
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
            "phase=%s shape=%d",
            phase,
            shape,
        )

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
                record
                for record in self._records.values()
                if record.phase == phase
                and record.state in (GraphState.QUEUED, GraphState.PARTIAL)
            ]
            if selected_shape is not None:
                queued = [
                    record
                    for record in queued
                    if record.shape == int(selected_shape)
                ]
            queued.sort(
                key=lambda record: (record.hit_count, record.shape),
                reverse=True,
            )
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
                record
                for record in self._records.values()
                if record.phase == phase
                and record.state in (GraphState.QUEUED, GraphState.PARTIAL)
            ]
            queued.sort(
                key=lambda record: (record.hit_count, record.shape),
                reverse=True,
            )
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
        if self._closed or (self.capture is not None and self.capture.enabled):
            return None
        if not self.enabled or slack_ms <= 0.0 or max_pieces <= 0:
            return None
        with self._lock:
            candidates = [
                GraphRecord(**vars(record))
                for record in self._records.values()
                if record.phase == "decode"
                and record.state in (GraphState.QUEUED, GraphState.PARTIAL)
            ]
            piece_values = [
                value
                for values in self._piece_capture_history_ms.values()
                for value in values
            ]
            fixed_values = list(self._fixed_capture_history_ms)

        fixed_p90 = (
            self._percentile(fixed_values, 0.90)
            if fixed_values
            else max(0.0, default_fixed_ms)
        )
        piece_p90 = (
            self._percentile(piece_values, 0.90)
            if piece_values
            else max(0.01, default_piece_ms)
        )
        fixed_p90 *= max(1.0, safety_factor)
        piece_p90 *= max(1.0, safety_factor)
        usable_ms = slack_ms - fixed_p90
        if usable_ms < piece_p90:
            return None

        affordable_pieces = min(max_pieces, max(1, int(usable_ms // piece_p90)))
        plans: list[CaptureMaintenancePlan] = []
        for record in candidates:
            remaining = (
                record.remaining_pieces
                if record.remaining_pieces > 0
                else max_pieces
            )
            pieces = min(affordable_pieces, remaining)
            predicted_ms = fixed_p90 + pieces * piece_p90
            if predicted_ms > slack_ms:
                continue
            score = max(1, record.hit_count) / predicted_ms
            plans.append(
                CaptureMaintenancePlan(
                    phase=record.phase,
                    shape=record.shape,
                    piece_budget=pieces,
                    predicted_ms=predicted_ms,
                    fixed_cost_p90_ms=fixed_p90,
                    piece_cost_p90_ms=piece_p90,
                    hit_count=record.hit_count,
                    remaining_pieces=remaining,
                    value_cost_score=score,
                )
            )
        if not plans:
            return None
        plan = max(
            plans,
            key=lambda item: (
                item.value_cost_score,
                item.hit_count,
                item.shape,
            ),
        )
        with self._lock:
            self._planned_capture_shape = plan.shape
            self._maintenance_plan_rows.append(
                {
                    **vars(plan),
                    "admitted_slack_ms": slack_ms,
                    "timestamp": time.time(),
                }
            )
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

    def mark_capture_failed(
        self, phase: str, shape: int, error: BaseException
    ) -> None:
        with self._lock:
            record = self._records[(phase, shape)]
            record.state = GraphState.FAILED
            record.capture_finished_at = time.time()
            record.failure = repr(error)

    def snapshot(self) -> list[GraphRecord]:
        with self._lock:
            return [
                GraphRecord(**vars(record)) for record in self._records.values()
            ]
