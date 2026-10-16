# SPDX-License-Identifier: Apache-2.0

"""Capture execution only: serial background queue or engine-admitted safe point."""

from __future__ import annotations

import inspect
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _CaptureRequest:
    phase: str
    shape: int
    warmup_ready: torch.cuda.Event
    queued_at: float


class CaptureExecutor:
    """Own one serial background capture worker for a GPUModelRunner."""

    def __init__(self, manager) -> None:
        self.manager = manager
        self.runner = runner = manager.runner
        self.enabled, self.disabled_reason = self._check_support(
            runner, manager.enabled
        )
        self._queue: queue.Queue[
            Optional[_CaptureRequest | Callable[[], None]]
        ] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopping = False
        self._scheduled_shapes: set[int] = set()
        self._graph_pools: dict[int, Any] = {}
        self._unpublished: dict[int, tuple] = {}
        self._worker_runtime_initialized = False
        if self.enabled:
            logger.warning(
                "Experimental concurrent piecewise CUDA graph capture is "
                "enabled for this worker."
            )
        elif _env_flag(
            "VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE", True
        ):
            logger.warning(
                "Concurrent piecewise CUDA graph capture requested but "
                "disabled: %s",
                self.disabled_reason,
            )

    @staticmethod
    def _check_support(
        runner: "GPUModelRunner", adaptive=True
    ) -> tuple[bool, str]:
        if not _env_flag(
            "VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE", True
        ):
            return False, "feature flag is off"
        if not adaptive:
            return False, "adaptive CUDA graph controller is disabled"
        if not getattr(runner, "use_cuda_graph", False):
            return False, "CUDA graph execution is disabled"
        if getattr(runner, "full_cuda_graph", False):
            return False, "full CUDA graphs are not piecewise graphs"
        parallel = runner.parallel_config
        if parallel.data_parallel_size != 1:
            return False, "DP background capture is not supported"
        if parallel.tensor_parallel_size * parallel.pipeline_parallel_size > 1:
            if not _env_flag('VLLM_ENABLE_DISTRIBUTED_BACKGROUND_CAPTURE'):
                return False, "distributed background capture is not enabled"
            from vllm.v1.worker.distributed_capture import COLLECTIVE_OPS
            if not set(COLLECTIVE_OPS) <= set(runner.compilation_config.splitting_ops):
                return False, "collectives must be eager graph boundaries"
        if getattr(runner, "is_multimodal_model", False):
            return False, "multimodal input buffers are not isolated"
        if getattr(runner.model_config, "is_hybrid", False):
            return False, "hybrid attention/state-cache models are not isolated"
        if getattr(runner, "lora_config", None) is not None:
            return False, "LoRA runtime state is not isolated"
        if getattr(runner, "speculative_config", None) is not None:
            return False, "speculative decoding state is not isolated"
        required = {
            "synchronize_before_capture",
            "collect_before_capture",
            "empty_cache_before_capture",
            "manage_default_generator",
        }
        try:
            parameters = set(inspect.signature(torch.cuda.graph).parameters)
        except (TypeError, ValueError):
            parameters = set(
                inspect.signature(torch.cuda.graph.__init__).parameters
            )
        missing = sorted(required - parameters)
        if missing:
            return False, (
                "experimental PyTorch CUDA graph API is missing "
                + ", ".join(missing)
            )
        return True, "supported"

    def _ensure_thread(self) -> bool:
        with self._lock:
            if self._stopping:
                return False
            if self._thread is not None:
                return True
            self._thread = threading.Thread(
                target=self._worker_main,
                name="vllm-concurrent-piecewise-cudagraph",
                daemon=True,
            )
            self._thread.start()
            return True

    def enqueue_after_warmup(self, phase: str, shape: int) -> bool:
        """Claim a queued adaptive shape and record its warmup completion."""
        if not self.enabled or phase not in {"decode", "mixed", "prefill"}:
            return False
        if not self._ensure_thread():
            return False
        with self._lock:
            if self._stopping or shape in self._scheduled_shapes:
                return False
            claimed = self.manager.take_queued_shapes(
                phase=phase, limit=1, selected_shape=shape
            )
            if claimed != [shape]:
                return False
            self._scheduled_shapes.add(shape)
            try:
                warmup_ready = torch.cuda.Event()
                warmup_ready.record(
                    torch.cuda.current_stream(self.runner.device)
                )
                # Publish under the shutdown lock: no request may follow the
                # sentinel, otherwise a claimed shape would never complete.
                self._queue.put(
                    _CaptureRequest(
                        phase, shape, warmup_ready, time.perf_counter()
                    )
                )
            except Exception as error:
                self._scheduled_shapes.discard(shape)
                self.manager.mark_capture_failed(phase, shape, error)
                self.manager.settle_shared_capture(phase, shape, error)
                raise
        logger.debug(
            "Concurrent piecewise CUDA graph capture queued after real "
            "warmup: phase=%s shape=%d",
            phase,
            shape,
        )
        return True

    def _piecewise_backends(self) -> list[Any]:
        return self.manager.registry.backends()

    def _prepare_distributed_backends(self, backends):
        if self.runner.parallel_config.world_size == 1:
            return
        from vllm.v1.worker.distributed_capture import COLLECTIVE_OPS
        static = {t.untyped_storage().data_ptr() for t in
                  list(self.runner.model.parameters()) + list(self.runner.model.buffers())}
        for backend in backends:
            for node in backend.graph.graph.nodes:
                target = str(node.target)
                if any(op in target for op in COLLECTIVE_OPS) or any(
                        term in target for term in ('c10d', 'nccl', 'custom_all_reduce')):
                    raise RuntimeError(f'Communication inside capture piece: {target}')
            backend.capture_static_storages = static

    def enqueue_auxiliary(self, capture) -> bool:
        """Serialize private composite capture with ordinary shape capture."""
        if not self.enabled or self.runner.parallel_config.world_size > 1:
            return False
        if not self._ensure_thread():
            return False
        with self._lock:
            if self._stopping:
                return False
            self._queue.put(capture)
        return True

    def _initialize_worker_runtime(self) -> None:
        """Create thread-local CUDA library state before formal capture.

        cuBLAS handles are acquired from a thread-local pool. Real-request
        warmup initializes the inference thread, but it cannot initialize the
        background capture thread's handle. Creating that handle from inside
        capture is illegal, so issue one tiny GEMM on a private stream first.
        This is runtime setup, not a second model warmup.
        """
        if self._worker_runtime_initialized:
            return
        stream = torch.cuda.Stream(device=self.runner.device)
        with torch.cuda.stream(stream):
            left = torch.ones(
                (16, 16), device=self.runner.device, dtype=torch.float16
            )
            right = torch.ones_like(left)
            output = left @ right
        stream.synchronize()
        del left, right, output
        self._worker_runtime_initialized = True

    def _arm_entries(
        self,
        shape: int,
        graph_pool: Any,
        capture_stream: torch.cuda.Stream,
        capture_gate: threading.Event,
    ) -> list[Any]:
        backends = self._piecewise_backends()
        if not backends:
            raise RuntimeError(
                "no piecewise CUDA graph backends are registered"
            )
        armed: list[Any] = []
        try:
            self._prepare_distributed_backends(backends)
            for backend in backends:
                if not backend.prepare_concurrent_capture(
                    shape, graph_pool, capture_stream, capture_gate
                ):
                    raise RuntimeError(
                        "piece is not ready for concurrent capture: "
                        f"piece={backend.piecewise_compile_index} "
                        f"shape={shape}"
                    )
                armed.append(backend)
        except BaseException:
            for backend in armed:
                backend.abort_concurrent_capture(shape)
            raise
        return backends

    def _publish(
        self,
        phase: str,
        shape: int,
        backends: list[Any],
        capture_gate: threading.Event,
        elapsed: float,
        context_enter_seconds: float,
        body_seconds: float,
        context_exit_seconds: float,
    ) -> None:
        self.manager.registry.publish(shape, dynamic=True, gate=capture_gate)
        self.manager.mark_capture_ready(
            phase,
            shape,
            elapsed,
            context_enter_seconds=context_enter_seconds,
            body_seconds=body_seconds,
            context_exit_seconds=context_exit_seconds,
            captured_pieces=len(backends),
        )
        self.manager.settle_shared_capture(phase, shape)

    def completed_distributed_shapes(self):
        with self._lock:
            return sorted(self._unpublished)

    def publish_distributed(self, shape):
        # EngineCore calls this on every rank at an inference-step boundary.
        with self._lock:
            payload = self._unpublished.pop(shape, None)
        if payload is None:
            raise RuntimeError(f'Rank has no completed capture for shape {shape}')
        self._publish(*payload)
        logger.info('Distributed background graph published: shape=%d', shape)

    def _capture_one(self, request: _CaptureRequest) -> None:
        shape = request.shape
        started = time.perf_counter()
        self.manager.mark_capture_started(request.phase, shape)
        backends: list[Any] = []
        try:
            runtime_init_started = time.perf_counter()
            self._initialize_worker_runtime()
            runtime_init_finished = time.perf_counter()
            warmup_wait_started = time.perf_counter()
            request.warmup_ready.synchronize()
            warmup_wait_finished = time.perf_counter()
            graph_pool = torch.cuda.graph_pool_handle()
            capture_gate = threading.Event()
            context_started = time.perf_counter()
            context = graph_capture
            if self.runner.parallel_config.world_size > 1:
                from vllm.v1.worker.distributed_capture import capture_stream
                context = capture_stream
            with context(device=self.runner.device) as capture_context:
                body_started = time.perf_counter()
                backends = self._arm_entries(
                    shape, graph_pool, capture_context.stream, capture_gate
                )
                self.runner._concurrent_piecewise_capture_forward(shape)
                body_finished = time.perf_counter()
            finished = time.perf_counter()
            self._graph_pools[shape] = graph_pool
            publication = (
                request.phase,
                shape,
                backends,
                capture_gate,
                finished - started,
                body_started - context_started,
                body_finished - body_started,
                finished - body_finished,
            )
            if self.runner.parallel_config.world_size > 1:
                with self._lock:
                    self._unpublished[shape] = publication
            else:
                self._publish(*publication)
            logger.info(
                "Concurrent CUDA graph recorded: phase=%s shape=%d capture_seconds=%.6f",
                request.phase, shape, finished - started,
            )
            logger.debug(
                "[TIMER] Concurrent Piecewise CUDA Graph Capture: "
                "phase=%s shape=%d total=%.6fs queue_delay=%.6fs "
                "runtime_init=%.6fs warmup_wait=%.6fs "
                "context_enter=%.6fs body=%.6fs "
                "context_exit=%.6fs pieces=%d",
                request.phase,
                shape,
                finished - started,
                started - request.queued_at,
                runtime_init_finished - runtime_init_started,
                warmup_wait_finished - warmup_wait_started,
                body_started - context_started,
                body_finished - body_started,
                finished - body_finished,
                len(backends),
            )
        except BaseException as error:
            for backend in backends or self._piecewise_backends():
                backend.abort_concurrent_capture(shape)
            self._graph_pools.pop(shape, None)
            self.manager.mark_capture_failed(request.phase, shape, error)
            self.manager.settle_shared_capture(request.phase, shape, error)
            logger.exception(
                "Concurrent piecewise CUDA graph capture failed; shape %d "
                "will remain eager",
                shape,
            )

    def _worker_main(self) -> None:
        torch.cuda.set_device(self.runner.device)
        while True:
            request = self._queue.get()
            try:
                if request is None:
                    return
                if callable(request):
                    request()
                else:
                    self._capture_one(request)
            finally:
                if isinstance(request, _CaptureRequest):
                    with self._lock:
                        self._scheduled_shapes.discard(request.shape)
                self._queue.task_done()

    def shutdown(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
            thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join()
        for shape in self._unpublished:
            for backend in self._piecewise_backends():
                backend.abort_concurrent_capture(shape)
        self._unpublished.clear()

    def precapture(self, shapes):
        from vllm.compilation.counter import compilation_counter
        from vllm.distributed.parallel_state import is_global_first_rank
        from tqdm import tqdm

        r = self.runner
        if os.environ.get('VLLM_ENABLE_DISTRIBUTED_BACKGROUND_CAPTURE') == '1':
            self._prepare_distributed_backends(self._piecewise_backends())
        compilation_counter.num_gpu_runner_capture_triggers += 1
        started = time.perf_counter()
        free_before = torch.cuda.mem_get_info()[0]
        with (
            self.manager.gc_scope.fast_capture_scope(),
            graph_capture(device=r.device),
        ):
            cases = list(reversed(shapes))
            if is_global_first_rank():
                cases = tqdm(cases, desc="Capturing CUDA graph shapes")
            for shape in cases:
                for _ in range(r.compilation_config.cudagraph_num_of_warmups):
                    r._dummy_run(
                        shape,
                        capture_attn_cudagraph=r.full_cuda_graph,
                        skip_eplb=True,
                    )
                r._dummy_run(
                    shape,
                    capture_attn_cudagraph=r.full_cuda_graph,
                    skip_eplb=True,
                )
                self.manager.registry.publish(shape)
        logger.info(
            "Graph capturing finished in %.2f secs, took %.2f GiB",
            time.perf_counter() - started,
            (free_before - torch.cuda.mem_get_info()[0]) / (1 << 30),
        )

    def capture_pending(
        self,
        max_pieces_per_shape: Optional[int] = None,
        deadline_ms: Optional[float] = None,
        selected_shape: Optional[int] = None,
    ) -> int:
        """Execute an engine-admitted safe-point capture budget."""
        manager, runner = self.manager, self.runner
        if not manager.enabled:
            return 0
        if max_pieces_per_shape is not None and getattr(
            runner, "full_cuda_graph", False
        ):
            # Full graphs bind attention metadata into one capture unit and
            # cannot use the piecewise compiler's incremental budget.
            return 0
        shapes = manager.take_queued_shapes(
            limit=(
                1
                if selected_shape is not None
                else manager.max_captures_per_idle
            ),
            selected_shape=selected_shape,
        )
        if not shapes:
            return 0

        # Import lazily so controller state tests do not initialize CUDA.
        from vllm.distributed.parallel_state import graph_capture
        from vllm.forward_context import (
            CUDAGraphCaptureBudget,
            CUDAGraphCaptureStop,
        )

        logger.debug("Adaptive CUDA graph capture started: shapes=%s", shapes)
        progressed = 0
        with manager.gc_scope.fast_capture_scope():
            for shape in shapes:
                started = time.perf_counter()
                with manager._lock:
                    manager._records[
                        ("decode", shape)
                    ].capture_started_at = time.time()
                try:
                    capture_budget = None
                    if max_pieces_per_shape is not None:
                        deadline_ns = None
                        if deadline_ms is not None:
                            deadline_ns = time.perf_counter_ns() + int(
                                max(0.0, deadline_ms) * 1_000_000
                            )
                        capture_budget = CUDAGraphCaptureBudget(
                            max_pieces=max(1, int(max_pieces_per_shape)),
                            deadline_ns=deadline_ns,
                            stop_after_budget=_env_flag(
                                "VLLM_ADAPTIVE_CUDAGRAPH_EARLY_STOP", True
                            ),
                        )
                    with graph_capture(device=runner.device):
                        context_entered = time.perf_counter()
                        if capture_budget is None:
                            for _ in range(manager.extra_warmups):
                                runner._dummy_run(
                                    shape,
                                    capture_attn_cudagraph=(
                                        runner.full_cuda_graph
                                    ),
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
                                "graph capture: shape=%d pieces=%d",
                                shape,
                                len(capture_budget.captured_piece_ids),
                            )
                        body_finished = time.perf_counter()
                    finished = time.perf_counter()
                    elapsed = finished - started
                    with manager._lock:
                        if capture_budget is not None:
                            piece_total_ms = sum(
                                seconds * 1000.0
                                for seconds in capture_budget.piece_capture_seconds.values()
                            )
                            fixed_capture_ms = max(
                                0.0,
                                elapsed * 1000.0 - piece_total_ms,
                            )
                            manager._fixed_capture_history_ms.append(
                                fixed_capture_ms
                            )
                            del manager._fixed_capture_history_ms[:-32]
                            for (
                                piece_id,
                                seconds,
                            ) in capture_budget.piece_capture_seconds.items():
                                samples = manager._piece_capture_history_ms.setdefault(
                                    int(piece_id), []
                                )
                                samples.append(seconds * 1000.0)
                                del samples[:-16]
                        manager._capture_chunk_rows.append(
                            {
                                "phase": "decode",
                                "shape": shape,
                                "mode": (
                                    "full"
                                    if capture_budget is None
                                    else "incremental"
                                ),
                                "captured_piece_ids": (
                                    ""
                                    if capture_budget is None
                                    else ";".join(
                                        str(piece_id)
                                        for piece_id in sorted(
                                            capture_budget.captured_piece_ids
                                        )
                                    )
                                ),
                                "captured_piece_count": (
                                    0
                                    if capture_budget is None
                                    else len(capture_budget.captured_piece_ids)
                                ),
                                "deferred_piece_count": (
                                    0
                                    if capture_budget is None
                                    else len(capture_budget.deferred_piece_ids)
                                ),
                                "total_pieces": (
                                    0
                                    if capture_budget is None
                                    else capture_budget.total_piecewise_compiles
                                ),
                                "elapsed_ms": elapsed * 1000.0,
                                "context_enter_ms": (context_entered - started)
                                * 1000.0,
                                "body_ms": (body_finished - context_entered)
                                * 1000.0,
                                "context_exit_ms": (finished - body_finished)
                                * 1000.0,
                                "piece_capture_ms": (
                                    ""
                                    if capture_budget is None
                                    else ";".join(
                                        f"{piece_id}:{seconds * 1000.0:.6f}"
                                        for piece_id, seconds in sorted(
                                            capture_budget.piece_capture_seconds.items()
                                        )
                                    )
                                ),
                                "admitted_deadline_ms": (
                                    "" if deadline_ms is None else deadline_ms
                                ),
                            }
                        )
                    capture_complete = (
                        capture_budget is None or capture_budget.pass_complete
                    )
                    made_progress = (
                        capture_budget is None or capture_budget.made_progress
                    )
                    if capture_complete:
                        manager.registry.publish(shape, dynamic=True)
                        manager.mark_capture_ready(
                            "decode",
                            shape,
                            elapsed,
                            context_enter_seconds=context_entered - started,
                            body_seconds=body_finished - context_entered,
                            context_exit_seconds=finished - body_finished,
                            captured_pieces=(
                                0
                                if capture_budget is None
                                else len(capture_budget.captured_piece_ids)
                            ),
                        )
                        logger.info(
                            "Adaptive CUDA graph ready: shape=%d capture_seconds=%.6f",
                            shape, elapsed,
                        )
                        logger.debug(
                            "Adaptive CUDA graph became ready: shape=%d "
                            "capture_seconds=%.6f enter=%.6f body=%.6f "
                            "exit=%.6f",
                            shape,
                            elapsed,
                            context_entered - started,
                            body_finished - context_entered,
                            finished - body_finished,
                        )
                    else:
                        assert capture_budget is not None
                        manager.mark_capture_partial(
                            "decode",
                            shape,
                            elapsed,
                            captured_pieces=len(
                                capture_budget.captured_piece_ids
                            ),
                            remaining_pieces=len(
                                capture_budget.deferred_piece_ids
                            ),
                            context_enter_seconds=context_entered - started,
                            body_seconds=body_finished - context_entered,
                            context_exit_seconds=finished - body_finished,
                        )
                        logger.debug(
                            "Adaptive CUDA graph capture progressed: "
                            "shape=%d captured_pieces=%d remaining_pieces=%d "
                            "chunk_seconds=%.6f",
                            shape,
                            len(capture_budget.captured_piece_ids),
                            len(capture_budget.deferred_piece_ids),
                            elapsed,
                        )
                    if made_progress:
                        progressed += 1
                except BaseException as error:
                    manager.mark_capture_failed("decode", shape, error)
                    logger.exception(
                        "Adaptive CUDA graph capture failed: shape=%d", shape
                    )
        return progressed
