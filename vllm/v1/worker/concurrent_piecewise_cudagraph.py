# SPDX-License-Identifier: Apache-2.0

"""Experimental single-GPU concurrent formal CUDA graph capture.

The target shape is warmed by a real eager request. This worker then records
only the piecewise CUDA graphs on a private stream and pool while foreground
inference continues. The feature is deliberately fail-closed and leaves the
existing idle-capture path untouched unless explicitly enabled.
"""

from __future__ import annotations

import inspect
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

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


class ConcurrentPiecewiseCUDAGraphCaptureManager:
    """Own one serial background capture worker for a GPUModelRunner."""

    def __init__(self, runner: "GPUModelRunner") -> None:
        self.runner = runner
        self.enabled, self.disabled_reason = self._check_support(runner)
        self._queue: queue.Queue[Optional[_CaptureRequest]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopping = False
        self._scheduled_shapes: set[int] = set()
        self._graph_pools: dict[int, Any] = {}
        self._worker_runtime_initialized = False
        if self.enabled:
            logger.warning(
                "Experimental concurrent piecewise CUDA graph capture is "
                "enabled for single-GPU inference.")
        elif _env_flag(
                "VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE", True):
            logger.warning(
                "Concurrent piecewise CUDA graph capture requested but "
                "disabled: %s", self.disabled_reason)

    @staticmethod
    def _check_support(
            runner: "GPUModelRunner") -> tuple[bool, str]:
        if not _env_flag(
                "VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE", True):
            return False, "feature flag is off"
        if not getattr(runner, "enable_adaptive_cudagraph", False):
            return False, "adaptive CUDA graph controller is disabled"
        if not getattr(runner, "use_cuda_graph", False):
            return False, "CUDA graph execution is disabled"
        if getattr(runner, "full_cuda_graph", False):
            return False, "full CUDA graphs are not piecewise graphs"
        parallel = runner.parallel_config
        if (parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1):
            return False, "only TP=PP=DP=1 is supported"
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
            parameters = set(inspect.signature(
                torch.cuda.graph.__init__).parameters)
        missing = sorted(required - parameters)
        if missing:
            return False, ("experimental PyTorch CUDA graph API is missing "
                           + ", ".join(missing))
        return True, "supported"

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._worker_main,
                name="vllm-concurrent-piecewise-cudagraph",
                daemon=True,
            )
            self._thread.start()

    def enqueue_after_warmup(self, phase: str, shape: int) -> bool:
        """Claim a queued adaptive shape and record its warmup completion."""
        if not self.enabled or phase not in {"decode", "mixed", "prefill"}:
            return False
        with self._lock:
            if self._stopping or shape in self._scheduled_shapes:
                return False
            claimed = self.runner.adaptive_cudagraph_controller.take_queued_shapes(
                phase=phase, limit=1, selected_shape=shape)
            if claimed != [shape]:
                return False
            self._scheduled_shapes.add(shape)
        warmup_ready = torch.cuda.Event()
        warmup_ready.record(torch.cuda.current_stream(self.runner.device))
        self._ensure_thread()
        self._queue.put(_CaptureRequest(phase, shape, warmup_ready,
                                        time.perf_counter()))
        logger.info(
            "Concurrent piecewise CUDA graph capture queued after real "
            "warmup: phase=%s shape=%d", phase, shape)
        return True

    def _piecewise_backends(self) -> list[Any]:
        backends = getattr(
            self.runner.compilation_config,
            "cuda_piecewise_backends_list",
            {},
        )
        return sorted(backends.values(),
                      key=lambda item: item.piecewise_compile_index)

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
                (16, 16), device=self.runner.device, dtype=torch.float16)
            right = torch.ones_like(left)
            output = left @ right
        stream.synchronize()
        del left, right, output
        self._worker_runtime_initialized = True

    def _arm_entries(self, shape: int, graph_pool: Any,
                     capture_stream: torch.cuda.Stream,
                     capture_gate: threading.Event) -> list[Any]:
        backends = self._piecewise_backends()
        if not backends:
            raise RuntimeError("no piecewise CUDA graph backends are registered")
        armed: list[Any] = []
        try:
            for backend in backends:
                if not backend.prepare_concurrent_capture(
                        shape, graph_pool, capture_stream, capture_gate):
                    raise RuntimeError(
                        "piece is not ready for concurrent capture: "
                        f"piece={backend.piecewise_compile_index} "
                        f"shape={shape}")
                armed.append(backend)
        except BaseException:
            for backend in armed:
                backend.abort_concurrent_capture(shape)
            raise
        return backends

    def _publish(self, phase: str, shape: int, backends: list[Any],
                 capture_gate: threading.Event, elapsed: float,
                 context_enter_seconds: float, body_seconds: float,
                 context_exit_seconds: float) -> None:
        if not all(
                backend.concrete_size_entries.get(shape) is not None
                and backend.concrete_size_entries[shape].cudagraph is not None
                for backend in backends):
            raise RuntimeError(
                f"shape {shape} did not capture every piecewise graph")
        if shape not in self.runner.cuda_graph_captured:
            self.runner.cuda_graph_captured.append(shape)
            self.runner.cuda_graph_captured.sort()
        self.runner.dynamic_cuda_graph_captured.add(shape)
        self.runner.graphMatcher.update_captured_sizes(
            self.runner.cuda_graph_captured)
        # One shared transition prevents foreground inference from observing a
        # mixed chain in which only a prefix of piece graphs is published.
        capture_gate.set()
        for backend in backends:
            if not backend.finish_concurrent_capture(shape):
                raise RuntimeError(
                    f"failed to publish piece {backend.piecewise_compile_index}")
        self.runner.adaptive_cudagraph_controller.mark_capture_ready(
            phase,
            shape,
            elapsed,
            context_enter_seconds=context_enter_seconds,
            body_seconds=body_seconds,
            context_exit_seconds=context_exit_seconds,
            captured_pieces=len(backends),
        )
        self.runner.adaptive_cudagraph_controller.settle_shared_capture(
            phase, shape)

    def _capture_one(self, request: _CaptureRequest) -> None:
        shape = request.shape
        started = time.perf_counter()
        self.runner.adaptive_cudagraph_controller.mark_capture_started(
            request.phase, shape)
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
            with graph_capture(device=self.runner.device) as capture_context:
                body_started = time.perf_counter()
                backends = self._arm_entries(shape, graph_pool,
                                             capture_context.stream,
                                             capture_gate)
                self.runner._concurrent_piecewise_capture_forward(shape)
                body_finished = time.perf_counter()
            finished = time.perf_counter()
            self._graph_pools[shape] = graph_pool
            self._publish(
                request.phase,
                shape,
                backends,
                capture_gate,
                finished - started,
                body_started - context_started,
                body_finished - body_started,
                finished - body_finished,
            )
            logger.info(
                "[TIMER] Concurrent Piecewise CUDA Graph Capture: "
                "phase=%s shape=%d total=%.6fs queue_delay=%.6fs "
                "runtime_init=%.6fs warmup_wait=%.6fs "
                "context_enter=%.6fs body=%.6fs "
                "context_exit=%.6fs pieces=%d",
                request.phase, shape,
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
            self.runner.adaptive_cudagraph_controller.mark_capture_failed(
                request.phase, shape, error)
            self.runner.adaptive_cudagraph_controller.settle_shared_capture(
                request.phase, shape, error)
            logger.exception(
                "Concurrent piecewise CUDA graph capture failed; shape %d "
                "will remain eager", shape)

    def _worker_main(self) -> None:
        torch.cuda.set_device(self.runner.device)
        while True:
            request = self._queue.get()
            try:
                if request is None:
                    return
                self._capture_one(request)
            finally:
                if request is not None:
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
