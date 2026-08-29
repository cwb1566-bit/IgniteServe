# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


class CUDAGraphCaptureStop(RuntimeError):
    """Internal control flow after a bounded capture unit is complete."""


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    cu_tokens_across_dp_cpu: torch.Tensor

    @staticmethod
    def num_tokens_across_dp(num_tokens: int, dp_size: int,
                             dp_rank: int) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
            parallel_config: ParallelConfig,
            attn_metadata: Any,
            num_tokens: int,
            num_tokens_across_dp: Optional[torch.Tensor] = None
    ) -> "DPMetadata":

        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        if attn_metadata is not None and hasattr(attn_metadata,
                                                 "num_prefill_tokens"):
            # for v0 attention backends
            batchsize = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens
        else:
            # for v1 attention backends or no attn_metadata
            batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert (num_tokens_across_dp is None
                or num_tokens_across_dp[dp_rank] == batchsize)
        if num_tokens_across_dp is None:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                batchsize, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        return DPMetadata(max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu)


@dataclass
class CUDAGraphCaptureBudget:
    """Per-forward authorization for incremental piecewise graph capture."""

    max_pieces: int
    deadline_ns: Optional[int] = None
    claimed_piece_ids: set[int] = field(default_factory=set)
    captured_piece_ids: set[int] = field(default_factory=set)
    deferred_piece_ids: set[int] = field(default_factory=set)
    observed_piece_ids: set[int] = field(default_factory=set)
    piece_capture_seconds: dict[int, float] = field(default_factory=dict)
    total_piecewise_compiles: int = 0
    stop_after_budget: bool = False

    def try_claim(self, piece_id: int, total_pieces: int) -> bool:
        self.total_piecewise_compiles = max(self.total_piecewise_compiles,
                                            int(total_pieces))
        self.observed_piece_ids.add(int(piece_id))
        if len(self.claimed_piece_ids) >= max(0, self.max_pieces):
            self.deferred_piece_ids.add(int(piece_id))
            return False
        if (self.deadline_ns is not None
                and time.perf_counter_ns() >= self.deadline_ns):
            self.deferred_piece_ids.add(int(piece_id))
            return False
        self.claimed_piece_ids.add(int(piece_id))
        return True

    def defer(self, piece_id: int, total_pieces: int) -> None:
        self.total_piecewise_compiles = max(self.total_piecewise_compiles,
                                            int(total_pieces))
        self.observed_piece_ids.add(int(piece_id))
        self.deferred_piece_ids.add(int(piece_id))

    def should_stop_before(self, piece_id: int, total_pieces: int) -> bool:
        if not self.stop_after_budget:
            return False
        piece_id = int(piece_id)
        total_pieces = int(total_pieces)
        self.total_piecewise_compiles = max(
            self.total_piecewise_compiles, total_pieces)
        self.deferred_piece_ids.update(range(piece_id, total_pieces))
        return True

    def mark_captured(self, piece_id: int,
                      capture_seconds: float = 0.0) -> None:
        piece_id = int(piece_id)
        self.captured_piece_ids.add(piece_id)
        self.deferred_piece_ids.discard(piece_id)
        self.piece_capture_seconds[piece_id] = float(capture_seconds)

    def should_stop_after(self, piece_id: int, total_pieces: int) -> bool:
        if not self.stop_after_budget:
            return False
        if len(self.captured_piece_ids) < max(0, self.max_pieces):
            return False
        piece_id = int(piece_id)
        total_pieces = int(total_pieces)
        if piece_id >= total_pieces - 1:
            return False
        self.total_piecewise_compiles = max(
            self.total_piecewise_compiles, total_pieces)
        self.deferred_piece_ids.update(range(piece_id + 1, total_pieces))
        return True

    @property
    def made_progress(self) -> bool:
        return bool(self.captured_piece_ids)

    @property
    def pass_complete(self) -> bool:
        return not self.deferred_piece_ids


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    """
    Type AttentionMetadata for v0,
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    set dynamically for each forward pass
    """
    attn_metadata: Union["AttentionMetadata", dict[str, "AttentionMetadata"]]
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None
    skip_cuda_graphs: bool = False
    # A real request is intentionally executing the target shape eagerly so
    # its kernels and workspaces can serve as CUDA graph warmup.
    cudagraph_runtime_warmup: bool = False
    # True only in the experimental background thread that performs formal
    # piecewise CUDA graph capture after a real request has warmed the shape.
    concurrent_cudagraph_capture: bool = False
    # Optional budget used only by adaptive background capture. Startup
    # capture leaves this unset and retains the original all-pieces behavior.
    cudagraph_capture_budget: Optional[CUDAGraphCaptureBudget] = None


_forward_context: ContextVar[Optional[ForwardContext]] = ContextVar(
    "vllm_forward_context", default=None)


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    forward_context = _forward_context.get()
    assert forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return forward_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: Optional[int] = None,
    num_tokens_across_dp: Optional[torch.Tensor] = None,
    skip_cuda_graphs: bool = False,
    cudagraph_runtime_warmup: bool = False,
    cudagraph_capture_budget: Optional[CUDAGraphCaptureBudget] = None,
    concurrent_cudagraph_capture: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1 and (
            attn_metadata is not None or num_tokens is not None):
        dp_metadata = DPMetadata.make(vllm_config.parallel_config,
                                      attn_metadata, num_tokens or 0,
                                      num_tokens_across_dp)

    context_token = _forward_context.set(ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        skip_cuda_graphs=skip_cuda_graphs,
        cudagraph_runtime_warmup=cudagraph_runtime_warmup,
        cudagraph_capture_budget=cudagraph_capture_budget,
        concurrent_cudagraph_capture=concurrent_cudagraph_capture,
    ))

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform
            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)

        _forward_context.reset(context_token)
