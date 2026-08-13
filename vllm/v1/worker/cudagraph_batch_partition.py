# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PhaseBatchPartition:
    phase: str
    request_indices: tuple[int, ...]
    token_indices: tuple[int, ...]
    logits_indices: tuple[int, ...]
    query_lens: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return len(self.token_indices)


@dataclass(frozen=True)
class DecodeExecutionPartition:
    request_indices: tuple[int, ...]
    graph_size: int

    @property
    def request_size(self) -> int:
        return len(self.request_indices)

    @property
    def padding_size(self) -> int:
        return self.graph_size - self.request_size


def partition_mixed_batch(
    query_lens: Sequence[int],
    is_prefill: Sequence[bool],
) -> tuple[PhaseBatchPartition, ...]:
    """Build lossless request/token maps for prefill/decode separation.

    This helper is deliberately tensor-free. Runtime code can use these maps
    to gather lane-specific input and attention metadata and scatter per-request
    logits back to the original scheduler order before one unified sampling
    operation.
    """
    if len(query_lens) != len(is_prefill):
        raise ValueError("query_lens and is_prefill must have equal lengths")
    if any(int(length) <= 0 for length in query_lens):
        raise ValueError("every scheduled request must contain at least one token")

    token_starts: list[int] = []
    cursor = 0
    for query_len in query_lens:
        token_starts.append(cursor)
        cursor += int(query_len)

    partitions: list[PhaseBatchPartition] = []
    for phase, selected in (("prefill", True), ("decode", False)):
        request_indices = tuple(
            index for index, flag in enumerate(is_prefill)
            if bool(flag) is selected)
        if not request_indices:
            continue
        token_indices: list[int] = []
        logits_indices: list[int] = []
        part_query_lens: list[int] = []
        for request_index in request_indices:
            query_len = int(query_lens[request_index])
            token_start = token_starts[request_index]
            token_indices.extend(range(token_start, token_start + query_len))
            logits_indices.append(token_start + query_len - 1)
            part_query_lens.append(query_len)
        partitions.append(
            PhaseBatchPartition(
                phase=phase,
                request_indices=request_indices,
                token_indices=tuple(token_indices),
                logits_indices=tuple(logits_indices),
                query_lens=tuple(part_query_lens),
            ))

    all_requests = sorted(index for part in partitions
                          for index in part.request_indices)
    all_tokens = sorted(index for part in partitions
                        for index in part.token_indices)
    if all_requests != list(range(len(query_lens))):
        raise AssertionError("phase partition lost or duplicated requests")
    if all_tokens != list(range(cursor)):
        raise AssertionError("phase partition lost or duplicated tokens")
    return tuple(partitions)


def partition_pure_decode(
    num_requests: int,
    request_sizes: Iterable[int],
    graph_sizes: Iterable[int],
) -> tuple[DecodeExecutionPartition, ...]:
    """Map a pure-decode batch to graph lanes without duplicating requests."""
    request_sizes = tuple(int(size) for size in request_sizes)
    graph_sizes = tuple(int(size) for size in graph_sizes)
    if len(request_sizes) != len(graph_sizes) or not request_sizes:
        raise ValueError("request_sizes and graph_sizes must be non-empty peers")
    if sum(request_sizes) != int(num_requests):
        raise ValueError("request partition does not cover the decode batch")

    partitions: list[DecodeExecutionPartition] = []
    cursor = 0
    for request_size, graph_size in zip(request_sizes, graph_sizes):
        if request_size <= 0 or graph_size < request_size:
            raise ValueError("each graph must cover its non-empty request part")
        request_indices = tuple(range(cursor, cursor + request_size))
        partitions.append(DecodeExecutionPartition(request_indices, graph_size))
        cursor += request_size
    return tuple(partitions)
