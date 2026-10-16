# SPDX-License-Identifier: Apache-2.0
"""Communication-free background recording for TP/PP piecewise graphs.

Collectives are eager graph boundaries. Only the recording thread substitutes
shape-only results; foreground inference always executes real communication.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import torch

COLLECTIVE_OPS = ('vllm.all_reduce', 'vllm.all_gather', 'vllm.reduce_scatter')
_recording = ContextVar('distributed_piecewise_recording', default=False)


def recording():
    return _recording.get()


@contextmanager
def capture_stream(device):
    # Do not enter GroupCoordinator.graph_capture: its custom-all-reduce state
    # belongs to the live communicator and is not thread-local.
    stream = torch.cuda.Stream(device=device)
    token = _recording.set(True)
    try:
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)
    finally:
        _recording.reset(token)


def private_inputs(args, static_storages):
    """Allocate layout-equivalent activation buffers; retain immutable weights.

    Aliased views and nonzero offsets require storage-aware copying, so reject
    them rather than silently changing an in-place/alias contract.
    """
    seen = set()

    def allocate(value):
        if isinstance(value, torch.Tensor):
            key = value.untyped_storage().data_ptr()
            if key in static_storages:
                return value
            if key in seen or value.storage_offset() or not value.is_contiguous():
                raise ValueError('Distributed capture requires unaliased contiguous activations')
            seen.add(key)
            return torch.empty_like(value)
        if isinstance(value, tuple):
            return tuple(allocate(v) for v in value)
        if isinstance(value, list):
            return [allocate(v) for v in value]
        if isinstance(value, (int, float, bool, str, type(None))):
            return value
        raise TypeError(f'Unsupported distributed capture input: {type(value)}')

    return tuple(allocate(v) for v in args)


def copy_inputs(destination, source):
    """Called on the inference stream after the input signature was checked."""
    for dst, src in zip(destination, source):
        if isinstance(dst, torch.Tensor):
            if dst.data_ptr() != src.data_ptr():
                dst.copy_(src, non_blocking=True)
        elif isinstance(dst, (tuple, list)):
            copy_inputs(dst, src)
