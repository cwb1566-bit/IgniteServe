# SPDX-License-Identifier: Apache-2.0
"""Rank-local synchronization for weight-loading/compile overlap."""

import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_lock = threading.Lock()
_active = False
_weights_ready = threading.Event()
_weights_error: BaseException | None = None


def begin() -> None:
    global _active, _weights_error
    with _lock:
        _weights_ready.clear()
        _weights_error = None
        _active = True


def mark_weights_ready() -> None:
    _weights_ready.set()


def mark_weights_failed(error: BaseException) -> None:
    global _weights_error
    with _lock:
        _weights_error = error
    _weights_ready.set()


def wait_for_weights_ready(reason: str = "Online autotune") -> float:
    with _lock:
        active = _active
    if not active:
        return 0.0
    started = time.perf_counter()
    _weights_ready.wait()
    elapsed = time.perf_counter() - started
    with _lock:
        error = _weights_error
    logger.info("%s waited %.3f seconds for finalized weights", reason,
                elapsed)
    if error is not None:
        raise RuntimeError(
            f"{reason} cannot run because weight loading failed") from error
    return elapsed


def end() -> None:
    global _active, _weights_error
    with _lock:
        _active = False
        _weights_error = None
        _weights_ready.set()
