# SPDX-License-Identifier: Apache-2.0
"""Rank-local synchronization for weight-loading/compile overlap."""

import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_lock = threading.Lock()
_active = False
_weights_ready = threading.Event()


def begin() -> None:
    global _active
    with _lock:
        _weights_ready.clear()
        _active = True


def mark_weights_ready() -> None:
    _weights_ready.set()


def wait_for_weights_ready() -> float:
    with _lock:
        active = _active
    if not active:
        return 0.0
    started = time.perf_counter()
    _weights_ready.wait()
    elapsed = time.perf_counter() - started
    logger.info("Online autotune waited %.3f seconds for finalized weights",
                elapsed)
    return elapsed


def end() -> None:
    global _active
    with _lock:
        _active = False
        _weights_ready.set()
