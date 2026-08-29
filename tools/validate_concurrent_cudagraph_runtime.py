#!/usr/bin/env python3
"""Validate the patched PyTorch primitives used by concurrent capture."""

from __future__ import annotations

import argparse
import threading
import time

import torch


def capture_kwargs(stream: torch.cuda.Stream) -> dict[str, object]:
    return {
        "stream": stream,
        "capture_error_mode": "thread_local",
        "synchronize_before_capture": False,
        "collect_before_capture": False,
        "empty_cache_before_capture": False,
        "manage_default_generator": False,
    }


def validate_no_global_sync(device: torch.device) -> dict[str, float]:
    foreground = torch.cuda.Stream(device=device)
    capture = torch.cuda.Stream(device=device)
    source = torch.ones(4096, device=device)

    # Populate allocator/runtime state before the long foreground kernel. A
    # first-use cudaMalloc here would otherwise make preparation look like a
    # synchronization performed by capture_begin().
    with torch.cuda.stream(capture):
        scratch = source + 1
    capture.synchronize()
    del scratch

    with torch.cuda.stream(foreground):
        torch.cuda._sleep(500_000_000)
        foreground_done = torch.cuda.Event()
        foreground_done.record()

    graph = torch.cuda.CUDAGraph()
    entered = time.perf_counter()
    context = torch.cuda.graph(graph, **capture_kwargs(capture))
    context.__enter__()
    after_enter = time.perf_counter()
    output = source + 1
    after_body = time.perf_counter()
    context.__exit__(None, None, None)
    after_exit = time.perf_counter()
    if foreground_done.query():
        raise AssertionError(
            "experimental graph context synchronized foreground work; "
            f"enter={after_enter - entered:.6f}s, "
            f"body={after_body - after_enter:.6f}s, "
            f"exit={after_exit - after_body:.6f}s")
    foreground.synchronize()
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output, torch.full_like(output, 2))
    return {
        "enter": after_enter - entered,
        "body": after_body - after_enter,
        "exit": after_exit - after_body,
    }


def validate_foreground_rng(device: torch.device) -> None:
    capture = torch.cuda.Stream(device=device)
    source = torch.ones(4096, device=device)
    torch.cuda.synchronize(device)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def capture_worker() -> None:
        try:
            torch.cuda.set_device(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, **capture_kwargs(capture)):
                source.add_(1)
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("foreground RNG test timed out")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=capture_worker)
    worker.start()
    if not entered.wait(timeout=10):
        raise TimeoutError("capture did not start")
    random_values = torch.rand(4096, device=device)
    release.set()
    worker.join(timeout=10)
    if worker.is_alive():
        raise TimeoutError("capture worker did not stop")
    if errors:
        raise errors[0]
    if not torch.isfinite(random_values).all():
        raise AssertionError("foreground RNG produced invalid values")


def validate_foreground_compute(device: torch.device) -> None:
    capture = torch.cuda.Stream(device=device)
    source = torch.ones(4096, device=device)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def capture_worker() -> None:
        try:
            torch.cuda.set_device(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, **capture_kwargs(capture)):
                source.mul_(2)
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("foreground compute test timed out")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=capture_worker)
    worker.start()
    if not entered.wait(timeout=10):
        raise TimeoutError("compute capture did not start")
    foreground = torch.cuda.Stream(device=device)
    with torch.cuda.stream(foreground):
        left = torch.randn((256, 256), device=device)
        right = torch.randn((256, 256), device=device)
        result = left @ right
    foreground.synchronize()
    if not torch.isfinite(result).all():
        raise AssertionError("foreground compute produced invalid values")
    release.set()
    worker.join(timeout=10)
    if worker.is_alive():
        raise TimeoutError("compute capture worker did not stop")
    if errors:
        raise errors[0]


def validate_default_contract(device: torch.device) -> None:
    foreground = torch.cuda.Stream(device=device)
    with torch.cuda.stream(foreground):
        torch.cuda._sleep(200_000_000)
        foreground_done = torch.cuda.Event()
        foreground_done.record()
    capture = torch.cuda.Stream(device=device)
    source = torch.ones(4096, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        output = source + 3
    if not foreground_done.query():
        raise AssertionError("default graph context no longer synchronizes")
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output, torch.full_like(output, 4))


def validate_random_capture_fails_closed(device: torch.device) -> None:
    capture = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, **capture_kwargs(capture)):
            torch.rand(4096, device=device)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "random op unexpectedly entered a capture without RNG ownership")
    random_values = torch.rand(4096, device=device)
    if not torch.isfinite(random_values).all():
        raise AssertionError("default generator was not restored after failure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    capture_context = validate_no_global_sync(device)
    validate_foreground_rng(device)
    validate_foreground_compute(device)
    validate_default_contract(device)
    validate_random_capture_fails_closed(device)
    print({
        "device": str(device),
        "capture_context_seconds": capture_context,
        "foreground_rng_during_capture": "passed",
        "foreground_compute_during_capture": "passed",
        "default_graph_contract": "passed",
        "random_capture_without_generator": "failed_closed",
    })


if __name__ == "__main__":
    main()
