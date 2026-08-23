#!/usr/bin/env python3
"""CPU-only checks for frontend tokenizer/EngineCore rebind overlap."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest import mock

import vllm.v1.engine.async_llm as async_module
import vllm.v1.engine.llm_engine as sync_module


def config(name: str):
    return SimpleNamespace(
        name=name,
        model_config=SimpleNamespace(model=name),
        scheduler_config=object(),
        lora_config=None,
        cache_config=object(),
    )


class SyncCore:
    def __init__(self):
        self.calls = []

    def rebind_model(self, value):
        self.calls.append(value.name)
        time.sleep(0.2)
        return {}


class AsyncCore:
    def __init__(self):
        self.calls = []
        self.resources = SimpleNamespace(engine_dead=False)

    def shutdown(self):
        return None

    async def rebind_model_async(self, value):
        self.calls.append(value.name)
        await asyncio.sleep(0.2)
        return {}


def test_sync_overlap() -> None:
    old, new = config("old"), config("new")
    engine = object.__new__(sync_module.LLMEngine)
    engine.vllm_config = old
    engine.engine_core = SyncCore()
    engine.has_unfinished_requests = lambda: False
    engine.mm_registry = object()
    engine.log_stats = False
    engine.reset_mm_cache = lambda: None

    def tokenizer(**_):
        time.sleep(0.2)
        return object()

    with mock.patch.object(sync_module, "init_tokenizer_from_configs",
                           tokenizer), \
            mock.patch.object(sync_module, "Processor", mock.Mock()), \
            mock.patch.object(sync_module, "OutputProcessor", mock.Mock()):
        started = time.perf_counter()
        result = engine.rebind_model(new)
        elapsed = time.perf_counter() - started
    assert elapsed < 0.35, elapsed
    assert result["frontend_rebind"]["overlapped"] is True
    assert engine.engine_core.calls == ["new"]


def test_sync_rollback() -> None:
    old, new = config("old"), config("new")
    engine = object.__new__(sync_module.LLMEngine)
    engine.vllm_config = old
    engine.engine_core = SyncCore()
    engine.has_unfinished_requests = lambda: False

    with mock.patch.object(
            sync_module, "init_tokenizer_from_configs",
            side_effect=RuntimeError("tokenizer failed")):
        try:
            engine.rebind_model(new)
        except RuntimeError as exc:
            assert str(exc) == "tokenizer failed"
        else:
            raise AssertionError("tokenizer failure was not propagated")
    assert engine.engine_core.calls == ["new", "old"]


async def test_async_overlap() -> None:
    old, new = config("old"), config("new")
    engine = object.__new__(async_module.AsyncLLM)
    engine.vllm_config = old
    engine.engine_core = AsyncCore()
    engine.output_handler = None
    engine.output_processor = SimpleNamespace(
        has_unfinished_requests=lambda: False)
    engine.mm_registry = object()
    engine.log_stats = False

    async def reset_mm_cache():
        return None

    engine.reset_mm_cache = reset_mm_cache

    def tokenizer(**_):
        time.sleep(0.2)
        return object()

    with mock.patch.object(async_module, "init_tokenizer_from_configs",
                           tokenizer), \
            mock.patch.object(async_module, "Processor", mock.Mock()), \
            mock.patch.object(async_module, "OutputProcessor", mock.Mock()):
        started = time.perf_counter()
        result = await engine.rebind_model(new)
        elapsed = time.perf_counter() - started
    assert elapsed < 0.35, elapsed
    assert result["frontend_rebind"]["overlapped"] is True
    assert engine.engine_core.calls == ["new"]


def main() -> None:
    test_sync_overlap()
    test_sync_rollback()
    asyncio.run(test_async_overlap())
    print("FRONTEND_REBIND_OVERLAP_OK")


if __name__ == "__main__":
    main()
