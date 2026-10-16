#!/usr/bin/env python3
"""CPU-only concurrency test for persistent-shell request routing."""

from __future__ import annotations

import asyncio
import os
from types import MethodType, SimpleNamespace

from multi_model_server import EngineState, MultiModelManager


async def main() -> None:
    os.environ["VLLM_ENABLE_PERSISTENT_ENGINE_SHELL"] = "1"
    args = SimpleNamespace(
        keepalive=60,
        max_concurrent_models=1,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    manager = MultiModelManager(args)
    shell = object()
    activations = []

    async def activate(self, model_name, hotspot_key):
        activations.append((model_name, hotspot_key))
        self.registry.setdefault(model_name, EngineState(model_name))
        await asyncio.sleep(0.01)
        return shell

    manager._activate_persistent_shell = MethodType(activate, manager)

    first = await manager._acquire_persistent_engine("model-a", "profile-a")
    second = await manager._acquire_persistent_engine("model-a", "profile-a")
    assert first is shell and second is shell
    assert manager.shell_active_requests == 2

    switch = asyncio.create_task(manager._acquire_persistent_engine(
        "model-b", "profile-b"))
    await asyncio.sleep(0.03)
    assert not switch.done(), "cross-model switch ran over active requests"
    await manager._release_persistent_engine()
    await asyncio.sleep(0.03)
    assert not switch.done(), "switch ran before every request drained"
    await manager._release_persistent_engine()
    assert await asyncio.wait_for(switch, timeout=1.0) is shell
    assert activations == [
        ("model-a", "profile-a"),
        ("model-b", "profile-b"),
    ]
    assert manager.shell_identity == ("model-b", "profile-b")
    await manager._release_persistent_engine()
    manager._idle_task.cancel()
    try:
        await manager._idle_task
    except asyncio.CancelledError:
        pass
    print("PERSISTENT_SHELL_ROUTING_OK")


if __name__ == "__main__":
    asyncio.run(main())
