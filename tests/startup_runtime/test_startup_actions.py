# SPDX-License-Identifier: Apache-2.0
"""Shared cold-start/rebind actions retain their lifecycle boundaries."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from igniteserve_startup.runtime import StartupContext
from vllm.v1.engine.vllm_startup_adapter import (
    build_action_registry, build_runtime_actions,
)


@pytest.mark.parametrize("overlap", [False, True])
def test_shared_actions_join_weights_and_compile_once(overlap):
    engine = NS(model_executor=Mock())
    context = StartupContext()
    probe = Mock()
    actions = build_runtime_actions(
        engine, overlap_weight_compile=overlap, after_load=probe)
    actions["load_weights"](context)
    actions["compile_model"](context)
    probe.assert_called_once_with(context)
    if overlap:
        engine.model_executor.load_weights_and_trigger_compile.assert_called_once()
        engine.model_executor.load_weights.assert_not_called()
        engine.model_executor.run_to_trigger_compile.assert_not_called()
        assert context.require("weight_compile_worker_results") is (
            engine.model_executor.load_weights_and_trigger_compile.return_value)
    else:
        engine.model_executor.load_weights.assert_called_once()
        engine.model_executor.run_to_trigger_compile.assert_called_once()
        engine.model_executor.load_weights_and_trigger_compile.assert_not_called()


def test_rebind_actions_do_not_probe_or_release_cold_start_assets():
    engine = NS(model_executor=Mock(), collective_rpc=Mock())
    actions = build_runtime_actions(engine, overlap_weight_compile=False)
    context = StartupContext()
    actions["load_weights"](context)
    actions["ready"](context)
    engine.collective_rpc.assert_not_called()


def test_declarative_actions_publish_routes_after_fused_action(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP", "1")
    events = []
    engine = NS(
        model_executor=Mock(),
        collective_rpc=Mock(side_effect=lambda method: (
            events.append(method) or [{"kv_profile_hit": True}]))
    )
    engine.model_executor.load_weights_and_trigger_compile.side_effect = (
        lambda: events.append("fused_complete") or [])
    actions = build_action_registry((engine, None, None, False, None))
    context = StartupContext()
    actions["load_weights"](context)
    actions["compile_already_ready"](context)
    assert events == ["fused_complete", "get_startup_state_snapshot"]
    assert context.require("weights.compile_completed") is True
    assert context.require("assets.kv_profile_usable") is True
    assert context.require("ready.compile_model") is True
    engine.model_executor.run_to_trigger_compile.assert_not_called()
    actions["ready"](context)
    assert events[-1] == "release_startup_assets"


def test_failed_fused_action_does_not_publish_readiness(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP", "1")
    engine = NS(model_executor=Mock(), collective_rpc=Mock())
    engine.model_executor.load_weights_and_trigger_compile.side_effect = RuntimeError("failed")
    actions = build_action_registry((engine, None, None, False, None))
    context = StartupContext()
    with pytest.raises(RuntimeError, match="failed"):
        actions["load_weights"](context)
    assert context.get("ready.load_weights") is None
    assert context.get("weights.compile_completed") is None
    engine.collective_rpc.assert_not_called()
