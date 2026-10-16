from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from igniteserve_startup.contract import ContractRejected
from igniteserve_startup.runtime import StartupContext
from vllm.v1.engine import declarative_startup as entry


def config(device="cuda"):
    return SimpleNamespace(device_config=SimpleNamespace(device=device),
                           parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1))


def test_reject_before_actions(monkeypatch):
    prepare = MagicMock(side_effect=ContractRejected("bad source"))
    monkeypatch.setattr(entry, "prepare", prepare)
    engine = SimpleNamespace(model_executor=MagicMock())
    assert entry.run(engine, config(), object, False, None) is False
    engine.model_executor.shutdown.assert_not_called()


def test_error_after_actions_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(tmp_path))
    node = SimpleNamespace(name="test", estimated_cost_s=1)
    dag = SimpleNamespace(nodes={"test": node})
    monkeypatch.setattr(entry, "prepare", lambda *a, **k: (
        dag, StartupContext(), {}, {"schema": "test"}))
    executor = MagicMock()
    executor.run.side_effect = RuntimeError("action started")
    monkeypatch.setattr(entry, "ResourceAwareDAGExecutor", lambda **k: executor)
    engine = SimpleNamespace(model_executor=MagicMock())
    with pytest.raises(RuntimeError, match="action started"):
        entry.run(engine, config(), object, False, None)
    executor.run.assert_called_once()
    engine.model_executor.shutdown.assert_called_once()


def test_unqualified_device_uses_existing_path(monkeypatch):
    prepare = MagicMock(side_effect=AssertionError("must not bind"))
    monkeypatch.setattr(entry, "prepare", prepare)
    assert entry.run(object(), config("cpu"), object, False, None) is False
    prepare.assert_not_called()
