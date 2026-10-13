# SPDX-License-Identifier: Apache-2.0
import ast
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vllm.compilation.startup_resources import (
    STARTUP_OPTIMIZATION_DEFAULTS, apply_startup_optimization_defaults)


@pytest.fixture
def environment(monkeypatch):
    values = {}
    monkeypatch.setattr(os, "environ", values)
    return values


def test_master_resolves_all_nine_defaults(environment):
    environment["VLLM_ENABLE_STARTUP_OPTIMIZATIONS"] = "1"
    assert apply_startup_optimization_defaults() == STARTUP_OPTIMIZATION_DEFAULTS
    assert len(STARTUP_OPTIMIZATION_DEFAULTS) == 9
    assert apply_startup_optimization_defaults() == STARTUP_OPTIMIZATION_DEFAULTS


@pytest.mark.parametrize("master", [None, "0", "false"])
def test_disabled_master_does_not_change_legacy_defaults(environment, master):
    if master is not None:
        environment["VLLM_ENABLE_STARTUP_OPTIMIZATIONS"] = master
    before = environment.copy()
    assert apply_startup_optimization_defaults() == {}
    assert environment == before


@pytest.mark.parametrize("name", list(STARTUP_OPTIMIZATION_DEFAULTS))
def test_explicit_child_off_wins(environment, name):
    environment.update(VLLM_ENABLE_STARTUP_OPTIMIZATIONS="1")
    environment[name] = "off" if name == "VLLM_STARTUP_DAG_MODE" else "0"
    result = apply_startup_optimization_defaults()
    assert result[name] == environment[name]
    assert all(result[k] == v for k, v in STARTUP_OPTIMIZATION_DEFAULTS.items()
               if k != name)


@pytest.mark.parametrize("mode", ["off", "shadow", "discover", "execute-generated"])
def test_explicit_dag_modes_survive_master(environment, mode):
    environment.update(VLLM_ENABLE_STARTUP_OPTIMIZATIONS="1",
                       VLLM_STARTUP_DAG_MODE=mode)
    assert apply_startup_optimization_defaults()["VLLM_STARTUP_DAG_MODE"] == mode


def test_config_initializes_defaults_before_validation(environment):
    environment["VLLM_ENABLE_STARTUP_OPTIMIZATIONS"] = "1"
    source = Path(__file__).parents[2] / "vllm/config.py"
    module = ast.parse(source.read_text())
    config = next(n for n in module.body if isinstance(n, ast.ClassDef)
                  and n.name == "VllmConfig")
    method = next(n for n in config.body if isinstance(n, ast.FunctionDef)
                  and n.name == "__post_init__")
    # Stop at validation: this checks the real setup ordering without GPUs or HF.
    statements = []
    for statement in method.body:
        statements.append(statement)
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "try_verify_and_update_config"):
            break
    method.body = statements
    code = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"logger": MagicMock()}
    exec(compile(code, str(source), "exec"), namespace)

    class Config:
        def try_verify_and_update_config(self):
            assert all(environment[k] == v for k, v in STARTUP_OPTIMIZATION_DEFAULTS.items())
    namespace["__post_init__"](Config())
