# SPDX-License-Identifier: Apache-2.0
"""Console verbosity must not change retained timing evidence."""
import ast
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1 import phase_profiler as module


@pytest.fixture
def timer(monkeypatch):
    instance = module.timer
    monkeypatch.setattr(instance, "start_times", {})
    monkeypatch.setattr(instance, "records", [])
    monkeypatch.setattr(module, "logger", logging.getLogger("timer-test"))
    ticks = iter([1.0, 1.25] * 10)
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    return instance


def test_single_timer_definition_and_instance():
    tree = ast.parse(Path(module.__file__).read_text())
    assert sum(isinstance(n, ast.ClassDef) and n.name == "SimpleTimer"
               for n in tree.body) == 1
    assert module.SimpleTimer() is module.timer


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_console_level_does_not_change_export(timer, tmp_path, caplog, level):
    caplog.set_level(level)
    for stage in ("Model Compilation", "split graph"):
        timer.start(stage)
        timer.end(stage)
    output = tmp_path / "timing.json"
    timer.save(str(output))
    records = json.loads(output.read_text())["records"]
    assert [(r["stage"], r["time_s"]) for r in records] == [
        ("Model Compilation", .25), ("split graph", .25)]
    assert all(r["timestamp"] for r in records)
    assert "Finished: Model Compilation (0.2500 s)" in caplog.text
    assert ("Finished: split graph" in caplog.text) == (level == logging.DEBUG)
    assert ("Started:" in caplog.text) == (level == logging.DEBUG)
    assert not timer.start_times


def test_ready_marker_is_not_a_compilation_summary(timer, caplog):
    caplog.set_level(logging.INFO)
    timer.start("Weight-Compile Overlap")
    timer.end("Weight-Compile Overlap")
    timer.start("Model Compilation")
    timer.end("Model Compilation", summary=False)
    assert "includes weight loading and compilation" in caplog.text
    assert "Finished: Model Compilation" not in caplog.text
    assert len(timer.records) == 2


def test_missing_start_still_warns(timer, caplog):
    timer.end("missing")
    assert caplog.records[-1].levelno == logging.WARNING
    assert not timer.records


@pytest.mark.parametrize("overlap", [True, False])
def test_adapter_compilation_summary_matches_actual_work(timer, caplog, overlap):
    from igniteserve_startup.runtime import StartupContext
    from vllm.v1.engine.vllm_startup_adapter import build_runtime_actions

    caplog.set_level(logging.INFO)
    engine = SimpleNamespace(model_executor=Mock())
    actions = build_runtime_actions(engine, overlap_weight_compile=overlap)
    context = StartupContext()
    actions["load_weights"](context)
    actions["compile_model"](context)
    assert ("Finished: Model Compilation" in caplog.text) is not overlap
    assert ("includes weight loading and compilation" in caplog.text) is overlap
    assert [r["stage"] for r in timer.records] == [
        "Weight-Compile Overlap" if overlap else "Loading Weights",
        "Model Compilation",
    ]


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_existing_environment_controls_timer(level):
    code = (
        "from vllm.v1.phase_profiler import timer; "
        "timer.start('Total Init'); timer.end('Total Init'); "
        "timer.start('split graph'); timer.end('split graph')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, "VLLM_LOGGING_LEVEL": level}, check=True,
    )
    text = result.stdout + result.stderr
    assert "Finished: Total Init" in text
    assert ("Finished: split graph" in text) == (level == "DEBUG")
    assert ("Started:" in text) == (level == "DEBUG")
