# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from pathlib import Path

from vllm.v1.engine.startup_analysis.contract import apply_startup_contract
from vllm.v1.engine.startup_analysis.generator import StartupContractGenerator
from vllm.v1.engine.startup_analysis.nsight import (
    extract_nsight_stage_resources,
)
from vllm.v1.engine.startup_analysis.pyspy import parse_speedscope_profiles
from vllm.v1.engine.startup_analysis.recorder import StartupAnalysisRecorder
from vllm.v1.engine.startup_analysis.validator import (
    generate_edge_removal_candidates,
)
from vllm.v1.engine.startup_orchestrator import StageNode, StartupDAG


def test_generator_derives_state_dependency(tmp_path: Path,
                                            monkeypatch) -> None:
    monkeypatch.setenv("VLLM_STARTUP_ANALYSIS_PYTHON_CALLS", "0")
    monkeypatch.setenv("VLLM_STARTUP_ANALYSIS_AUDIT", "0")
    monkeypatch.setenv("VLLM_STARTUP_ANALYSIS_NVTX", "0")
    recorder = StartupAnalysisRecorder(tmp_path)
    recorder.start()
    with recorder.stage("producer"):
        recorder.state_access("put", "model_ready")
        recorder.resource("cpu", source="test")
    with recorder.stage("consumer"):
        recorder.state_access("require", "model_ready")
    trace = recorder.flush()
    assert trace is not None

    contract = StartupContractGenerator([trace]).generate()
    assert contract["summary"]["node_count"] == 2
    edge = next(item for item in contract["edges"]
                if item["target"] == "consumer")
    assert edge["source"] == "producer"
    assert edge["confidence"] == 0.98
    producer = next(item for item in contract["nodes"]
                    if item["name"] == "producer")
    assert producer["resources"] == {"cpu": 1}


def test_contract_overlay_preserves_actions(tmp_path: Path) -> None:
    action = lambda _: None
    dag = StartupDAG()
    dag.add_node(StageNode("a", action=action))
    dag.add_node(StageNode("b", dependencies={"a"}))
    contract = {
        "schema": "igniteserve-startup-contract-v1",
        "nodes": [
            {"name": "a", "dependencies": [], "resources": {"cpu": 1},
             "estimated_cost_s": 2.0, "confidence": 0.9},
            {"name": "b", "dependencies": [], "resources": {},
             "estimated_cost_s": 1.0, "confidence": 0.8},
        ],
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    result = apply_startup_contract(dag, path)
    assert result["applied"] == ["a", "b"]
    assert dag.nodes["a"].action is action
    assert dag.nodes["a"].resources == {"cpu": 1}
    assert dag.nodes["b"].dependencies == set()


def test_perturbation_only_removes_reviewable_edges(tmp_path: Path) -> None:
    contract = {
        "schema": "igniteserve-startup-contract-v1",
        "nodes": [
            {"name": "a", "dependencies": []},
            {"name": "b", "dependencies": ["a"]},
            {"name": "c", "dependencies": ["b"]},
        ],
        "edges": [
            {"source": "a", "target": "b", "confidence": 0.5},
            {"source": "b", "target": "c", "confidence": 0.98},
        ],
    }
    source = tmp_path / "contract.json"
    source.write_text(json.dumps(contract), encoding="utf-8")
    candidates = generate_edge_removal_candidates(source, tmp_path / "cases")
    assert len(candidates) == 1
    candidate = json.loads(candidates[0].read_text(encoding="utf-8"))
    b = next(node for node in candidate["nodes"] if node["name"] == "b")
    assert b["dependencies"] == []


def test_nsight_import_correlates_cuda_resources(tmp_path: Path) -> None:
    database = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER,
            nameId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, correlationId INTEGER,
            streamId INTEGER, contextId INTEGER, deviceId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, correlationId INTEGER,
            streamId INTEGER, contextId INTEGER, deviceId INTEGER,
            bytes INTEGER);
    """)
    connection.execute("INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel')")
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES (100, 500, ?, 7)",
        ("igniteserve.startup::compile_model",))
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (150, 170, 10, 1, 7)")
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (180, 190, 11, 1, 7)")
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (200, 300, 10, 3, 2, 0)")
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES "
        "(210, 240, 11, 4, 2, 0, 4096)")
    connection.commit()
    connection.close()

    result = extract_nsight_stage_resources(database)
    stage = result["stages"]["compile_model"]
    assert stage["kernel_count"] == 1
    assert stage["memcpy_bytes"] == 4096
    assert stage["resources"] == ["copy_engine", "cuda_runtime", "gpu_compute"]


def test_pyspy_profile_proposes_boundaries(tmp_path: Path) -> None:
    profile = {
        "shared": {"frames": [
            {"name": "startup", "file": "/repo/vllm/core.py", "line": 10},
            {"name": "load", "file": "/repo/vllm/loader.py", "line": 20},
        ]},
        "profiles": [{
            "type": "sampled",
            "name": "MainThread",
            "samples": [[0, 1], [0]],
            "weights": [0.02, 0.02],
        }],
    }
    path = tmp_path / "calls.speedscope.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    candidates = parse_speedscope_profiles([path])
    assert candidates[0]["qualname"] == "startup"
    assert candidates[0]["sampled_inclusive_s"] == 0.04
    load = next(item for item in candidates if item["qualname"] == "load")
    assert load["sampled_self_s"] == 0.02
