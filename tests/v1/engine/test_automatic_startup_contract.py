# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.engine.startup_analysis.discovery import StartupDiscoveryBuilder
from vllm.v1.engine.startup_analysis.enricher import StartupEvidenceEnricher
from vllm.v1.engine.startup_analysis.extractor import StartupActionExtractor
from vllm.v1.engine.startup_analysis.finalizer import (
    FinalizationBlocked,
    finalize_startup_contract,
)
from vllm.v1.engine.startup_analysis.generated import (
    GeneratedStartupRejected,
    prepare_generated_startup,
)
from vllm.v1.engine.startup_analysis.recorder import StartupAnalysisRecorder
from vllm.v1.engine.startup_analysis.source_instrumentation import (
    build_instrumented_bound_method,
)
from vllm.v1.engine.startup_analysis.workflow import (
    apply_edge_validation_report,
    verify_serial_equivalence,
)
from vllm.v1.engine.startup_analysis.validator import run_contract_validation


def _event(path: Path, *items: dict) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in items),
                    encoding="utf-8")


def _discovery(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "igniteserve-startup-discovery-v1",
        "call_boundaries": [],
        "sampled_call_boundaries": [],
    }), encoding="utf-8")


def _synthetic_source(path: Path, *, opaque: bool = False) -> None:
    middle = ("        if config.enabled:\n"
              "            self.prepare()\n" if opaque else
              "        self.prepare()\n")
    path.write_text(
        "class Engine:\n"
        "    def initialize(self, config):\n"
        f"{middle}"
        "        value = self.produce(config)\n"
        "        self.consume(value)\n",
        encoding="utf-8")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="model", architectures=["Synthetic"], dtype="float16",
            max_model_len=128),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=128, max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1,
            data_parallel_size=1),
    )


def _mark_validated(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["validation"] = {
        "serial_equivalence": {
            "status": "passed",
            "execution_kind": "generated_adapter",
        },
        "edge_perturbations": {"status": "complete"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_root_discovery_reconstructs_call_order_and_parents(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _event(
        trace,
        {"event": "call_span", "stage_at_call": "engine_startup.root",
         "call_id": 2, "parent_call_id": 1, "module": "vllm.child",
         "qualname": "child", "filename": "/repo/child.py", "lineno": 2,
         "started_ns": 30, "timestamp_ns": 50, "elapsed_ns": 20,
         "thread_id": 1, "thread_name": "MainThread"},
        {"event": "call_span", "stage_at_call": "engine_startup.root",
         "call_id": 1, "parent_call_id": None, "module": "vllm.root",
         "qualname": "root", "filename": "/repo/root.py", "lineno": 1,
         "started_ns": 10, "timestamp_ns": 60, "elapsed_ns": 50,
         "thread_id": 1, "thread_name": "MainThread"},
        {"event": "stage_end", "stage": "engine_startup.root",
         "name": "engine_startup.root", "elapsed_ns": 100},
    )
    payload = StartupDiscoveryBuilder([trace]).generate()
    assert payload["root_cost_s"]["observations"] == 1
    assert [item["qualname"] for item in payload["original_call_orders"][0]
            ] == ["root", "child"]
    assert payload["call_edges"][0]["caller"]["qualname"] == "root"
    assert payload["call_edges"][0]["callee"]["qualname"] == "child"


def test_extractor_builds_stateful_adapter_and_stage_map(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    contract = json.loads(outputs["contract"].read_text(encoding="utf-8"))
    nodes = contract["variants"][0]["nodes"]
    assert all(node["executable"] for node in nodes)
    assert len(nodes) == 3
    adapter = outputs["adapter"].read_text(encoding="utf-8")
    assert "context.require('input.config')" in adapter
    assert "context.require('value')" in adapter
    stage_map = json.loads(outputs["stage_map"].read_text(encoding="utf-8"))
    assert len(stage_map["regions"]) == 3
    review = json.loads(
        (outputs["review"] / "action_bindings.json").read_text(
            encoding="utf-8"))
    assert review["items"] == []


def test_line_stage_map_attributes_original_sequential_execution(
        tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "target.py"
    source.write_text(
        "class Target:\n"
        "    def startup(self):\n"
        "        self.first()\n"
        "        self.second()\n"
        "    def first(self):\n"
        "        return 1\n"
        "    def second(self):\n"
        "        return 2\n", encoding="utf-8")
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({
        "schema": "igniteserve-line-stage-map-v1",
        "source_path": str(source.resolve()),
        "function": "Target.startup",
        "regions": [
            {"stage": "first", "lineno": 3, "end_lineno": 3},
            {"stage": "second", "lineno": 4, "end_lineno": 4},
        ],
    }), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("target", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VLLM_STARTUP_ANALYSIS_AUDIT", "0")
    monkeypatch.setenv("VLLM_STARTUP_ANALYSIS_NVTX", "0")
    recorder = StartupAnalysisRecorder(tmp_path / "trace")
    recorder.start()
    target = module.Target()
    instrumented = build_instrumented_bound_method(
        target.startup, mapping, recorder)
    with recorder.stage("engine_startup.root"):
        instrumented()
    trace = recorder.flush()
    assert trace
    events = [json.loads(line) for line in trace.read_text(
        encoding="utf-8").splitlines()]
    child_stages = {item.get("name") for item in events
                    if item.get("event") == "stage_end"}
    assert {"first", "second"} <= child_stages


def test_enricher_builds_cache_variants_and_resources(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    first = json.loads(outputs["contract"].read_text(
        encoding="utf-8"))["variants"][0]["nodes"][0]["name"]
    trace = tmp_path / "D0.jsonl"
    _event(
        trace,
        {"event": "stage_end", "stage": first, "name": first,
         "elapsed_ns": 2_000_000_000},
        {"event": "audit", "stage": first, "audit_event": "open",
         "arguments": ["weights"]},
    )
    enriched = StartupEvidenceEnricher(
        outputs["contract"], {"D0": [trace]},
        matches_by_variant={"D0": {"cache_state": "D0"}},
    ).generate()
    variant = enriched["variants"][0]
    node = next(item for item in variant["nodes"] if item["name"] == first)
    assert node["estimated_cost_s"] == 2.0
    assert node["resources"] == {"disk_io": 1}
    assert variant["match"] == {"cache_state": "D0"}


class _SyntheticEngine:
    def __init__(self) -> None:
        self.events = []

    def prepare(self) -> None:
        self.events.append("prepare")

    def produce(self, config) -> str:
        self.events.append("produce")
        return config.value

    def consume(self, value: str) -> None:
        self.events.append(value)


def test_finalize_and_execute_generated_contract(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    _mark_validated(outputs["contract"])
    final = finalize_startup_contract(
        outputs["contract"], outputs["adapter"], outputs["review"],
        tmp_path / "final")
    engine = _SyntheticEngine()
    plan = prepare_generated_startup(
        engine, _config(), {"input.config": SimpleNamespace(value="done")},
        contract_path=final["contract"], source_path=source)
    plan.execute()
    assert engine.events == ["prepare", "produce", "done"]


def test_finalize_requires_generated_adapter_serial_evidence(
        tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    contract = json.loads(outputs["contract"].read_text(encoding="utf-8"))
    contract["validation"] = {
        "serial_equivalence": {
            "status": "passed",
            "execution_kind": "source_instrumentation",
        },
        "edge_perturbations": {"status": "complete"},
    }
    outputs["contract"].write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(FinalizationBlocked) as failure:
        finalize_startup_contract(
            outputs["contract"], outputs["adapter"], outputs["review"],
            tmp_path / "final")
    assert any(
        item["kind"] == "generated_adapter_serial_validation_missing"
        for item in failure.value.blockers)


def test_finalize_blocks_opaque_regions(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source, opaque=True)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    with pytest.raises(FinalizationBlocked) as failure:
        finalize_startup_contract(
            outputs["contract"], outputs["adapter"], outputs["review"],
            tmp_path / "final")
    assert any(item["kind"] == "pending_manual_review"
               for item in failure.value.blockers)
    assert not (tmp_path / "final" / "startup_contract.json").exists()


def test_generated_contract_rejects_variant_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    contract = json.loads(outputs["contract"].read_text(encoding="utf-8"))
    contract["variants"][0]["match"] = {"cache_state": "D0"}
    contract["validation"] = {
        "serial_equivalence": {
            "status": "passed",
            "execution_kind": "generated_adapter",
        },
        "edge_perturbations": {"status": "complete"},
    }
    outputs["contract"].write_text(json.dumps(contract), encoding="utf-8")
    final = finalize_startup_contract(
        outputs["contract"], outputs["adapter"], outputs["review"],
        tmp_path / "final")
    with pytest.raises(GeneratedStartupRejected, match="matching variant"):
        prepare_generated_startup(
            _SyntheticEngine(), _config(),
            {"input.config": SimpleNamespace(value="done")},
            contract_path=final["contract"], source_path=source)


def test_serial_equivalence_uses_semantic_artifact_check(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    discovery = tmp_path / "startup_discovery.json"
    _synthetic_source(source)
    _discovery(discovery)
    outputs = StartupActionExtractor(
        source, discovery, class_name="Engine", method_name="initialize",
    ).generate(tmp_path / "analysis")
    contract = json.loads(outputs["contract"].read_text(encoding="utf-8"))
    names = [node["name"] for node in contract["variants"][0]["nodes"]]
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    common = {
        "validation_output_fingerprint": "same-output",
        "memory_invariant": {"blocks": 4},
        "artifacts": {"file_count": 3},
    }
    baseline.write_text(json.dumps({
        **common, "artifacts": {**common["artifacts"], "digest": "path-a"},
    }), encoding="utf-8")
    candidate.write_text(json.dumps({
        **common, "artifacts": {**common["artifacts"], "digest": "path-b"},
    }), encoding="utf-8")
    trace = tmp_path / "candidate.jsonl"
    _event(trace, *({"event": "stage_end", "name": name}
                    for name in names))
    report = verify_serial_equivalence(
        outputs["contract"], {"D2": [baseline]}, {"D2": [candidate]},
        {"D2": [trace]}, tmp_path / "serial.json")
    assert report["passed"] is True
    assert report["execution_kind"] == "source_instrumentation"
    assert report["states"][0]["artifact_byte_hashes_match"] is False


def test_apply_edge_validation_removes_only_safe_edges(tmp_path: Path) -> None:
    contract = {
        "schema": "igniteserve-startup-contract-v2",
        "variants": [{
            "name": "D2",
            "nodes": [
                {"name": "a", "dependencies": []},
                {"name": "b", "dependencies": ["a"]},
            ],
            "edges": [{"source": "a", "target": "b",
                       "kind": "original-order", "confidence": 0.5}],
        }],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    validation = {
        "schema": "igniteserve-startup-contract-validation-v2",
        "summary": {"candidate_count": 1},
        "results": [{
            "classification": "safe_remove",
            "perturbation": {"variant": "D2", "source": "a", "target": "b"},
        }],
    }
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    result = apply_edge_validation_report(
        contract_path, validation_path, tmp_path / "output.json",
        review_file=tmp_path / "review.json")
    assert result["variants"][0]["nodes"][1]["dependencies"] == []
    assert result["validation"]["edge_perturbations"]["status"] == "complete"


def test_edge_validation_reads_generated_stage_report(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "perturbation": {
            "variant": "D2",
            "source": "a",
            "target": "b",
        },
    }), encoding="utf-8")
    script = (
        "import json, os; from pathlib import Path; "
        "Path(os.environ['VLLM_STARTUP_VALIDATION_RESULT']).write_text("
        "json.dumps({'fingerprint':'same'})); "
        "Path(os.environ['VLLM_STARTUP_GENERATED_REPORT']).write_text("
        "json.dumps({'stages':{"
        "'a':{'started_at':0.0,'finished_at':2.0},"
        "'b':{'started_at':1.0,'finished_at':3.0}}}))"
    )
    report = run_contract_validation(
        [sys.executable, "-c", script], [candidate], tmp_path / "runs",
        timeout_s=10.0, expected_fingerprint="same", repeats=1,
        baseline_elapsed_s=[10.0], maximum_regression_fraction=0.05)
    result = report["results"][0]
    assert result["classification"] == "safe_remove"
    assert result["attempts"][0]["edge_exercised"] is True
    assert result["attempts"][0]["startup_report"] is not None
