import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path

from igniteserve_startup.contract import ContractRejected, digest, identity, prepare
from igniteserve_startup.offline import analyze, finalize
from igniteserve_startup.runtime import (
    ResourceAwareDAGExecutor, StageNode, StartupContext, StartupDAG,
    StartupExecutionError,
)


class SchedulerTests(unittest.TestCase):
    def test_runtime_has_no_engine_or_torch_imports(self):
        import igniteserve_startup
        root = str(Path(igniteserve_startup.__file__).parent.parent)
        code = (f"import sys; sys.path.insert(0, {root!r}); "
                "import igniteserve_startup.runtime, igniteserve_startup.contract; "
                "assert 'torch' not in sys.modules and 'vllm' not in sys.modules")
        subprocess.run([sys.executable, "-I", "-c", code], check=True)

    def test_state_probe_updates_pending_priority(self):
        dag, sequence = StartupDAG(), []
        def probe(ctx):
            sequence.append("probe")
            ctx.estimate("b", 9)
        dag.add_node(StageNode("probe", probe))
        dag.add_node(StageNode("a", lambda _: sequence.append("a"), {"probe"}, 2))
        dag.add_node(StageNode("b", lambda _: sequence.append("b"), {"probe"}, 1))
        ResourceAwareDAGExecutor(1).run(dag, StartupContext())
        self.assertEqual(sequence, ["probe", "b", "a"])

    def test_action_failure_is_not_retried(self):
        dag, calls = StartupDAG(), []
        def action(ctx):
            calls.append("action")
            raise RuntimeError("action failure")
        dag.add_node(StageNode("action", action))
        with self.assertRaises(StartupExecutionError):
            ResourceAwareDAGExecutor().run(dag, StartupContext())
        self.assertEqual(calls, ["action"])

    def test_state_arrival_releases_consumer_before_producer_finishes(self):
        context, dag = StartupContext(), StartupDAG()
        consumed = threading.Event()
        def producer(ctx):
            ctx.put("structure", "valid")
            if not consumed.wait(2):
                raise AssertionError("consumer was gated on whole producer")
        def consumer(ctx):
            self.assertEqual(ctx.require("structure"), "valid")
            consumed.set()
        dag.add_node(StageNode("producer", producer, produced_state={"structure"}))
        dag.add_node(StageNode("consumer", consumer, required_state={"structure"}))
        result = ResourceAwareDAGExecutor(2).run(dag, context)
        self.assertEqual(len(result), 2)

    def test_future_keeps_resources_until_ready(self):
        dag, context, future = StartupDAG(), StartupContext(), Future()
        timer = threading.Timer(0.05, lambda: future.set_result(42))
        def launch(ctx):
            timer.start()
            return future
        def consume(ctx):
            self.assertEqual(ctx.require("stage_result.prepare"), 42)
        dag.add_node(StageNode("prepare", launch, resources={"gpu": 1}))
        dag.add_node(StageNode("consume", consume, {"prepare"}, resources={"gpu": 1}))
        result = ResourceAwareDAGExecutor(2, {"gpu": 1}).run(dag, context)
        timer.join()
        self.assertGreaterEqual(result["consume"].started_at, result["prepare"].finished_at)

    def test_real_main_thread_and_named_affinity(self):
        dag, ids = StartupDAG(), []
        for name in ("a", "b"):
            dag.add_node(StageNode(name, lambda _: ids.append(threading.get_ident()),
                                   details={"thread_affinity": "cuda"}))
        dag.add_node(StageNode("main", lambda _: self.assertIs(
            threading.current_thread(), threading.main_thread()),
            details={"thread_affinity": "main_thread"}))
        ResourceAwareDAGExecutor(3).run(dag, StartupContext())
        self.assertEqual(len(set(ids)), 1)

    def test_unpublished_output_fails(self):
        dag = StartupDAG()
        dag.add_node(StageNode("bad", lambda _: None, produced_state={"missing"}))
        with self.assertRaises(StartupExecutionError):
            ResourceAwareDAGExecutor().run(dag, StartupContext())

    def test_state_cycle_rejected(self):
        dag = StartupDAG()
        dag.add_node(StageNode("a", required_state={"b"}, produced_state={"a"}))
        dag.add_node(StageNode("b", required_state={"a"}, produced_state={"b"}))
        with self.assertRaises(ValueError):
            dag.topological_order()


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.adapter = self.root / "adapter.py"
        self.adapter.write_text(
            "def build_action_registry(host):\n"
            "    def build(ctx): ctx.put('answer', host + 2)\n"
            "    return {'build': build}\n")
        self.contract = self.root / "contract.json"
        self.payload = dict(
            schema="igniteserve-startup-v3", status="candidate",
            adapter_sha256=digest(self.adapter), sources={"adapter.py": digest(self.adapter)},
            compatibility={"engine": "synthetic"}, initial_state=[],
            resource_capacities={"cpu": 1}, variants=[dict(name="generic", match={}, nodes=[
                dict(name="build", action_id="build", dependencies=[],
                     required_state=[], produced_state=["answer"],
                     resources={"cpu": 1}, estimated_cost_s=0.01)])])

    def load(self, **kwargs):
        self.contract.write_text(json.dumps(self.payload))
        return prepare(self.contract, self.adapter, 40, {},
                       {"engine": "synthetic"}, self.root, **kwargs)

    def test_complete_portable_offline_online(self):
        dag, context, capacities, _ = self.load(allow_candidate=True)
        ResourceAwareDAGExecutor(1, capacities).run(dag, context)
        self.assertEqual(context.require("answer"), 42)
        review, validation = self.root / "review.json", self.root / "validation.json"
        review.write_text(json.dumps(dict(identity=identity(self.payload),
            actions={"build": "approved"}, constraints="approved",
            rationale="One pure state producer; no hidden effects", reviewer="test")))
        validation.write_text(json.dumps(dict(identity=identity(self.payload),
            serial_equivalence=True, scheduled_equivalence=True, variants=["generic"])))
        final = self.root / "final.json"
        finalize(self.contract, self.adapter, review, validation, self.root, final)
        dag, context, capacities, _ = prepare(final, self.adapter, 40, {},
                                              {"engine": "synthetic"}, self.root)
        ResourceAwareDAGExecutor(2, capacities).run(dag, context)
        self.assertEqual(context.require("answer"), 42)

    def test_candidate_is_not_final(self):
        with self.assertRaises(ContractRejected):
            self.load()

    def test_missing_initial_value(self):
        self.payload["initial_state"] = ["fictional"]
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_malformed_state_list_rejected_before_action(self):
        self.payload["variants"][0]["nodes"][0]["required_state"] = ""
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_disabled_producer(self):
        self.payload["variants"][0]["nodes"][0]["enabled"] = False
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_compatibility(self):
        self.payload["compatibility"]["engine"] = "different"
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_source_hash(self):
        self.payload["sources"]["adapter.py"] = "outdated"
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_multiple_producers(self):
        self.payload["variants"][0]["nodes"][0]["produced_state"] = ["answer", "answer"]
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_missing_state_producer(self):
        self.payload["variants"][0]["nodes"][0]["required_state"] = ["missing"]
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_adapter_hash(self):
        self.adapter.write_text("raise RuntimeError('must not import')")
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_cycle_normalized_before_import(self):
        self.payload["variants"][0]["nodes"][0]["dependencies"] = ["build"]
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_unknown_resource(self):
        self.payload["variants"][0]["nodes"][0]["resources"] = {"unknown": 1}
        with self.assertRaises(ContractRejected):
            self.load(allow_candidate=True)

    def test_discovery_covers_statements(self):
        source = self.root / "engine.py"
        source.write_text("class Engine:\n    def boot(self):\n        self.a()\n        if True:\n            self.b()\n")
        found = analyze(source, "Engine", "boot")
        self.assertEqual(len(found["regions"]), 2)
        self.assertEqual(found["regions"][1]["end"], 5)


if __name__ == "__main__":
    unittest.main()
