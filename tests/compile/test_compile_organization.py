# SPDX-License-Identifier: Apache-2.0
"""Boundaries and scheduling invariants of the compilation coordinator."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.compilation import backends
from vllm.compilation.compile_manager import CompilerManager
from vllm.compilation.compiler_interface import FullFxCompileBackend
from vllm.compilation.input_metadata import SubmoduleInputSerializer
from vllm.compilation.parallel_compile import ParallelCompileManager
from vllm.compilation.piecewise_inputs import PiecewiseCompileInterpreter


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_SUBPROCESS_COMPILE", "0")


def test_legacy_exports_are_identical():
    assert backends.CompilerManager is CompilerManager
    assert backends.ParallelCompileManager is ParallelCompileManager
    assert backends.SubmoduleInputSerializer is SubmoduleInputSerializer


def test_input_collection_and_restore_need_no_engine(tmp_path):
    piece = torch.fx.symbolic_trace(lambda x: x + 1)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_module("piece", (x,)))
    root = torch.nn.Module()
    root.add_module("piece", piece)
    module = torch.fx.GraphModule(root, graph)
    mode = FakeTensorMode()
    with mode:
        arg = torch.empty(4)
    x.meta["val"] = arg
    next(n for n in piece.graph.nodes if n.op == "placeholder").meta["val"] = arg
    store = SubmoduleInputSerializer(str(tmp_path), "toy", {"engine": "test"})
    first = PiecewiseCompileInterpreter(
        module,
        ["piece"],
        store,
        restore_input_metadata=False,
        structural_reuse=True,
        fake_mode=mode,
    )
    first.run(arg)
    second = PiecewiseCompileInterpreter(
        module,
        ["piece"],
        store,
        restore_input_metadata=True,
        structural_reuse=True,
        fake_mode=mode,
    )
    second.call_module = Mock(side_effect=AssertionError("unexpected fake propagation"))
    second.run(arg)
    assert second.metadata["piece"]["args"][0].shape == (4,)
    assert second.reuse_index_map == {0: 0}
    assert not hasattr(second, "vllm_backend")


def test_coordinator_preserves_objects_and_preparation_order(monkeypatch):
    order = []
    metadata, reuse = {}, {0: 0}
    collector = NS(
        metadata=metadata,
        reuse_index_map=reuse,
        compile_submod_names=["piece"],
        run=lambda *a: order.append("inputs"),
    )
    coordinator = ParallelCompileManager(1)

    def materialize(meta, names, compiler, config, engine, **kw):
        assert order == ["inputs"] and meta is metadata
        assert kw["reuse_index_map"] is reuse
        order.append("materialize")
        return {"piece": object()}

    monkeypatch.setattr(coordinator, "materialize_all", materialize)
    try:
        result, returned = coordinator.compile_subgraphs(
            lambda: collector, (), None, None, None
        )
        assert returned is collector and "piece" in result
        assert order == ["inputs", "materialize"]
    finally:
        coordinator.thread_executor.shutdown()


def test_all_process_jobs_submitted_before_wait(monkeypatch):
    events = []

    def submit(*args):
        index = sum(e[0] == "submit" for e in events)
        events.append(("submit", index))

        class Result:
            def add_done_callback(self, fn):
                self.callback = fn

            def result(self):
                assert [e for e in events if e[0] == "submit"] == [
                    ("submit", 0),
                    ("submit", 1),
                ]
                events.append(("wait", index))
                self.callback(self)
                return NS(
                    artifacts=[NS(cache_key=f"key{index}", artifact_path="unused")],
                    compile_time_s=0,
                    worker_pid=1,
                    serialized_input_nbytes=0,
                    timings={},
                )

        return Result()

    monkeypatch.setattr(FullFxCompileBackend, "configure", lambda *a, **k: None)
    monkeypatch.setattr(FullFxCompileBackend, "prewarm_futures", lambda: [])
    monkeypatch.setattr(FullFxCompileBackend, "submit", submit)
    monkeypatch.setattr(
        FullFxCompileBackend, "begin_shutdown", lambda: events.append(("shutdown", 0))
    )
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    metadata = {i: {"submod": graph, "args": ()} for i in range(2)}
    service = NS(compiler=NS(name="test"), record_compiled_handle=Mock())
    coordinator = ParallelCompileManager(2)
    try:
        success, failures = coordinator._compile_sources_in_subprocesses(
            [0, 1], metadata, service, NS(inductor_compile_config={}), None, None
        )
        assert success == {0, 1} and not failures
        assert events[:2] == [("submit", 0), ("submit", 1)]
        assert service.record_compiled_handle.call_count == 2
    finally:
        coordinator.thread_executor.shutdown()


def test_runtime_adapter_returns_original_future(monkeypatch):
    future = object()
    pool = NS(submit=Mock(return_value=future))
    monkeypatch.setattr(FullFxCompileBackend, "pool", lambda: pool)
    graph, inputs, config, contexts = object(), [], {}, ()
    assert FullFxCompileBackend.submit(graph, inputs, config, contexts) is future
    pool.submit.assert_called_once_with(
        graph,
        inputs,
        config,
        unsafe_force_cache=True,
        worker_context_factories=contexts,
    )


def test_compile_timers_are_isolated_by_manager_and_shape(monkeypatch):
    from vllm.compilation import compile_manager as module

    monkeypatch.setattr(module, "make_compiler", lambda _: object())
    first = CompilerManager(NS(compilation_time=0.0))
    second = CompilerManager(NS(compilation_time=0.0))
    ticks = iter([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    first._maybe_start_timer(0, True, None)
    second._maybe_start_timer(0, True, None)
    first._maybe_start_timer(0, True, 8)
    first._maybe_finish_timer(1, 2, None, True, "compile")
    second._maybe_finish_timer(1, 2, None, True, "load")
    first._maybe_finish_timer(1, 2, 8, True, "compile")
    assert first.compilation_config.compilation_time == 60.0
    assert second.compilation_config.compilation_time == 30.0
    assert not first._compilation_started and not second._compilation_started


def test_partial_compile_does_not_report_a_fabricated_batch_time(monkeypatch):
    from vllm.compilation import compile_manager as module

    monkeypatch.setattr(module, "make_compiler", lambda _: object())
    manager = CompilerManager(NS(compilation_time=0.0))
    manager._maybe_start_timer(0, False, None)
    manager._maybe_finish_timer(1, 2, None, True, "compile")
    assert manager.compilation_config.compilation_time == 0.0
