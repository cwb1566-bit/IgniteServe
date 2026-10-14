# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm.compilation import startup_assets as assets
from vllm.compilation import startup_resources as resources
from vllm.compilation.backends import CompilerManager, ParallelCompileManager
from vllm.compilation.cache_reuse import GlobalSubgraphCache, subgraph_identity
from vllm.compilation.compiler_interface import ThreadSafeInductorAdaptor
from vllm.compilation.input_contract import input_contract


@pytest.fixture
def root(tmp_path, monkeypatch):
    path = tmp_path / "assets"
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(path))
    monkeypatch.setenv("VLLM_ENABLE_STRUCTURAL_REUSE", "1")
    monkeypatch.setenv("VLLM_ENABLE_SUBPROCESS_COMPILE", "0")
    resources._configured_root.cache_clear()
    yield path
    resources._configured_root.cache_clear()


def graph_and_inputs():
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    with FakeTensorMode(shape_env=ShapeEnv()):
        x = torch.empty(4)
        y = x + 1
    for node in graph.graph.nodes:
        node.meta["val"] = x if node.op == "placeholder" else y
    return graph, [x]


def identity(graph, inputs):
    return subgraph_identity(graph, inputs, {"compiler": "test", "tp": 1})


def publish(root, value):
    local = root / "compile/launch-a/rank_0_0"
    code = local / "inductor_cache/aa/code.py"
    fx = local / "inductor_cache/fxgraph/ab/fabcd/variant"
    assets.atomic_write(code, b"# generated code\n")
    assets.atomic_write(fx, b"trusted backend payload")
    cache = GlobalSubgraphCache(str(local))
    assert cache.register(value, ("fabcd", str(code)), 3, None)
    return cache, code, fx


def test_key_ignores_index_and_node_names_but_preserves_semantics():
    graph, inputs = graph_and_inputs()
    before = identity(graph, inputs)
    renamed = copy.deepcopy(graph)
    for i, n in enumerate(renamed.graph.nodes):
        n.name = f"renamed_{i}"
    assert identity(renamed, inputs) == before
    op = next(n for n in renamed.graph.nodes if n.op == "call_function")
    op.args = (op.args[0], 2)
    assert identity(renamed, inputs) != before
    assert subgraph_identity(graph, inputs, {"compiler": "new", "tp": 1}) != before
    assert subgraph_identity(graph, inputs, {"compiler": "test", "tp": 2}) != before
    assert subgraph_identity(graph, inputs, before["context"], 16) != before


def test_key_preserves_output_signature_and_input_aliases():
    graph, inputs = graph_and_inputs()
    before = identity(graph, inputs)
    with FakeTensorMode(shape_env=ShapeEnv()):
        changed = torch.empty(4, dtype=torch.float64)
    next(n for n in graph.graph.nodes if n.op == "call_function").meta["val"] = changed
    assert identity(graph, inputs) != before
    assert input_contract((inputs[0], inputs[0])) != input_contract((inputs[0], changed))
    next(n for n in graph.graph.nodes if n.op == "call_function").meta.clear()
    with pytest.raises(KeyError):
        identity(graph, inputs)


def test_output_aliases_survive_reconstructed_input_storage():
    graph, inputs = graph_and_inputs()
    output = next(n for n in graph.graph.nodes if n.op == "output")
    placeholder = next(n for n in graph.graph.nodes if n.op == "placeholder")
    output.args = ((placeholder, output.args[0]),)
    with FakeTensorMode(shape_env=ShapeEnv()):
        restored = torch.empty(4)
    assert identity(graph, inputs) == identity(graph, [restored])
    with pytest.raises(ValueError, match="evidence"):
        identity(graph, [restored.double()])
    with pytest.raises(ValueError, match="frozen"):
        subgraph_identity(graph, inputs, {"inductor": {"freezing": True}})


def test_symbol_renaming_preserves_relationships():
    first, second = ShapeEnv(), ShapeEnv()
    a = first.create_unbacked_symint()
    second.create_unbacked_symint()
    b = second.create_unbacked_symint()
    assert input_contract((a, 2 * a), canonical=True) == input_contract((b, 2 * b), canonical=True)
    assert input_contract((a, 2 * a), canonical=True) != input_contract((b, b + 1), canonical=True)


def test_unknown_code_dependencies_not_globally_reused():
    graph, inputs = graph_and_inputs()
    op = next(n for n in graph.graph.nodes if n.op == "call_function")
    def opaque(x, value):
        return x + value
    op.target = opaque
    with pytest.raises(ValueError, match="code dependency"):
        identity(graph, inputs)


def test_global_lookup_across_launch_and_index(root):
    graph, inputs = graph_and_inputs()
    value = identity(graph, inputs)
    cache, code, fx = publish(root, value)
    other = GlobalSubgraphCache(str(root / "compile/launch-b/rank_0_0"))
    entry = other.lookup(value)
    assert entry["graph_index"] == 3
    assert list(other.payloads(entry)) == [fx.read_bytes()]
    assert not Path(entry["handle"][1]).is_absolute()
    assert other.lookup({**value, "runtime_shape": 16}) is None
    fx.write_bytes(b"corrupt")
    assert other.lookup(value) is None
    from tools.startup_assets import verify
    assert verify(root)["failures"]
    with pytest.raises(ValueError):
        list(other.payloads(entry))


def test_missing_or_escaping_artifact_is_miss(root):
    graph, inputs = graph_and_inputs()
    value = identity(graph, inputs)
    cache, code, fx = publish(root, value)
    code.unlink()
    assert cache.lookup(value) is None
    cache, code, fx = publish(root, value)
    path = cache.path / f"{assets.digest(value)}.json"
    record = json.loads(path.read_text())
    record["data"]["files"]["../outside"] = "x"
    assets.write_json(path, assets.record(value, record["data"]))
    assert cache.lookup(value) is None


def test_relocated_repository_and_external_cache(root, tmp_path, monkeypatch):
    graph, inputs = graph_and_inputs()
    value = identity(graph, inputs)
    cache, _, _ = publish(root, value)
    moved = tmp_path / "moved"
    root.rename(moved)
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(moved))
    assert GlobalSubgraphCache(str(moved / "compile/launch-b")).lookup(value)
    external = tmp_path / "external"
    external.mkdir()
    assert not GlobalSubgraphCache(str(external)).register(value, ("fabcd", str(external / "code")), 0, None)


def manager(root, graph, inputs):
    from vllm.config import CompilationConfig
    manager = CompilerManager(CompilationConfig())
    manager.initialize_cache(str(root / "compile/launch-b/rank_0_0"), identity={})
    manager.global_cache = GlobalSubgraphCache(manager.cache_dir)
    manager.reuse_context = identity(graph, inputs)["context"]
    return manager


def test_global_guards_rejected_then_compile(root, monkeypatch):
    graph, inputs = graph_and_inputs()
    publish(root, identity(graph, inputs))
    mgr = manager(root, graph, inputs)
    calls = []
    def reject(*args, **kwargs):
        calls.append("load")
        raise ValueError("guard mismatch")
    monkeypatch.setattr(mgr.compiler, "load", reject)
    monkeypatch.setattr(mgr.compiler, "compile", lambda *a, **k: (calls.append("compile") or (lambda x: x), None))
    assert callable(mgr.compile(graph, inputs, {}, mgr.compilation_config, graph_index=8, track_metrics=False))
    assert calls == ["load", "compile"]


def test_instance_reuse_respects_structural_override(root, monkeypatch):
    graph, inputs = graph_and_inputs()
    mgr = manager(root, graph, inputs)
    mgr.global_cache = None
    compiled = lambda x: x + 1
    mgr._remember_semantic(graph, inputs, 0, None, compiled, None)
    assert mgr.load(graph, inputs, 8, track_metrics=False) is compiled
    monkeypatch.setenv("VLLM_DISABLE_STRUCTURAL_REUSE", "1")
    assert mgr.load(graph, inputs, 9, track_metrics=False) is None


def test_parallel_global_lookup_precedes_subprocess_and_deduplicates(root, monkeypatch):
    graph, inputs = graph_and_inputs()
    publish(root, identity(graph, inputs))
    mgr = manager(root, graph, inputs)
    loads = []
    def load(*args, **kwargs):
        loads.append(list(kwargs["artifact_payloads"]))
        return lambda x: x + 1
    monkeypatch.setattr(mgr.compiler, "load", load)
    pool = ParallelCompileManager(2)
    monkeypatch.setattr(pool, "_can_use_subprocess_compile", lambda *a: True)
    monkeypatch.setattr(pool, "_compile_sources_in_subprocesses", lambda *a: pytest.fail("global hit submitted for compilation"))
    metadata = {f"g{i}": {"submod": graph, "args": inputs, "kwargs": {}, "index": i}
                for i in (0, 1)}
    try:
        result = pool.materialize_all(metadata, list(metadata), mgr,
                                      mgr.compilation_config, SimpleNamespace())
        assert len(loads) == 1
        assert result["g0"] is result["g1"]
    finally:
        pool.thread_executor.shutdown()


def test_parallel_cold_identical_subgraphs_compile_once(root, monkeypatch):
    graph, inputs = graph_and_inputs()
    mgr = manager(root, graph, inputs)
    pool = ParallelCompileManager(2)
    calls = []
    monkeypatch.setattr(pool, "_can_use_subprocess_compile", lambda *a: False)
    def compile(*args):
        calls.append(args[3])
        return (lambda x: x + 1), "compile"
    monkeypatch.setattr(pool, "_compile_source_graph_serial", compile)
    metadata = {f"g{i}": {"submod": graph, "args": inputs, "kwargs": {}, "index": i}
                for i in (0, 1)}
    try:
        result = pool.materialize_all(metadata, list(metadata), mgr,
                                      mgr.compilation_config, SimpleNamespace())
        assert calls == [0]
        assert result["g0"] is result["g1"]
    finally:
        pool.thread_executor.shutdown()


def test_subprocess_handle_published_only_after_successful_load(root, monkeypatch):
    graph, inputs = graph_and_inputs()
    mgr = manager(root, graph, inputs)
    cache, code, fx = publish(root, identity(graph, inputs))
    mgr.global_cache = cache
    mgr.cache_dir = str(cache.local)
    handle = ("fabcd", str(code))
    mgr.record_compiled_handle(8, None, handle, graph, inputs)
    assert not mgr.pending_artifacts
    monkeypatch.setattr(mgr.compiler, "load", lambda *a, **k: (lambda x: x))
    assert callable(mgr.load(graph, inputs, 8, track_metrics=False))
    assert len(mgr.pending_artifacts) == 1


def test_loader_uses_source_bytes_and_real_shape_env(monkeypatch):
    from torch._inductor.codecache import FxGraphCache
    from torch._guards import TracingContext
    graph, inputs = graph_and_inputs()
    seen = []
    def lookup(key, args, local, remote, constants):
        assert not local
        assert TracingContext.get().fake_mode.shape_env is not None
        from torch._dynamo.utils import CompileEventLogger
        CompileEventLogger.increment_toplevel("num_triton_bundles")
        import base64
        seen.append(base64.b64decode(remote.get(key)["data"]))
        return (lambda args: (args[0] + 1,)), {}
    monkeypatch.setattr(FxGraphCache, "_lookup_graph", lookup)
    compiled = ThreadSafeInductorAdaptor().load(("fabcd", "unused"), graph, inputs, 0,
                                               artifact_payloads=iter([b"payload"]))
    assert callable(compiled)
    assert seen == [b"payload"]


@pytest.mark.skipif(os.getenv("VLLM_TEST_REUSE_GPU") != "1", reason="explicit GPU integration test")
def test_real_inductor_cross_process_reuse(root):
    import subprocess
    import sys
    for phase in ("producer", "consumer"):
        result = subprocess.run([sys.executable, __file__, str(root), phase],
                                text=True, capture_output=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"REUSE_OK {phase}" in result.stdout


def _gpu_process(root, phase):
    from vllm.config import CompilationConfig
    from vllm.compilation.cache_reuse import compilation_context
    os.environ["VLLM_STARTUP_RESOURCE_ROOT"] = root
    os.environ["VLLM_ENABLE_GLOBAL_SUBGRAPH_CACHE"] = "1"
    os.environ["VLLM_ENABLE_SUBPROCESS_COMPILE"] = "0"
    mgr = CompilerManager(CompilationConfig())
    mgr.initialize_cache(str(Path(root) / "compile" / phase / "rank_0_0"),
                         identity={"launch": phase})
    config = SimpleNamespace(parallel_config=SimpleNamespace(
        compute_hash=lambda: "single", rank=0, data_parallel_rank=0))
    mgr.reuse_context = compilation_context(config, mgr.compiler, {})
    index = 0 if phase == "producer" else 8

    def backend(graph, inputs):
        assert mgr.semantic_identity(graph, inputs) is not None
        if phase == "consumer":
            from concurrent.futures import ThreadPoolExecutor
            # vLLM submits profiled FakeTensor inputs, not Dynamo's real tensors.
            load_inputs = [n.meta["example_value"] for n in graph.graph.nodes
                           if n.op == "placeholder"]
            with ThreadPoolExecutor(1) as loader:
                compiled = loader.submit(mgr.load, graph, load_inputs, index,
                                         track_metrics=False).result()
            assert compiled is not None, "cross-process lookup missed"
        else:
            compiled = mgr.compile(graph, inputs, {}, mgr.compilation_config,
                                   graph_index=index, track_metrics=False)
            mgr.save_to_file()
            assert list(mgr.global_cache.path.glob("*.json")), "no SubArtifact published"
        return compiled

    def function(x, weight):
        return torch.sin(x) * weight + 1

    x = torch.randn(1024, device="cuda")
    # Different parameter values in the consumer must bind to today's inputs.
    weight = torch.randn_like(x) + (10 if phase == "consumer" else 0)
    compiled = torch.compile(function, backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(x, weight), function(x, weight))
    print(f"REUSE_OK {phase}")


if __name__ == "__main__":
    import sys
    _gpu_process(sys.argv[1], sys.argv[2])
