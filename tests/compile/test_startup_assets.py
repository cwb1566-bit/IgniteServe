# SPDX-License-Identifier: Apache-2.0
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.compilation import startup_assets as assets
from vllm.compilation import startup_resources as resources
from vllm.compilation.backends import SubmoduleInputSerializer
from vllm.compilation.input_contract import graph_key, input_contract
from vllm.v1.worker import kv_profile_cache as kv
from tools.startup_assets import export, import_archive, publish, verify


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(tmp_path / "assets"))
    resources._configured_root.cache_clear()
    assets.clear_prepared()
    yield tmp_path / "assets"
    assets.clear_prepared()
    resources._configured_root.cache_clear()


def test_root_priority(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_STARTUP_RESOURCE_ROOT")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert resources.resource_root() == tmp_path / "cache/igniteserve/assets"
    resources._configured_root.cache_clear()
    config = tmp_path / "config/igniteserve/config.json"
    assets.write_json(config, {"startup_resource_root": str(tmp_path / "configured")})
    assert resources.resource_root() == tmp_path / "configured"
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(tmp_path / "override"))
    assert resources.resource_root() == tmp_path / "override"


def test_root_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(alias))
    assert resources.resource_root() == real


def test_record_validation(root):
    identity = {"model": "x", "dtype": "bfloat16"}
    path = assets.save("kv_profile", identity, {"bytes": 42})
    assert path.parent == root / "kv_profile"
    assert assets.load("kv_profile", identity) == {"bytes": 42}
    assert assets.load("kv_profile", {**identity, "dtype": "float16"}) is None
    value = json.loads(path.read_text())
    value["data"]["bytes"] = 43
    assets.write_json(path, value)
    assert assets.load("kv_profile", identity) is None
    assert verify(root)["failures"]


def test_prepared_records_reuse_without_bypassing_validation(root, monkeypatch):
    key = {"model": "x"}
    path = assets.save("kv_profile", key, {"peak": 42})
    assert assets.prefetch([path])["files"] == 1
    def forbid_read(*args, **kwargs):
        raise AssertionError("prepared record was read again")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", forbid_read)
        value = assets.read_record(path, key)
        assert value == {"peak": 42}
        value["peak"] = 123
        assert assets.read_record(path, key) == {"peak": 42}
        assert assets.read_record(path, {"model": "wrong"}) is None
    assets.write_json(path, assets.record(key, {"peak": 99}))
    assert assets.read_record(path, key) == {"peak": 99}
    path.unlink()
    assert assets.read_record(path, key) is None


def test_prefetch_corruption_bounds_and_release(root):
    bad = root / "bad.json"
    assets.atomic_write(bad, b"not-json")
    path = assets.save("kv_profile", {}, {"peak": 42})
    assert assets.prefetch([bad, path], max_bytes=1)["files"] == 0
    assert assets.prefetch([bad, path])["files"] == 1
    corrupted = assets.record({}, {"peak": 42})
    corrupted["checksum"] = "invalid"
    assets.write_json(path, corrupted)
    assets.clear_prepared()
    assert assets.prefetch([path])["files"] == 1
    assert assets.read_record(path, {}) is None
    summary = assets.clear_prepared()
    assert summary["reused_reads"] == 1
    assert assets.clear_prepared()["files"] == 0


def test_prepared_legacy_override_without_json_suffix(root):
    override = str(root / "legacy-cache")
    path = assets.save("kv_profile", {"model": "x"}, {"peak": 42}, override)
    assets.prefetch([path])
    assert assets.load("kv_profile", {"model": "x"}, override) == {"peak": 42}


def test_cached_graph_binding_is_serial_across_adaptors(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from torch._guards import TracingContext
    from torch._inductor.codecache import FxGraphCache
    from vllm.compilation.compiler_interface import ThreadSafeInductorAdaptor

    active, peak = 0, 0
    lock, barrier = threading.Lock(), threading.Barrier(4)
    def lookup(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.02)
        with lock:
            active -= 1
        return lambda args: tuple(args), None
    monkeypatch.setattr(FxGraphCache, "_lookup_graph", lookup)
    monkeypatch.setattr(TracingContext, "try_get", lambda: object())
    monkeypatch.setattr(ThreadSafeInductorAdaptor, "metrics_context", lambda _: nullcontext())
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output((x,))
    gm = torch.fx.GraphModule({}, graph)
    def load(index):
        barrier.wait()
        return ThreadSafeInductorAdaptor().load((str(index), "file"), gm, [], index)(index)
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(load, range(4))) == [(0,), (1,), (2,), (3,)]
    assert peak == 1


def test_identity_json_roundtrip(root):
    identity = {"labels": {0: "zero"}, "partition": (1, 2)}
    assets.save("kv_profile", identity, {"peak": 123})
    assert assets.load("kv_profile", identity) == {"peak": 123}


def _save_external(path, index):
    assets.save("best_config", {"index": index}, index, path)


def test_concurrent_legacy_override(tmp_path):
    path = str(tmp_path / "legacy.json")
    processes = [multiprocessing.get_context("fork").Process(target=_save_external,
                 args=(path, i)) for i in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert all(assets.load("best_config", {"index": i}, path) == i for i in range(8))


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../../escape", "a\\b"])
def test_path_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        assets.member(tmp_path, relative)


def test_bundle_integrity_and_symlink(tmp_path):
    assets.atomic_write(tmp_path / "data", b"abc")
    assets.publish_bundle(tmp_path, {"type": "test"}, ["data"])
    assert assets.verify_bundle(tmp_path, {"type": "test"})
    assert not assets.verify_bundle(tmp_path, {"type": "other"})
    (tmp_path / "data").unlink()
    external = tmp_path.parent / "outside"
    external.write_bytes(b"abc")
    (tmp_path / "data").symlink_to(external)
    assert not assets.verify_bundle(tmp_path, {"type": "test"})


def test_relocation_and_import(root, tmp_path):
    assets.save("best_config", {"kernel": "x"}, {"warps": 4})
    archive = tmp_path / "assets.zip"
    export(root, archive)
    moved = tmp_path / "moved"
    import_archive(moved, archive, False)
    assert not verify(moved)["failures"]
    path = next((moved / "best_config").glob("*.json"))
    assert assets.read_record(path, {"kernel": "x"}) == {"warps": 4}


def test_executable_import_requires_trust(root, tmp_path):
    bundle = root / "dynamo_frontend/x"
    assets.atomic_write(bundle / "bytecode", b"not executed")
    assets.publish_bundle(bundle, {"runtime": "x"}, ["bytecode"])
    archive = tmp_path / "code.zip"
    export(root, archive)
    with pytest.raises(ValueError, match="untrusted"):
        import_archive(tmp_path / "moved", archive, False)
    import_archive(tmp_path / "moved", archive, True)


def test_stride_restore(tmp_path):
    serializer = SubmoduleInputSerializer(str(tmp_path / "inputs.json"), "model")
    mode = FakeTensorMode()
    with mode:
        tensor = torch.empty_strided((2, 3), (1, 2))
    meta = serializer.extract_submodule_metadata("a", (tensor,), {}, [], 0)
    serializer.save_all_metadata([meta])
    loaded = serializer.load_all_metadata()[0]
    args, _ = serializer.reconstruct_submodule_inputs(loaded, mode)
    assert args[0].stride() == (1, 2)
    assert input_contract(args) == input_contract((tensor,))


def test_nested_containers(tmp_path):
    serializer = SubmoduleInputSerializer(str(tmp_path), "model")
    mode = FakeTensorMode()
    with mode:
        tensor = torch.empty(3)
    data = ([tensor, (4, None)],)
    meta = serializer.extract_submodule_metadata("a", data, {"x": (tensor,)}, [], 0)
    serializer.save_all_metadata([meta])
    loaded = serializer.load_all_metadata()[0]
    args, kwargs = serializer.reconstruct_submodule_inputs(loaded, mode)
    assert isinstance(args[0][1], tuple)
    assert isinstance(kwargs["x"], tuple)


def test_alias_and_offset_not_silently_erased(tmp_path):
    serializer = SubmoduleInputSerializer(str(tmp_path), "model")
    mode = FakeTensorMode()
    with mode:
        tensor = torch.empty(8)
        view = tensor[1:]
    meta = serializer.extract_tensor_metadata(view)
    with pytest.raises(ValueError, match="offset"):
        serializer.reconstruct_fake_tensor(meta, mode)
    pair = serializer.extract_submodule_metadata("a", (tensor, tensor), {}, [], 0)
    args, _ = serializer.reconstruct_submodule_inputs(pair, mode)
    assert input_contract(args) != input_contract((tensor, tensor))


def test_unknown_symbol_rejected(tmp_path):
    serializer = SubmoduleInputSerializer(str(tmp_path), "model")
    with pytest.raises(ValueError, match="unbound"):
        serializer._create_symint("s0 * 2", FakeTensorMode())


def test_graph_topology_and_constants():
    def graph(multiplier, subtract=False):
        g = torch.fx.Graph()
        x = g.placeholder("renamable")
        a = g.call_function(torch.mul, (x, multiplier))
        b = g.call_function(torch.sub if subtract else torch.add, (a, x))
        g.output(b)
        return torch.fx.GraphModule({}, g)
    assert graph_key(graph(2)) == graph_key(graph(2))
    assert graph_key(graph(2)) != graph_key(graph(3))
    assert graph_key(graph(2)) != graph_key(graph(2, True))


def test_graph_ellipsis_is_stable():
    import operator
    g = torch.fx.Graph()
    x = g.placeholder("x")
    g.output(g.call_function(operator.getitem, (x, (Ellipsis, slice(None)))))
    module = torch.fx.GraphModule({}, g)
    assert graph_key(module) == graph_key(module)


def test_compiler_handle_relocation(root, tmp_path, monkeypatch):
    from vllm.compilation.backends import CompilerManager
    manager = object.__new__(CompilerManager)
    saved = manager._disk_handle(("hash", str(root / "compile/x/code.py")))
    assert saved == ("hash", "compile/x/code.py", "repository-relative")
    moved = tmp_path / "moved"
    monkeypatch.setenv("VLLM_STARTUP_RESOURCE_ROOT", str(moved))
    assert manager._disk_handle(saved, restore=True) == ("hash", str(moved / "compile/x/code.py"))


def test_workload_hotspot_profiles_are_distinct(root, tmp_path):
    from vllm.v1.worker.cudagraph_profile import load_cudagraph_profile_manifest
    source = tmp_path / "offline"
    source.mkdir()
    identity = {"schema_version": 1, "model": {"name": "test"}}
    assets.write_json(source / "cuda_graph_profile_manifest.json", identity)
    (source / "cuda_graph_hits.csv").write_text("batch_size,hit_count\n1,100\n")
    assets.write_json(source / "workload.json", {"rps": 1, "requests": 2000})
    first = Path(publish(root, "hotspots", source)["published"])
    assets.write_json(source / "workload.json", {"rps": 10, "requests": 2000})
    second = Path(publish(root, "hotspots", source)["published"])
    assert first != second
    assert load_cudagraph_profile_manifest(first) == identity


def test_rejected_artifact_recompiles(monkeypatch):
    from vllm.compilation.backends import ParallelCompileManager
    manager = ParallelCompileManager(1)
    monkeypatch.setattr(manager, "_can_use_subprocess_compile", lambda *a: False)
    monkeypatch.delenv("VLLM_ENABLE_SUBPROCESS_COMPILE", raising=False)
    compiler = SimpleNamespace(semantic_identity=lambda *a: None,
                               has_cached_graph=lambda *a: True,
                               load=lambda *a, **kw: None)
    runnable = lambda x: x + 1
    monkeypatch.setattr(manager, "_compile_source_graph_serial",
                        lambda *a: (runnable, "compile"))
    try:
        result = manager.materialize_all({"a": {"index": 0, "submod": None,
                                         "args": ()}}, ["a"], compiler,
                                         SimpleNamespace(compilation_time=0), None)
        assert result["a"](2) == 3
    finally:
        manager.thread_executor.shutdown()


def test_kv_budget_recomputed(root, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_KV_PROFILE_CACHE", "1")
    identity = {"device": {"gpu_total_memory": 2_000_000_000}}
    monkeypatch.setattr(kv, "build_key", lambda *_: (assets.digest(identity), identity))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (1_800_000_000, 2_000_000_000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 100)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 100)
    profile = SimpleNamespace(torch_peak_increase=200, non_torch_increase=50, non_kv_cache_memory=350)
    _, path = kv.save(None, None, 999, profile_result=profile, weights_memory=100)
    assets.prefetch([path])
    value, *_ = kv.lookup(None, None, weights_memory=100, requested_memory=1000,
                          init_snapshot=SimpleNamespace(free_memory=2000))
    assert value == 650 and value != 999
    value, *_ = kv.lookup(None, None, weights_memory=101, requested_memory=1000,
                          init_snapshot=SimpleNamespace(free_memory=2000))
    assert value is None
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (100, 2_000_000_000))
    value, *_ = kv.lookup(None, None, weights_memory=100, requested_memory=1000,
                          init_snapshot=SimpleNamespace(free_memory=2000))
    assert value is None  # A prepared file cannot certify current GPU headroom.


def test_frozen_hotspot_publish(root, tmp_path):
    from vllm.v1.worker.cudagraph_profile import (
        ensure_cudagraph_profile_manifest, load_cudagraph_profile_manifest)
    source = tmp_path / "offline"
    identity = {"schema_version": 1, "model": {"hidden_size": 4}}
    ensure_cudagraph_profile_manifest(source, identity)
    (source / "cuda_graph_hits.csv").write_text("batch_size,hit_count\n1,100\n")
    result = publish(root, "hotspots", source)
    bundle = Path(result["published"])
    assert load_cudagraph_profile_manifest(bundle) == identity
    assert not (bundle / "cuda_graph_profile_manifest.json").exists()
    (source / "cuda_graph_hits.csv").write_text("changed")
    assert assets.verify_bundle(bundle, identity)
