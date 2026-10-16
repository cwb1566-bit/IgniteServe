from types import SimpleNamespace as NS
from threading import Event
from unittest.mock import Mock

import pytest

from vllm.v1.worker.CudaGraphManager import CUDAGraphManager, GraphState
from vllm.v1.worker.HotspotAnalyzer import HotspotAnalyzer
from vllm.v1.worker.cudagraph_adaptation import GraphAdapter
from vllm.v1.worker.piecewise_graph_registry import PiecewiseGraphRegistry


def manager(min_hits=2, enabled=True):
    m = CUDAGraphManager(enabled=enabled, min_hits=min_hits)
    m.runner = NS(
        use_cuda_graph=True,
        cudagraph_batch_sizes=[1, 2, 4, 8],
        vllm_config=NS(
            pad_for_cudagraph=lambda n: next(s for s in [1, 2, 4, 8] if s >= n)
        ),
        parallel_config=NS(world_size=1),
    )
    m.adapter = GraphAdapter(registry=m.registry)
    m.adapter.pad_limit = lambda: 0.2
    m.adapter.plan = Mock(return_value=("split",))
    m.capture = NS(enabled=True, enqueue_after_warmup=Mock(), shutdown=Mock())
    m.cross_phase = True
    m.capture_phases = frozenset({"decode", "mixed", "prefill"})
    return m


def select(m, phase="decode"):
    return m.select(phase, 7, {}, [1] * 7, [16] * 7, [phase != "decode"] * 7)


def test_select_counts_once_and_warmup_overrides_split():
    m = manager()
    d, parts = select(m)
    assert parts and not d.mark_runtime_warmup
    d, parts = select(m)
    assert not parts and d.mark_runtime_warmup and d.execution_shape == 8
    m.after_warmup(d)
    m.capture.enqueue_after_warmup.assert_called_once_with("decode", 8)
    assert m.hotspots.total_requests == 2
    assert m.hotspots.actual_stats == {7: 2}
    assert m.hotspots.stats[8].hit_count == m.snapshot()[0].hit_count == 2
    d, parts = select(m)
    assert parts and not d.mark_runtime_warmup


def test_shared_shape_does_not_warm_or_queue_twice():
    m = manager(min_hits=1)
    d, _ = select(m, "decode")
    m.after_warmup(d)
    d, parts = select(m, "mixed")
    assert parts and not d.mark_runtime_warmup
    m.capture.enqueue_after_warmup.assert_called_once()


def test_instances_do_not_share_hotspots_or_state():
    a, b = manager(), manager()
    select(a)
    assert not b.snapshot() and b.hotspots.total_requests == 0
    assert not b.registry.published


def test_distributed_policy_keeps_collective_shapes_identical():
    waiting, warming = manager(min_hits=10), manager(min_hits=1)
    for m in (waiting, warming):
        m.runner.parallel_config.world_size = 2
        m.adapter.find_best_match = Mock(side_effect=AssertionError('rank-local cost policy'))
    a, parts_a = select(waiting)
    b, parts_b = select(warming)
    assert a.execution_shape == b.execution_shape == 8
    assert not a.mark_runtime_warmup and b.mark_runtime_warmup
    assert not parts_a and not parts_b
    warming.mark_capture_failed('decode', 8, RuntimeError('test failure'))
    failed, parts = select(warming, 'mixed')
    assert failed.execution_shape == 8 and not parts
    waiting.adapter.plan.assert_not_called()
    warming.adapter.plan.assert_not_called()


def test_distributed_disabled_adaptive_still_uses_common_shape():
    m = manager(enabled=False)
    m.runner.parallel_config.world_size = 2
    d, parts = select(m)
    assert d.skip_cudagraph and d.execution_shape == 8 and not parts


def test_distributed_precapture_uses_leader_shapes(monkeypatch):
    from vllm.distributed import parallel_state
    m = manager()
    m.runner.parallel_config = NS(world_size=2, data_parallel_size=1)
    m.profile_dir = m.profile_manifest = None
    m.hotspots.select_precapture = Mock(return_value=[2])
    world = NS(broadcast_object=Mock(return_value=[1, 2]))
    monkeypatch.setattr(parallel_state, 'get_world_group', lambda: world)
    m.capture.precapture = Mock()
    m.precapture()
    world.broadcast_object.assert_called_once_with([2], src=0)
    m.capture.precapture.assert_called_once_with([1, 2])


def test_disabled_adaptive_has_no_hidden_legacy_capture():
    m = manager(enabled=False)
    d, parts = select(m)
    assert d.skip_cudagraph and not parts and not d.mark_runtime_warmup
    m.adapter.plan.assert_not_called()
    m.registry.publish(8)
    d, _ = select(m)
    assert not d.skip_cudagraph and d.execution_shape == 8
    m.capture.enqueue_after_warmup.assert_not_called()


def test_distributed_idle_needs_coordinated_admission():
    m = manager()
    m.capture_pending = Mock()
    m.runner.parallel_config.world_size = 2
    m.on_idle()
    m.capture_pending.assert_not_called()
    m.runner.parallel_config.world_size = 1
    m.on_idle()
    m.capture_pending.assert_called_once()


def test_background_owner_blocks_safe_point_competitor():
    m = manager()
    m.capture.capture_pending = Mock()
    assert m.capture_pending(selected_shape=8) == 0
    m.capture.capture_pending.assert_not_called()
    m.capture.enabled = False
    m.capture.capture_pending.return_value = 1
    assert m.capture_pending(selected_shape=8) == 1


def test_background_owner_declines_maintenance_admission():
    m = manager(min_hits=1)
    d, _ = select(m)
    m.after_warmup(d)
    assert m.plan_capture_maintenance(100, 2, 1, 1) is None
    m.capture.enabled = False
    assert m.plan_capture_maintenance(100, 2, 1, 1).shape == 8


def test_safe_point_budget_keeps_partial_graph_unpublished(monkeypatch):
    from contextlib import nullcontext
    from vllm.v1.worker.concurrent_piecewise_cudagraph import CaptureExecutor

    m = manager(min_hits=1)
    m.capture = object.__new__(CaptureExecutor)
    m.capture.manager, m.capture.runner = m, m.runner
    m.capture.enabled = False
    m.runner.device, m.runner.full_cuda_graph = "cpu", False
    m.gc_scope = NS(fast_capture_scope=nullcontext)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.graph_capture",
        lambda **kw: nullcontext(),
    )
    calls = []

    def forward(shape, **kwargs):
        budget = kwargs["cudagraph_capture_budget"]
        calls.append(shape)
        budget.total_piecewise_compiles = 2
        budget.observed_piece_ids.update({0, 1})
        budget.mark_captured(len(calls) - 1, 0.001)
        if len(calls) == 1:
            budget.defer(1, 2)

    m.runner._dummy_run = forward
    d, _ = select(m)
    m.finish_runtime_warmup(d.phase, d.target_shape)
    assert m.capture_pending(max_pieces_per_shape=1, selected_shape=8) == 1
    assert m.snapshot()[0].state == GraphState.PARTIAL
    assert not m.registry.published
    assert m.capture_pending(max_pieces_per_shape=1, selected_shape=8) == 1
    assert m.snapshot()[0].state == GraphState.READY
    assert m.registry.published == (8,)


def test_registry_never_publishes_incomplete_chain():
    entry = NS(cudagraph=object())
    b = NS(
        piecewise_compile_index=0,
        total_piecewise_compiles=2,
        concrete_size_entries={8: entry},
        finish_concurrent_capture=Mock(return_value=True),
    )
    registry = PiecewiseGraphRegistry(NS(cuda_piecewise_backends_list={0: b}))
    gate = Event()
    with pytest.raises(RuntimeError, match="incomplete"):
        registry.publish(8, dynamic=True, gate=gate)
    assert not registry.published and not gate.is_set()
    b.total_piecewise_compiles = 1
    registry.publish(8, dynamic=True, gate=gate)
    assert (
        registry.published == (8,)
        and registry.dynamic_shapes == {8}
        and gate.is_set()
    )


def test_top99_precedes_filter_and_mandatory_shapes(tmp_path, monkeypatch):
    from vllm.v1.worker import HotspotAnalyzer as module

    monkeypatch.setattr(
        module, "validate_cudagraph_profile_manifest", lambda *a: (True, "test")
    )
    monkeypatch.setattr(
        "vllm.compilation.startup_assets.read_text", lambda p: p.read_text()
    )
    for key in (
        "VLLM_CUDA_GRAPH_HOTSPOT_TOP_K",
        "VLLM_CUDA_GRAPH_HOTSPOT_DIR",
        "VLLM_CUDA_GRAPH_HOTSPOT_CSV",
        "VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_CUDA_GRAPH_PRECAPTURE_MIN_HITS", "10")
    # The oversized shape belongs to the prefix; filtering it must not refill
    # the tail with eligible shape 8 (10 hits).
    (tmp_path / "cuda_graph_hits.csv").write_text(
        "batch_size,hit_count\n4,990\n999,990\n16,9\n8,10\n"
    )
    a = HotspotAnalyzer()
    assert a.select_precapture(
        [1, 2, 4, 8, 16], tmp_path, {}, adaptive=True
    ) == [1, 2, 4]


def test_incompatible_profile_falls_back_without_overwriting(
    tmp_path, monkeypatch
):
    from vllm.v1.worker import HotspotAnalyzer as module

    monkeypatch.setattr(
        module,
        "validate_cudagraph_profile_manifest",
        lambda *a: (False, "wrong GPU"),
    )
    path = tmp_path / "cuda_graph_hits.csv"
    path.write_text("batch_size,hit_count\n4,100\n")
    before = path.read_bytes()
    for key in (
        "VLLM_CUDA_GRAPH_HOTSPOT_TOP_K",
        "VLLM_CUDA_GRAPH_HOTSPOT_DIR",
        "VLLM_CUDA_GRAPH_HOTSPOT_CSV",
        "VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE",
    ):
        monkeypatch.delenv(key, raising=False)
    assert HotspotAnalyzer().select_precapture(
        [1, 2, 4, 8], tmp_path, {}, adaptive=True
    ) == [1, 2, 4, 8]
    assert path.read_bytes() == before
