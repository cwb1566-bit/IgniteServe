from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.worker.GraphMatcher import GraphMatcher, GraphCostTable
from vllm.v1.worker.cudagraph_adaptation import (
    GraphAdapter, GraphPartition, plan_partitions, workload_key)


def put(table, mode, shape, query, seq, flags, ms, parts=1):
    table._workloads[(mode, shape, workload_key(query, seq, flags), parts)] = ms


def ready_registry(*sizes):
    return NS(cuda_piecewise_backends_list={0: NS(total_piecewise_compiles=1,
        concrete_size_entries={size: NS(use_cudagraph=True, cudagraph=object(),
            concurrent_capture_in_progress=False, concurrent_capture_gate=None)
            for size in sizes})})


def test_costs_request_boundaries_and_overhead():
    m = GraphMatcher([2, 4])
    q, s, f = [1] * 6, [128] * 6, [False] * 6
    put(m.cost_table, "eager", 6, q, s, f, 10)
    for n in (2, 4):
        put(m.cost_table, "graph", n, q[:n], s[:n], f[:n], 2)
    assert not plan_partitions(m, q, s, f)  # No free coordination assumption.
    put(m.cost_table, "split_overhead", 6, q, s, f, 1, parts=2)
    parts = plan_partitions(m, q, s, f)
    assert [p.graph_size for p in parts] == [4, 2]
    assert [(p.first_request, p.last_request) for p in parts] == [(0, 4), (4, 6)]
    put(m.cost_table, "split_overhead", 6, q, s, f, 6, parts=2)
    assert not plan_partitions(m, q, s, f)


def test_mixed_requests_not_tokens_and_no_extrapolation():
    m = GraphMatcher([2, 4])
    q, s, f = [3, 1, 1], [8, 128, 128], [True, False, False]
    put(m.cost_table, "eager", 5, q, s, f, 10)
    put(m.cost_table, "graph", 4, q[:2], s[:2], f[:2], 2)
    put(m.cost_table, "graph", 2, q[2:], s[2:], f[2:], 1)
    put(m.cost_table, "split_overhead", 5, q, s, f, 1, parts=2)
    parts = plan_partitions(m, q, s, f)
    assert [p.graph_size for p in parts] == [4, 2]  # One padded token in five.
    assert not plan_partitions(m, q, s, f, max_pad_ratio=.1)
    m.update_captured_sizes([1, 2, 4])
    put(m.cost_table, "graph", 1, q[2:], s[2:], f[2:], 1)
    parts = plan_partitions(m, q, s, f)
    assert parts[0].last_token == 4 and parts[0].last_request == 2
    assert not plan_partitions(m, q, [4096] * 3, f)
    assert not plan_partitions(m, [5], [8], [True])


def test_batch_padding_avoids_an_extra_forward_but_requires_tail_cost():
    m = GraphMatcher([1, 2, 4])
    q, s, f = [1] * 7, [128] * 7, [False] * 7
    put(m.cost_table, 'eager', 7, q, s, f, 20)
    for n in (1, 2, 4):
        put(m.cost_table, 'graph', n, q[:n], s[:n], f[:n], 4)
    for n in (2, 3):
        put(m.cost_table, 'split_overhead', 7, q, s, f, .5, parts=n)
    assert [p.graph_size for p in plan_partitions(m, q, s, f)] == [4, 2, 1]
    put(m.cost_table, 'graph', 4, q[:3], s[:3], f[:3], 4)
    assert [p.graph_size for p in plan_partitions(m, q, s, f)] == [4, 4]
    assert [p.graph_size for p in plan_partitions(m, q, s, f, max_pad_ratio=.1)] == [4, 2, 1]
    put(m.cost_table, 'eager', 7, q, s, f, 8)
    assert not plan_partitions(m, q, s, f)  # Fewer forwards alone do not prove a win.


def test_padding_budget_is_shared_across_all_parts():
    m = GraphMatcher([4])
    q, s, f = [3, 3], [128, 128], [True, True]
    put(m.cost_table, 'eager', 6, q, s, f, 20)
    put(m.cost_table, 'graph', 4, q[:1], s[:1], f[:1], 2)
    put(m.cost_table, 'split_overhead', 6, q, s, f, .5, parts=2)
    assert not plan_partitions(m, q, s, f)  # 2/6, not two independent 1/6 budgets.


def test_qualified_costs_do_not_mix_with_legacy(tmp_path):
    p = tmp_path / "cost.csv"
    p.write_text('phase,mode,execution_shape,mean_gpu_ms,workload,scope,samples,parts\n'
                 'decode,eager,4,2,abc,forward,3,1\n'
                 'decode,graph,4,1,abc,forward,2,1\n'
                 'decode,graph,8,nan,abc,forward,3,1\n')
    t = GraphCostTable(str(p))
    assert t.estimate_workload("eager", 4, "abc") == 2
    assert t.estimate_workload("graph", 4, "abc") is None
    assert t.estimate("decode", "eager", 4) is None


def test_sequential_copy_preserves_outputs_and_restores_inputs(monkeypatch):
    from vllm.v1.worker import cudagraph_adaptation as module
    stats = {}
    @contextmanager
    def context(*a, **kw):
        nonlocal stats
        stats = kw['cudagraph_adaptation_stats']
        yield
    monkeypatch.setattr(module, "set_forward_context", context)
    output = torch.empty(2, 2)
    calls = []
    def model(input_ids, **kwargs):
        calls.append(input_ids.data_ptr())
        output.copy_(input_ids[:, None].expand(-1, 2))
        stats['pieces'] = stats.get('pieces', 0) + 1
        stats['replays'] = stats.get('replays', 0) + 1
        return output
    class Meta:
        def split_for_graph(self, part, *args):
            return self
    r = NS(graphMatcher=GraphMatcher([2]), input_ids=torch.arange(4),
           positions=torch.arange(4), hidden_size=2, dtype=torch.float32,
           device='cpu', vllm_config=None, model=model, compilation_config=ready_registry(2))
    adapter = GraphAdapter(r)
    parts = (GraphPartition(0, 2, 0, 2, 2), GraphPartition(2, 4, 2, 4, 2))
    result = adapter.execute(parts, {'x': Meta()}, [1] * 4, [8] * 4)
    torch.testing.assert_close(result[:, 0], torch.arange(4).float())
    assert calls[0] == calls[1]
    assert torch.equal(r.input_ids, torch.arange(4))
    assert adapter.fallbacks == 0


def test_flash_metadata_partition_preserves_kv_slots():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    m = FlashAttentionMetadata(6, 3, torch.tensor([0, 3, 4, 6]), 128,
        torch.tensor([8, 128, 30]), torch.arange(12).view(3, 4),
        torch.tensor([5, 7, 9, 200, 99, 100]), False, 0, None, None, None)
    part = m.split_for_graph(GraphPartition(1, 3, 3, 6, 4), [3, 1, 2], [8, 128, 30])
    assert part.query_start_loc.tolist() == [0, 1, 3]
    assert part.slot_mapping.tolist() == [200, 99, 100]
    assert part.block_table.tolist() == m.block_table[1:].tolist()
    assert part.scheduler_metadata is None
    assert part.num_actual_tokens == 3
    assert m.query_start_loc.tolist() == [0, 3, 4, 6]
    m.use_cascade = True
    with pytest.raises(ValueError):
        m.split_for_graph(GraphPartition(1, 3, 3, 6, 4), [3, 1, 2], [8, 128, 30])


def test_pressure_limit_has_hysteresis(monkeypatch):
    from vllm.v1.worker import cudagraph_adaptation as module
    r = NS(graphMatcher=GraphMatcher([2]), device='cpu')
    a = GraphAdapter(r)
    total = 100 << 30
    free = [20 << 30]
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free[0], total))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    for available, expected in [(20, .2), (6, 0.), (8, 0.), (10, .1), (13, .1), (16, .2)]:
        free[0] = available << 30
        a._memory_checked = 0
        assert a.pad_limit() == expected


def test_split_observation_is_not_capture_warmup():
    from vllm.v1.worker.adaptive_cudagraph import AdaptiveCUDAGraphController
    c = AdaptiveCUDAGraphController(enabled=True, min_hits=2)
    c.capture_phases = frozenset({'decode', 'prefill', 'mixed'})
    decision = c.plan_runtime('mixed', 6, 8, True, False, adapted=True)
    assert decision.reason == 'split-reuse'
    assert not decision.mark_runtime_warmup
    assert c.snapshot()[0].hit_count == 1
    decision = c.plan_runtime('mixed', 6, 8, True, False, adapted=True)
    assert decision.mark_runtime_warmup
    assert decision.execution_shape == 8
    assert c.snapshot()[0].hit_count == 2
    c.finish_runtime_warmup('mixed', 8)
    decision = c.plan_runtime('mixed', 6, 8, True, False, adapted=True)
    assert decision.reason == 'split-reuse'
    assert not decision.mark_runtime_warmup
    assert c.snapshot()[0].hit_count == 3


@pytest.mark.parametrize('phase', ['decode', 'prefill', 'mixed'])
def test_split_during_capture_then_whole_ready(phase):
    from vllm.v1.worker.adaptive_cudagraph import AdaptiveCUDAGraphController
    c = AdaptiveCUDAGraphController(enabled=True, min_hits=1)
    c.capture_phases = frozenset({'decode', 'prefill', 'mixed'})
    assert c.plan_runtime(phase, 64, 64, True, False, adapted=True).mark_runtime_warmup
    c.finish_runtime_warmup(phase, 64)
    c.mark_capture_started(phase, 64)
    for _ in range(3):
        d = c.plan_runtime(phase, 64, 64, True, False, adapted=True)
        assert d.reason == 'split-reuse' and not d.mark_runtime_warmup
    c.register_ready_shapes([64], phase)
    d = c.plan_runtime(phase, 64, 64, True, True, adapted=True)
    assert d.reason == 'ready-replay' and not d.skip_cudagraph
    assert c.snapshot()[0].hit_count == 5


def test_split_requires_complete_published_chains():
    from threading import Event
    registry = ready_registry(2, 4)
    a = GraphAdapter(NS(compilation_config=registry, graphMatcher=GraphMatcher([2, 4])))
    assert a.ready_sizes() == (2, 4)
    entry = registry.cuda_piecewise_backends_list[0].concrete_size_entries[4]
    entry.concurrent_capture_gate = Event()
    assert a.ready_sizes() == (2,)
    entry.concurrent_capture_gate.set()
    entry.concurrent_capture_in_progress = True
    assert a.ready_sizes() == (2,)
    entry.concurrent_capture_in_progress = False
    assert a.ready_sizes() == (2, 4)
    registry.cuda_piecewise_backends_list[0].total_piecewise_compiles = 2
    assert a.ready_sizes() == ()


def test_lost_graph_declines_before_any_tensor_access():
    r = NS(compilation_config=ready_registry(2), graphMatcher=GraphMatcher([2]))
    a = GraphAdapter(r)
    r.compilation_config.cuda_piecewise_backends_list[0].concrete_size_entries.clear()
    parts = (GraphPartition(0, 2, 0, 2, 2), GraphPartition(2, 4, 2, 4, 2))
    assert a.execute(parts, {}, [1] * 4, [4] * 4) is None
    assert a.declines == 1 and a.fallbacks == 0


def test_filtered_graphs_cannot_use_stale_cost_evidence():
    m = GraphMatcher([2, 4])
    q, s, f = [1] * 6, [128] * 6, [False] * 6
    put(m.cost_table, 'eager', 6, q, s, f, 10)
    for n in (2, 4):
        put(m.cost_table, 'graph', n, q[:n], s[:n], f[:n], 1)
    put(m.cost_table, 'split_overhead', 6, q, s, f, 1, parts=2)
    assert plan_partitions(m, q, s, f)
    assert not plan_partitions(m, q, s, f, captured_sizes=())
    assert not plan_partitions(m, q, s, f, captured_sizes=(2,))


def test_no_retry_after_partial_execution(monkeypatch):
    from vllm.v1.worker import cudagraph_adaptation as module
    from contextlib import nullcontext
    monkeypatch.setattr(module, 'set_forward_context', lambda *a, **kw: nullcontext())
    class Meta:
        def split_for_graph(self, *args):
            return self
    calls = []
    def model(input_ids, **kw):
        calls.append(input_ids.clone())
        if len(calls) == 2:
            raise RuntimeError('device failure')
        return input_ids[:, None].float()
    r = NS(graphMatcher=GraphMatcher([2]), input_ids=torch.arange(4),
           positions=torch.arange(4), hidden_size=1, dtype=torch.float32,
           device='cpu', vllm_config=None, model=model, compilation_config=ready_registry(2))
    a = GraphAdapter(r)
    parts = (GraphPartition(0, 2, 0, 2, 2), GraphPartition(2, 4, 2, 4, 2))
    with pytest.raises(RuntimeError, match='device failure'):
        a.execute(parts, {'x': Meta()}, [1] * 4, [4] * 4)
    assert len(calls) == 2
    assert torch.equal(r.input_ids, torch.arange(4))


def test_allocation_failure_declines_before_kv_write(monkeypatch):
    class Meta:
        def split_for_graph(self, *args):
            return self
    def must_not_run(**kw):
        raise AssertionError('model ran before allocation completed')
    r = NS(graphMatcher=GraphMatcher([2]), input_ids=torch.arange(4),
           positions=torch.arange(4), hidden_size=1, dtype=torch.float32,
           device='cpu', vllm_config=None, model=must_not_run, compilation_config=ready_registry(2))
    def oom(*a, **kw):
        raise torch.OutOfMemoryError('synthetic allocation failure')
    monkeypatch.setattr(torch, 'empty', oom)
    a = GraphAdapter(r)
    parts = (GraphPartition(0, 2, 0, 2, 2), GraphPartition(2, 4, 2, 4, 2))
    assert a.execute(parts, {'x': Meta()}, [1] * 4, [4] * 4) is None
    assert a.fallbacks == 1
    assert torch.equal(r.input_ids, torch.arange(4))


def test_cost_profile_checks_full_model_structure_but_not_model_path():
    from vllm.v1.worker.cudagraph_profile import build_cudagraph_profile_manifest
    config = {'intermediate_size': 8192, '_name_or_path': '/models/first'}
    args = dict(model_config=NS(hf_config=NS(to_dict=lambda: dict(config))),
                cache_config=NS(), scheduler_config=NS(), parallel_config=NS(),
                compilation_config=NS(), device_properties=NS(), dtype='float16',
                torch_version='test', vllm_version='test')
    first = build_cudagraph_profile_manifest(**args)
    config['_name_or_path'] = '/models/relocated'
    assert build_cudagraph_profile_manifest(**args) == first
    config['intermediate_size'] = 16384
    assert build_cudagraph_profile_manifest(**args) != first


def test_small_sample_p90_does_not_round_down_to_median():
    from tools.profile_cudagraph_adaptation import conservative_p90
    assert conservative_p90([1., 2., 10.]) == 10.
    assert conservative_p90(list(range(1, 11))) == 9


@pytest.mark.parametrize('state', ['missing', 'capturing', 'ready'])
def test_split_piece_uses_only_ready_graphs(monkeypatch, state):
    from vllm.compilation import cuda_piecewise_backend as module
    backend = object.__new__(module.CUDAPiecewiseBackend)
    backend.first_run_finished = True
    backend.sym_shape_indices = [0]
    backend.piecewise_compile_index = 0
    backend.is_debugging_mode = False
    output = torch.zeros(4, 2)
    replayed = []
    entry = module.ConcreteSizeEntry(4, False, True, runnable=lambda *a: output)
    entry.cudagraph = None if state == 'missing' else NS(replay=lambda: replayed.append(4))
    entry.concurrent_capture_in_progress = state == 'capturing'
    entry.output = output
    backend.concrete_size_entries = {4: entry}
    backend._run = lambda mode, runnable, args: runnable(*args)
    context = NS(skip_cuda_graphs=False, concurrent_cudagraph_capture=False,
                 cudagraph_runtime_warmup=False, cudagraph_adaptation_stats={})
    monkeypatch.setattr(module, 'get_forward_context', lambda: context)
    assert backend(4, output) is output
    assert replayed == ([4] if state == 'ready' else [])
    assert context.skip_cuda_graphs == (state != 'ready')
    assert context.cudagraph_adaptation_stats['pieces'] == 1
    assert context.cudagraph_adaptation_stats.get('replays', 0) == (state == 'ready')


def test_unregistered_split_shape_disables_suffix(monkeypatch):
    from vllm.compilation import cuda_piecewise_backend as module
    backend = object.__new__(module.CUDAPiecewiseBackend)
    backend.first_run_finished = True
    backend.sym_shape_indices = [0]
    backend.piecewise_compile_index = 0
    backend.concrete_size_entries = {}
    output = torch.zeros(3, 2)
    backend.compiled_graph_for_general_shape = lambda *a: output
    backend._run = lambda mode, runnable, args: runnable(*args)
    context = NS(skip_cuda_graphs=False, cudagraph_adaptation_stats={})
    monkeypatch.setattr(module, 'get_forward_context', lambda: context)
    assert backend(3, output) is output
    assert context.skip_cuda_graphs
    assert context.cudagraph_adaptation_stats == {'pieces': 1}
