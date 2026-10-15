from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
import queue
import threading

import pytest
import torch

from vllm.v1.worker import composite_cudagraph as module
from vllm.v1.worker.GraphMatcher import GraphMatcher
from vllm.v1.worker.cudagraph_adaptation import (
    GraphPartition, composite_workload, plan_partitions, workload_key)


def test_composite_cost_is_measured_whole_execution_not_sum_or_max():
    matcher = GraphMatcher([1, 2, 4])
    q, s, f = [1]*7, [12]*7, [False]*7
    parts = (GraphPartition(0,4,0,4,4), GraphPartition(4,7,4,7,4))
    prices = matcher.cost_table._workloads
    prices['eager',7,workload_key(q,s,f),1] = 20.
    for n in (3,4):
        prices['graph',4,workload_key(q[:n],s[:n],f[:n]),1] = 1.
    assert not plan_partitions(matcher,q,s,f,composite=True)
    key = ('composite',7,composite_workload(parts,q,s,f),2)
    prices[key] = 12.
    assert plan_partitions(matcher,q,s,f,composite=True) == parts
    assert not plan_partitions(matcher,q,s,f,composite=True,captured_sizes=[1,2])
    prices[key] = 19.
    assert not plan_partitions(matcher,q,s,f,composite=True)


@dataclass
class Metadata:
    max_seq_len: int
    table: torch.Tensor

    def split_for_graph(self, part, q, s):
        return replace(self, max_seq_len=max(s[part.first_request:part.last_request]),
                       table=self.table[part.first_request:part.last_request])


def test_metadata_contract_covers_shape_stride_scalars_not_contents():
    a = Metadata(16, torch.zeros((2, 3)))
    assert module.metadata_signature(a) == module.metadata_signature(replace(a, table=torch.ones(2,3)))
    assert module.metadata_signature(a) != module.metadata_signature(replace(a, max_seq_len=32))
    assert module.metadata_signature(a) != module.metadata_signature(replace(a, table=torch.zeros(3,2).t()))


def setup_registry(monkeypatch):
    jobs = []
    r = NS(device='cpu', model_config=NS(max_model_len=128), model=object(),
           input_ids=torch.zeros(4), positions=torch.zeros(4), kv_caches=[torch.zeros(4)],
           concurrent_piecewise_cudagraph_manager=NS(enabled=True,
               enqueue_auxiliary=lambda job: jobs.append(job) or True))
    class FakeGraph:
        def __init__(self, runner, parts, metadata):
            self.ready = threading.Event()
            self.failed = False
            self.staged = NS(record=lambda stream: None)
            self.stages = self.replays = 0
        def stage(self, lanes):
            self.stages += 1
        def capture(self):
            self.ready.set()
        def replay(self, lanes):
            self.replays += 1
            return 'output'
    monkeypatch.setattr(module, 'CompositeGraph', FakeGraph)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: None)
    return module.CompositeGraphs(r), jobs


def test_pending_snapshot_is_immutable_and_publication_is_required(monkeypatch):
    registry, jobs = setup_registry(monkeypatch)
    parts = (GraphPartition(0,2,0,2,2), GraphPartition(2,4,2,4,2))
    meta = {'layer': Metadata(7, torch.zeros(4,3))}
    call = lambda s: registry.execute(parts, meta, [1]*4, [s]*4)
    assert call(7) is None
    plan = next(iter(registry.plans.values()))
    assert call(8) is None  # Same context bucket, no overwriting capture inputs.
    assert plan.stages == 1 and len(jobs) == 1
    jobs[0]()
    assert call(8) == 'output' and registry.replays == 1
    assert call(9) is None  # New scalar FA contract needs a different graph.
    assert len(jobs) == 2


def test_failed_or_disabled_builds_do_not_retry_or_launch(monkeypatch):
    registry, jobs = setup_registry(monkeypatch)
    parts = (GraphPartition(0,1,0,1,1), GraphPartition(1,2,1,2,1))
    metadata = {'x': Metadata(7, torch.zeros(2,1))}
    assert registry.execute(parts, metadata, [1,1], [7,7]) is None
    plan = next(iter(registry.plans.values()))
    plan.failed = True
    assert registry.execute(parts, metadata, [1,1], [7,7]) is None
    assert len(jobs) == 1 and registry.replays == 0
    registry.runner.concurrent_piecewise_cudagraph_manager.enabled = False
    assert registry.execute(parts, metadata, [1,1], [9,9]) is None
    assert len(jobs) == 1


def test_no_whole_batch_retry_after_replay_error(monkeypatch):
    registry, jobs = setup_registry(monkeypatch)
    parts = (GraphPartition(0,1,0,1,1), GraphPartition(1,2,1,2,1))
    meta = {'x': Metadata(7, torch.zeros(2,1))}
    registry.execute(parts, meta, [1,1], [7,7])
    jobs[0]()
    plan = next(iter(registry.plans.values()))
    def fail(_):
        raise RuntimeError('KV may already be written')
    plan.replay = fail
    with pytest.raises(RuntimeError, match='KV'):
        registry.execute(parts, meta, [1,1], [7,7])


def test_bounded_residency(monkeypatch):
    registry, jobs = setup_registry(monkeypatch)
    parts = (GraphPartition(0,1,0,1,1), GraphPartition(1,2,1,2,1))
    meta = {'x': Metadata(1, torch.zeros(2,1))}
    for length in (1,2,4,8,16):
        assert registry.execute(parts, meta, [1,1], [length,length]) is None
    assert len(registry.plans) == len(jobs) == 4


def test_model_or_kv_rebind_declines_before_replay(monkeypatch):
    registry, jobs = setup_registry(monkeypatch)
    parts = (GraphPartition(0,1,0,1,1), GraphPartition(1,2,1,2,1))
    meta = {'x': Metadata(7, torch.zeros(2,1))}
    registry.execute(parts, meta, [1,1], [7,7])
    jobs[0]()
    registry.runner.kv_caches = [torch.zeros(4)]
    assert registry.execute(parts, meta, [1,1], [7,7]) is None
    assert registry.replays == 0


def test_auxiliary_and_whole_capture_share_serial_worker(monkeypatch):
    from vllm.v1.worker.concurrent_piecewise_cudagraph import (
        ConcurrentPiecewiseCUDAGraphCaptureManager as Manager, _CaptureRequest)
    manager = object.__new__(Manager)
    manager.runner = NS(device='cpu')
    manager._queue = queue.Queue()
    manager._lock = threading.Lock()
    manager._scheduled_shapes = {8}
    order = []
    manager._capture_one = lambda request: order.append(request.shape)
    manager._queue.put(lambda: order.append('composite'))
    manager._queue.put(_CaptureRequest('decode',8,None,0))
    manager._queue.put(None)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    manager._worker_main()
    assert order == ['composite',8] and not manager._scheduled_shapes


def test_shutdown_prevents_new_capture_thread():
    from vllm.v1.worker.concurrent_piecewise_cudagraph import ConcurrentPiecewiseCUDAGraphCaptureManager
    manager = object.__new__(ConcurrentPiecewiseCUDAGraphCaptureManager)
    manager._stopping = True
    manager._lock = threading.Lock()
    manager._thread = None
    manager.enabled = True
    assert not manager.enqueue_auxiliary(lambda: None)
    assert manager._thread is None
