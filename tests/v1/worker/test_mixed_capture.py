import ast
import os
from pathlib import Path
import torch
from types import SimpleNamespace
from unittest.mock import patch
from vllm.v1.worker.adaptive_cudagraph import AdaptiveCUDAGraphController, GraphState


def test_branch_defaults_and_explicit_opt_out():
    from vllm.v1.worker.concurrent_piecewise_cudagraph import (
        ConcurrentPiecewiseCUDAGraphCaptureManager as Manager)
    runner = SimpleNamespace(enable_adaptive_cudagraph=False)
    with patch.dict(os.environ, {}, clear=True):
        c = AdaptiveCUDAGraphController()
        assert c.enabled and c.min_hits == 1
        assert Manager._check_support(runner) == (
            False, 'adaptive CUDA graph controller is disabled')
    with patch.dict(os.environ, {
        'VLLM_ENABLE_ADAPTIVE_CUDAGRAPH': '0',
        'VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS': '5',
        'VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE': '0',
    }, clear=True):
        c = AdaptiveCUDAGraphController()
        assert not c.enabled and c.min_hits == 5
        assert Manager._check_support(runner) == (False, 'feature flag is off')


def test_mixed_and_prefill_lifecycle():
    for phase in ['mixed', 'prefill']:
        c = AdaptiveCUDAGraphController(enabled=True, min_hits=2)
        c.capture_phases = frozenset({'decode', 'mixed', 'prefill'})
        assert not c.plan_runtime(phase, 31, 32, True, False).mark_runtime_warmup
        assert c.plan_runtime(phase, 31, 32, True, False).mark_runtime_warmup
        c.finish_runtime_warmup(phase, 32)
        assert c.take_queued_shapes(phase) == [32]
        c.mark_capture_started(phase, 32)
        c.mark_capture_ready(phase, 32, .1)
        assert c.plan_runtime(phase, 31, 32, True, True).reason == 'ready-replay'
        assert c.snapshot()[0].capture_attempts == 1


def test_deduplicated_completion_and_failure():
    for fail in [False, True]:
        c = AdaptiveCUDAGraphController(enabled=True, min_hits=1)
        c.capture_phases = frozenset({'decode', 'mixed', 'prefill'})
        for phase in c.capture_phases:
            c.plan_runtime(phase, 31, 32, True, False)
            c.finish_runtime_warmup(phase, 32)
        assert c.take_queued_shapes('mixed') == [32]
        error = RuntimeError('injected') if fail else None
        if fail:
            c.mark_capture_failed('mixed', 32, error)
        else:
            c.mark_capture_ready('mixed', 32, .1)
        c.settle_shared_capture('mixed', 32, error)
        expected = GraphState.FAILED if fail else GraphState.READY
        assert all(s.state == expected for s in c.snapshot())
        assert not c.pending_shapes('prefill')


def test_signature_layout_alias_and_constants():
    source = (Path(__file__).resolve().parents[3]
              / 'vllm/compilation/cuda_piecewise_backend.py')
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_input_signature')
    scope = {'torch': torch, 'Any': object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    sig = scope['_input_signature']
    x, y = torch.zeros(4,4), torch.ones(4,4)
    assert sig((x, 3)) == sig((y, 3))
    assert sig((x, 3)) != sig((y.t(), 3))
    assert sig((x, x)) != sig((x, y))
    assert sig((x, 3)) != sig((x, 4))


def test_backend_guard_disables_suffix_and_abort_clears_contract():
    from vllm.compilation import cuda_piecewise_backend as module
    b = object.__new__(module.CUDAPiecewiseBackend)
    b.first_run_finished = True
    b.sym_shape_indices = [0]
    b.piecewise_compile_index = 0
    x = torch.zeros(4,4)
    entry = module.ConcreteSizeEntry(4, False, True, runnable=lambda *args: x)
    entry.concurrent_signature = ('invalid',)
    entry.cudagraph = SimpleNamespace(reset=lambda: None)
    b.concrete_size_entries = {4: entry}
    b._run = lambda mode, runnable, args: runnable(*args)
    ctx = SimpleNamespace(skip_cuda_graphs=False, concurrent_cudagraph_capture=False)
    with patch.object(module, 'get_forward_context', return_value=ctx):
        assert b(4, x) is x
    assert ctx.skip_cuda_graphs
    entry.concurrent_capture_in_progress = True
    b.abort_concurrent_capture(4)
    assert entry.cudagraph is None and entry.concurrent_signature is None


if __name__ == '__main__':
    test_branch_defaults_and_explicit_opt_out()
    test_mixed_and_prefill_lifecycle()
    test_deduplicated_completion_and_failure()
    test_signature_layout_alias_and_constants()
    test_backend_guard_disables_suffix_and_abort_clears_contract()
    print('5 mixed capture unit tests passed')
