# SPDX-License-Identifier: Apache-2.0
"""Exercise the merged EngineCore dispatch without constructing GPU workers."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.startup_analysis import generated, recorder
from vllm.v1.engine.startup_analysis import source_instrumentation
from vllm.v1.engine.startup_analysis.extractor import StartupActionExtractor


SOURCE = (Path(__file__).resolve().parents[3]
          / 'vllm/v1/engine/core.py')


def _engine_class(*methods):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == 'EngineCore')
    engine.body = [n for n in engine.body if isinstance(n, ast.FunctionDef)
                   and n.name in methods]
    future = ast.ImportFrom(module='__future__',
                            names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(
        body=[future, engine], type_ignores=[]))
    scope = {'os': os, 'Path': Path, '__file__': str(SOURCE),
             'logger': MagicMock(), 'timer': MagicMock()}
    exec(compile(module, str(SOURCE), 'exec'), scope)
    return scope['EngineCore']


@pytest.mark.parametrize('mode', ['off', 'shadow', 'execute'])
def test_existing_modes_delegate_without_generated_preflight(monkeypatch, mode):
    monkeypatch.setenv('VLLM_STARTUP_DAG_MODE', mode)
    engine = object.__new__(_engine_class('__init__'))
    engine._initialize_engine_core = MagicMock()
    prepare = MagicMock(side_effect=AssertionError('unexpected preflight'))
    monkeypatch.setattr(generated, 'prepare_generated_startup', prepare)
    engine.__init__('config', 'executor', False)
    engine._initialize_engine_core.assert_called_once_with(
        'config', 'executor', False, None)
    prepare.assert_not_called()


def test_discovery_wraps_sequential_entry_and_flushes(monkeypatch):
    monkeypatch.setenv('VLLM_STARTUP_DAG_MODE', 'discover')
    monkeypatch.setenv('VLLM_STARTUP_DISCOVERY_STAGE_MAP', '/tmp/stages.json')
    engine = object.__new__(_engine_class('__init__'))
    engine._initialize_engine_core = MagicMock()
    analysis = MagicMock()
    instrumented = MagicMock()
    wrap = MagicMock(return_value=instrumented)
    monkeypatch.setattr(recorder, 'get_startup_recorder', lambda: analysis)
    monkeypatch.setattr(source_instrumentation,
                        'build_instrumented_bound_method', wrap)
    monkeypatch.setattr(generated, 'runtime_compatibility_facts',
                        lambda *args, **kwargs: {'cache_state': 'D0'})
    engine.__init__('config', 'executor', False)
    wrap.assert_called_once_with(engine._initialize_engine_core,
                                 Path('/tmp/stages.json'), analysis)
    instrumented.assert_called_once_with('config', 'executor', False, None)
    engine._initialize_engine_core.assert_not_called()
    analysis.stage.assert_called_once_with('engine_startup.root')
    analysis.record.assert_called_once_with(
        'startup_compatibility', facts={'cache_state': 'D0'})
    analysis.flush.assert_called_once()


def test_discovery_failure_flushes_without_retry(monkeypatch):
    monkeypatch.setenv('VLLM_STARTUP_DAG_MODE', 'discover')
    monkeypatch.delenv('VLLM_STARTUP_DISCOVERY_STAGE_MAP', raising=False)
    engine = object.__new__(_engine_class('__init__'))
    engine._initialize_engine_core = MagicMock(side_effect=RuntimeError('action'))
    analysis = MagicMock()
    monkeypatch.setattr(recorder, 'get_startup_recorder', lambda: analysis)
    with pytest.raises(RuntimeError, match='action'):
        engine.__init__('config', 'executor', False)
    engine._initialize_engine_core.assert_called_once()
    analysis.flush.assert_called_once()


@pytest.mark.parametrize('contract', [None, '/tmp/contract.json'])
def test_generated_preflight_rejection_falls_back_once(monkeypatch, contract):
    monkeypatch.setenv('VLLM_STARTUP_DAG_MODE', 'execute-generated')
    if contract:
        monkeypatch.setenv('VLLM_STARTUP_CONTRACT_PATH', contract)
    else:
        monkeypatch.delenv('VLLM_STARTUP_CONTRACT_PATH', raising=False)
    engine = object.__new__(_engine_class('__init__'))
    engine._initialize_engine_core = MagicMock()
    prepare = MagicMock(side_effect=generated.GeneratedStartupRejected('hash'))
    monkeypatch.setattr(generated, 'prepare_generated_startup', prepare)
    engine.__init__('config', 'executor', False)
    engine._initialize_engine_core.assert_called_once()
    assert prepare.call_count == bool(contract)


@pytest.mark.parametrize('fails', [False, True])
def test_generated_actions_never_restart_sequentially(monkeypatch, fails):
    monkeypatch.setenv('VLLM_STARTUP_DAG_MODE', 'execute-generated')
    monkeypatch.setenv('VLLM_STARTUP_CONTRACT_PATH', '/tmp/contract.json')
    engine = object.__new__(_engine_class('__init__'))
    engine._initialize_engine_core = MagicMock()
    plan = SimpleNamespace(
        contract_path=Path('/tmp/contract.json'),
        adapter_path=Path('/tmp/adapter.py'), variant={'name': 'D2'},
        execute=MagicMock(side_effect=RuntimeError('action') if fails else None))
    monkeypatch.setattr(generated, 'prepare_generated_startup',
                        MagicMock(return_value=plan))
    if fails:
        with pytest.raises(RuntimeError, match='action'):
            engine.__init__('config', 'executor', False)
    else:
        engine.__init__('config', 'executor', False)
    plan.execute.assert_called_once()
    engine._initialize_engine_core.assert_not_called()


def test_deferred_execute_still_reaches_full_startup_dag(monkeypatch):
    monkeypatch.setenv('VLLM_STARTUP_DAG_MODE', 'execute')
    engine = object.__new__(_engine_class('_initialize_engine_core'))
    engine._initialize_with_full_startup_dag = MagicMock()
    executor = SimpleNamespace(supports_deferred_startup=lambda: True)
    engine._initialize_engine_core('config', executor, False)
    engine._initialize_with_full_startup_dag.assert_called_once_with(
        'config', executor, False, None)


def test_manual_dag_flushes_analysis_before_returning_overlap_report(monkeypatch):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == 'EngineCore')
    method = next(n for n in engine.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_initialize_with_startup_dag')
    tail_start = next(i for i, n in enumerate(method.body)
                      if isinstance(n, ast.ImportFrom)
                      and n.module == 'vllm.v1.engine.startup_analysis.recorder')
    tail = ast.FunctionDef(name='finish',
                          args=ast.arguments(posonlyargs=[], args=[],
                                             kwonlyargs=[], kw_defaults=[],
                                             defaults=[]),
                          body=method.body[tail_start:], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[]))
    analysis = MagicMock()
    monkeypatch.setattr(recorder, 'get_startup_recorder', lambda: analysis)
    scope = {
        'results': {'compile': SimpleNamespace(elapsed_s=1.25)},
        'context': SimpleNamespace(get=lambda *args: ['worker-report']),
        'overlap_weight_compile': True, 'plan_path': Path('/tmp/plan.json'),
    }
    exec(compile(module, str(SOURCE), 'exec'), scope)
    result = scope['finish']()
    analysis.flush.assert_called_once()
    assert result == {'stage_seconds': {'compile': 1.25},
                      'weight_compile_worker_results': ['worker-report'],
                      'weight_compile_overlap': True,
                      'plan_path': '/tmp/plan.json'}


def test_merged_source_extracts_without_existing_stage_evidence(tmp_path):
    discovery = tmp_path / 'discovery.json'
    discovery.write_text(json.dumps({
        'schema': 'igniteserve-startup-discovery-v1',
        'call_boundaries': [], 'sampled_call_boundaries': [],
    }), encoding='utf-8')
    outputs = StartupActionExtractor(SOURCE, discovery).generate(
        tmp_path / 'candidate')
    contract = json.loads(outputs['contract'].read_text(encoding='utf-8'))
    assert contract['status'] != 'final'
    nodes = contract['variants'][0]['nodes']
    assert nodes and any(not node['executable'] for node in nodes)
    assert all(node['action_id'] for node in nodes)
    review = json.loads((outputs['review'] / 'action_bindings.json').read_text(
        encoding='utf-8'))
    assert review['items']
