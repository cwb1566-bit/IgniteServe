# SPDX-License-Identifier: Apache-2.0
"""Single-host startup acceptance and asset-cold timing on explicitly free GPUs."""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from igniteserve_startup.contract import identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cold-timing', action='store_true',
                        help='Compare DAG off/on with separate empty caches; keep OS page cache unchanged')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    gpus = args.gpus.split(',')
    if len(gpus) not in (2, 4) or len(set(gpus)) != len(gpus):
        parser.error('Two or four distinct free GPUs are required')
    contract = root / 'vllm/v1/engine/startup_contract.distributed.json'
    payload = json.loads(contract.read_text())
    contract_id = identity(payload)
    results = []
    with tempfile.TemporaryDirectory(prefix='distributed-startup-') as tmp:
        directory = Path(tmp)
        for tp, pp in ((len(gpus), 1), (1, len(gpus))):
            baseline = None
            cases = [('off', True, 1)] + [
                ('execute', prewarm, workers)
                for prewarm in (True, False) for workers in (1, 4)]
            if args.cold_timing:
                cases = [('off', True, 1), ('execute', True, 4)]
            for mode, prewarm, workers in cases:
                for gpu in gpus:
                    active = subprocess.check_output([
                        'nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                        '--format=csv,noheader,nounits'], text=True).strip()
                    if active:
                        raise RuntimeError(f'GPU {gpu} is occupied: {active}')
                name = f'tp{tp}-pp{pp}-{mode}-prewarm{int(prewarm)}-workers{workers}'
                tokens, report = directory / f'{name}.json', directory / f'{name}.dag.json'
                env = dict(os.environ)
                env.update(PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=args.gpus,
                    VLLM_WORKER_MULTIPROC_METHOD='spawn',
                    VLLM_ENABLE_STARTUP_OPTIMIZATIONS='1',
                    VLLM_STARTUP_DAG_MODE=mode,
                    VLLM_STARTUP_DAG_MAX_WORKERS=str(workers),
                    VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM='1' if prewarm else '0',
                    VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION='1',
                    VLLM_STARTUP_CONTRACT_PATH=str(contract),
                    VLLM_STARTUP_GENERATED_REPORT=str(report),
                    VLLM_STARTUP_RESOURCE_ROOT=str(directory / f'assets-{tp}-{pp}'),
                    VLLM_ENABLE_ADAPTIVE_CUDAGRAPH='0',
                    TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS='2')
                timing = directory / f'{name}.timing.json'
                if args.cold_timing:
                    cache = directory / name
                    env.update(VLLM_STARTUP_RESOURCE_ROOT=str(cache / 'assets'),
                        VLLM_CACHE_ROOT=str(cache / 'vllm'),
                        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
                        TRITON_CACHE_DIR=str(cache / 'triton'),
                        CUDA_CACHE_PATH=str(cache / 'cuda'),
                        XDG_CACHE_HOME=str(cache / 'xdg'),
                        VLLM_ENABLE_DISTRIBUTED_BACKGROUND_CAPTURE='1',
                        VLLM_ENABLE_ADAPTIVE_CUDAGRAPH='1',
                        VLLM_MIXED_CAPTURE_SHADOW_VALIDATE='0',
                        VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE='1')
                command = [sys.executable, str(root / 'tests/startup_runtime/smoke_declarative_startup.py'),
                           '--model', args.model, '--output', str(tokens), '--tp', str(tp), '--pp', str(pp)]
                if args.eager:
                    command.append('--eager')
                if args.cold_timing:
                    command.extend(['--timing-output', str(timing), '--bounded-capture',
                                    '--compile-cache-dir', str(cache / 'compile')])
                print(f'START {name}', flush=True)
                with (directory / f'{name}.log').open('w+') as log:
                    proc = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = proc.wait(timeout=900)
                    finally:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=20)
                    if code:
                        log.seek(0)
                        raise RuntimeError(log.read()[-24000:])
                    log.seek(0)
                    log_text = log.read()
                value = json.loads(tokens.read_text())
                if mode == 'off':
                    baseline = value
                else:
                    assert value == baseline, name
                    dag = json.loads(report.read_text())
                    assert dag['execution_kind'] == 'generated_adapter'
                    assert dag['identity'] == contract_id
                    assert len(dag['results']) == 26
                    assert all(r['status'] == 'completed' for r in dag['results'].values())
                    assert len(dag['released_assets']) == tp * pp
                    results.append({'tp': tp, 'pp': pp, 'nodes': 26, 'token_match': True,
                                    'identity': dag['identity'], 'eager': args.eager,
                                    'prewarm': prewarm, 'workers': workers})
                if args.cold_timing:
                    stages = {}
                    for stage, seconds in re.findall(r'\[TIMER\] Finished: (.*?) \(([0-9.]+) s\)', log_text):
                        stages.setdefault(stage, []).append(float(seconds))
                    row = dict(tp=tp, pp=pp, mode=mode, **json.loads(timing.read_text()),
                               stages_s=stages, token_match=value == baseline)
                    if mode == 'off':
                        results.append(row)
                    else:
                        results[-1].update(row)
                        results[-1]['dag_results'] = dag['results']
                    print('COLD_TIMING ' + json.dumps(row), flush=True)
                print(f'PASS {name}', flush=True)
    assert identity(json.loads(contract.read_text())) == contract_id
    evidence = dict(identity=contract_id, serial_equivalence=not args.cold_timing,
                    scheduled_equivalence=True,
                    variants=([v['name'] for v in payload['variants']] if not args.cold_timing else []),
                    model=args.model, scope=f'single-host TP{len(gpus)} or PP{len(gpus)}; NCCL; greedy token equality',
                    cache_state='empty artifact/compiler caches; OS page cache uncontrolled' if args.cold_timing else 'shared per topology',
                    baseline='same IgniteServe runtime with DAG off, not native vLLM',
                    cases=results)
    Path(args.output).write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
