# SPDX-License-Identifier: Apache-2.0
"""Isolated-process acceptance of a reviewed vLLM candidate (one free GPU)."""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from igniteserve_startup.contract import identity, variant_nodes


def validate(root, model, gpu, output, contract=None):
    root = Path(root).resolve()
    contract = Path(contract).resolve() if contract else root / "vllm/v1/engine/startup_contract.json"
    adapter = root / "vllm/v1/engine/vllm_startup_adapter.py"
    payload = json.loads(contract.read_text())
    contract_id = identity(payload)
    summaries = []
    with tempfile.TemporaryDirectory(prefix="startup-acceptance-") as temporary:
        directory = Path(temporary)
        cases = [("baseline", "off", 1, True, "assets")]
        for prewarm in (True, False):
            for workers in (1, 4):
                cases.append((f"{'prewarm' if prewarm else 'standard'}-{workers}",
                              "execute", workers, prewarm, "assets"))
        cases.append(("cold", "execute", 4, True, "cold-assets"))
        baseline = None
        for name, mode, workers, prewarm, assets in cases:
            report = directory / f"{name}.report.json"
            tokens = directory / f"{name}.json"
            env = dict(os.environ)
            for key in ("VLLM_STARTUP_CONTRACT_PATH", "VLLM_STARTUP_ADAPTER_PATH"):
                env.pop(key, None)
            env.update(PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=str(gpu),
                VLLM_ENABLE_STARTUP_OPTIMIZATIONS="1", VLLM_STARTUP_DAG_MODE=mode,
                VLLM_STARTUP_DAG_MAX_WORKERS=str(workers),
                VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM="1" if prewarm else "0",
                VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION="1",
                VLLM_STARTUP_GENERATED_REPORT=str(report),
                VLLM_STARTUP_RESOURCE_ROOT=str(directory / assets),
                VLLM_WORKER_MULTIPROC_METHOD="spawn")
            env["VLLM_STARTUP_CONTRACT_PATH"] = str(contract)
            env["VLLM_STARTUP_ADAPTER_PATH"] = str(adapter)
            command = [sys.executable, str(root / "tests/startup_runtime/smoke_declarative_startup.py"),
                       "--model", model, "--output", str(tokens)]
            print(f"START {name}", flush=True)
            with (directory / f"{name}.log").open("w+") as log:
                proc = subprocess.Popen(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=600)
                finally:
                    # Only the process group created for this test is owned here.
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                if code:
                    log.seek(0)
                    raise RuntimeError(log.read()[-18000:])
            value = json.loads(tokens.read_text())
            if name == "baseline":
                baseline = value
            elif value != baseline:
                raise AssertionError(f"Token outputs differ: {name}")
            if mode == "execute":
                result = json.loads(report.read_text())
                expected = {n["name"] for n in variant_nodes(payload, payload["variants"][0])}
                assert result["identity"] == contract_id
                assert result["execution_kind"] == "generated_adapter"
                assert set(result["results"]) == expected
                assert all(v["status"] == "completed" for v in result["results"].values())
                intervals = list(result["results"].values())
                overlap = any(a["started_at"] < b["finished_at"]
                              and b["started_at"] < a["finished_at"]
                              for i, a in enumerate(intervals) for b in intervals[i+1:])
                if workers == 1:
                    assert not overlap
                elif not overlap:
                    raise AssertionError("Scheduled test did not exercise overlap")
                summaries.append(dict(case=name, nodes=len(expected), overlap=overlap,
                                      token_match=True,
                                      elapsed=max(v["finished_at"] for v in intervals)
                                      - min(v["started_at"] for v in intervals)))
            print(f"PASS {name}", flush=True)
        # Never finalize evidence across a source/contract change during testing.
        assert identity(json.loads(contract.read_text())) == contract_id
    evidence = dict(identity=contract_id, serial_equivalence=True,
                    scheduled_equivalence=True, variants=[v["name"] for v in payload["variants"]],
                    model=model, scope="single GPU EngineCore; greedy token equivalence",
                    cases=summaries)
    Path(output).write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract")
    validate(**vars(parser.parse_args()))
