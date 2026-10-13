# SPDX-License-Identifier: Apache-2.0
"""Offline, reviewed vLLM recipe. Never imported by online startup.

Boundaries use the existing deferred executor API after source review. This
is not a claim that arbitrary Python constructors can be split automatically.
"""
import argparse
import json
from pathlib import Path

from igniteserve_startup.contract import digest
from igniteserve_startup.offline import analyze


def build(root, output):
    root, output = Path(root), Path(output)
    directory = root / "vllm/v1/engine"
    discovery = analyze(directory / "core.py", "EngineCore", "_initialize_engine_core")
    # Reviewed state producers; dependencies refer to readiness, not call order.
    recipe = [
        ("engine_core_construct", [], {"cpu": 1}, .01),
        ("plugins", ["engine_core_construct"], {"cpu": 1}, .1),
        ("executor_shell", ["engine_core_construct"], {"cpu": 1}, .01),
        ("worker_wrapper", ["executor_shell"], {"cpu": 1}, .05),
        ("worker_prepare", ["worker_wrapper", "plugins"], {"cpu": 1}, .1),
        ("worker_construct", ["worker_prepare"], {"cpu": 1}, .2),
        ("static_resource_probe", ["worker_construct"], {"worker_rpc": 1}, .05),
        ("compile_worker_prewarm", ["worker_construct"], {"worker_rpc": 1, "compiler_cpu": 1}, .2),
        ("cuda_device", ["compile_worker_prewarm"], {"worker_rpc": 1, "gpu_runtime": 1}, .8),
        ("distributed_environment", ["cuda_device"], {"worker_rpc": 1, "gpu_runtime": 1}, .5),
        ("model_runner", ["compile_worker_prewarm", "distributed_environment"], {"worker_rpc": 1, "gpu_memory": 1}, .8),
        ("usage_reporting", ["model_runner"], {"worker_rpc": 1}, .01),
        ("state_probe", ["usage_reporting", "static_resource_probe"], {"worker_rpc": 1}, .05),
        ("init_model", ["state_probe"], {"worker_rpc": 1}, 1.),
        ("stop_weight_prewarm", ["init_model"], {}, .01),
        ("load_weights", ["stop_weight_prewarm"], {"worker_rpc": 1, "disk_io": 1}, 10.),
        ("compile_model", ["load_weights"], {"worker_rpc": 1, "compiler_cpu": 1}, 10.),
        ("kv_profile", ["compile_model", "load_weights"], {"worker_rpc": 1, "gpu_compute": 1}, 2.),
        ("kv_cache_allocate", ["kv_profile", "load_weights"], {"worker_rpc": 1, "gpu_memory": 1}, 1.),
        ("cuda_graph_precapture", ["compile_model", "kv_cache_allocate"], {"worker_rpc": 1, "gpu_compute": 1, "gpu_memory": 1}, 3.),
        ("structured_output", ["init_model"], {"cpu": 1}, .2),
        ("mm_input_cache", ["init_model"], {"cpu": 1}, .1),
        ("scheduler", ["structured_output", "kv_cache_allocate"], {"cpu": 1}, .2),
        ("batch_queue", ["scheduler"], {"cpu": 1}, .01),
        ("ready", ["cuda_graph_precapture", "batch_queue", "mm_input_cache"], {}, 0.),
    ]
    source_names = [
        "vllm/v1/engine/core.py", "vllm/v1/engine/declarative_startup.py",
        "vllm/v1/worker/gpu_worker.py", "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/executor/abstract.py", "vllm/executor/uniproc_executor.py",
        "vllm/v1/executor/multiproc_executor.py", "vllm/worker/worker_base.py",
        "igniteserve_startup/runtime.py", "igniteserve_startup/contract.py",
    ]
    variants = []
    for prewarm in (True, False):
        nodes = []
        for name, deps, resources, cost in recipe:
            if not prewarm:
                if name == "compile_worker_prewarm": deps = ["distributed_environment"]
                if name == "cuda_device": deps = ["worker_construct"]
            inputs = ["ready." + dep for dep in deps]
            outputs = ["ready." + name]
            if name == "static_resource_probe": outputs += ["static_worker_states"]
            if name == "state_probe":
                inputs += ["static_worker_states"]
                outputs += ["startup_state"]
            if name == "kv_profile": outputs += ["available_gpu_memory"]
            if name == "kv_cache_allocate":
                inputs += ["available_gpu_memory"]
                outputs += ["kv_cache_config"]
            if name == "scheduler": inputs += ["kv_cache_config"]
            nodes.append(dict(name=name, action_id=name, dependencies=[],
                required_state=inputs, produced_state=outputs, resources=resources,
                estimated_cost_s=cost,
                thread_affinity="worker-0" if "worker_rpc" in resources else None))
        variants.append(dict(name="prewarm" if prewarm else "standard",
                             match={"two_phase_prewarm": prewarm}, nodes=nodes))
    common = variants[0].pop("nodes")
    alternate = variants[1].pop("nodes")
    variants[1]["overrides"] = {
        alt["name"]: {"required_state": alt["required_state"]}
        for base, alt in zip(common, alternate)
        if base["required_state"] != alt["required_state"]}
    payload = dict(schema="igniteserve-startup-v3", status="candidate", nodes=common,
        compatibility={"engine": "vllm-v1-deferred"}, initial_state=[],
        adapter_sha256=digest(directory / "vllm_startup_adapter.py"),
        sources={name: digest(root / name) for name in source_names},
        resource_capacities=dict(cpu=2, worker_rpc=2, compiler_cpu=1,
                                 disk_io=1, gpu_compute=1, gpu_memory=1, gpu_runtime=1),
        discovery={"entry": discovery["entry"], "source_sha256": discovery["source_sha256"],
                   "regions": len(discovery["regions"]),
                   "boundary_policy": "reviewed deferred API; conservative opaque manager actions"},
        variants=variants)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.root, args.output)
