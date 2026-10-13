# SPDX-License-Identifier: Apache-2.0
"""Strict loading and execution of generated startup contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional

from vllm.v1.engine.startup_orchestrator import (
    ResourceAwareDAGExecutor,
    StageNode,
    StartupContext,
    StartupDAG,
    StartupState,
)


class GeneratedStartupRejected(RuntimeError):
    """Raised during preflight, before any generated action is invoked."""


def repository_contract(vllm_config, source_path):
    from vllm.compilation.startup_assets import digest, verify_bundle
    from vllm.compilation.startup_resources import resource_dir
    try:
        facts = runtime_compatibility_facts(vllm_config, source_path=source_path)
        facts.pop("cache_state", None)
        root = resource_dir("startup_contract") / digest(facts)
        if verify_bundle(root, facts):
            return str(root / "startup_contract.json")
    except (ValueError, TypeError, AttributeError, OSError):
        pass
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_architecture(model_config: Any) -> list[str]:
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        architectures = getattr(getattr(model_config, "hf_config", None),
                                "architectures", None)
    return sorted(str(item) for item in (architectures or []))


def runtime_compatibility_facts(vllm_config: Any, *,
                                source_path: Optional[Path] = None
                                ) -> dict[str, Any]:
    """Collect facts that are safe to inspect before EngineCore mutation."""
    model = vllm_config.model_config
    scheduler = vllm_config.scheduler_config
    parallel = vllm_config.parallel_config
    try:
        import torch
        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
        if torch.cuda.is_available():
            capability = ".".join(str(item) for item in
                                  torch.cuda.get_device_capability())
            gpu_name = str(torch.cuda.get_device_name())
        else:
            capability = None
            gpu_name = None
    except Exception:
        torch_version = None
        cuda_version = None
        capability = None
        gpu_name = None
    facts = {
        "cache_state": os.environ.get("VLLM_STARTUP_CACHE_STATE"),
        "model": str(getattr(model, "model", "")),
        "model_architecture": _model_architecture(model),
        "dtype": str(getattr(model, "dtype", "")),
        "max_model_len": getattr(model, "max_model_len", None),
        "max_num_batched_tokens": getattr(
            scheduler, "max_num_batched_tokens", None),
        "max_num_seqs": getattr(scheduler, "max_num_seqs", None),
        "tensor_parallel_size": getattr(
            parallel, "tensor_parallel_size", None),
        "pipeline_parallel_size": getattr(
            parallel, "pipeline_parallel_size", None),
        "data_parallel_size": getattr(parallel, "data_parallel_size", None),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_compute_capability": capability,
        "gpu_name": gpu_name,
    }
    if source_path is not None:
        facts["source_sha256"] = _sha256(Path(source_path))
    return facts


def _matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(key in actual and actual[key] == value
               for key, value in expected.items())


def _load_adapter(path: Path, expected_sha: Optional[str]) -> ModuleType:
    if not path.is_file():
        raise GeneratedStartupRejected(f"adapter does not exist: {path}")
    actual_sha = _sha256(path)
    if expected_sha and actual_sha != expected_sha:
        raise GeneratedStartupRejected(
            f"adapter hash mismatch: expected={expected_sha} actual={actual_sha}")
    name = f"igniteserve_startup_adapter_{actual_sha[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GeneratedStartupRejected(f"cannot import adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_action_registry", None)):
        raise GeneratedStartupRejected(
            "adapter must export build_action_registry(engine_core)")
    return module


@dataclass
class GeneratedStartupPlan:
    contract_path: Path
    adapter_path: Path
    payload: dict[str, Any]
    variant: dict[str, Any]
    dag: StartupDAG
    context: StartupContext
    executor: ResourceAwareDAGExecutor
    facts: dict[str, Any]

    def execute(self) -> dict[str, Any]:
        started = time.perf_counter()
        from vllm.v1.engine.startup_analysis.recorder import (
            get_startup_recorder,
        )
        recorder = get_startup_recorder()
        try:
            results = self.executor.run(self.dag, self.context)
            report_path = os.environ.get("VLLM_STARTUP_GENERATED_REPORT")
            if report_path:
                output = Path(report_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({
                    "schema": "igniteserve-generated-startup-run-v1",
                    "contract": str(self.contract_path),
                    "adapter": str(self.adapter_path),
                    "variant": self.variant["name"],
                    "elapsed_s": time.perf_counter() - started,
                    "facts": self.facts,
                    "stages": {name: result.__dict__
                               for name, result in results.items()},
                    "state_keys": sorted(self.context.snapshot()),
                }, indent=2, sort_keys=True, default=str), encoding="utf-8")
            return results
        finally:
            if recorder is not None:
                recorder.flush()


def prepare_generated_startup(
    engine_core: Any,
    vllm_config: Any,
    constructor_inputs: Mapping[str, Any],
    *,
    contract_path: Path,
    adapter_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
) -> GeneratedStartupPlan:
    """Validate every binding before returning an executable plan."""
    contract_path = Path(contract_path).resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GeneratedStartupRejected(
            f"cannot read generated contract {contract_path}: {exc}") from exc
    if payload.get("schema") != "igniteserve-startup-contract-v2":
        raise GeneratedStartupRejected("unsupported generated contract schema")
    allow_candidate = os.environ.get(
        "VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION", "0").lower() in (
            "1", "true", "yes", "on")
    if payload.get("status") != "final" and not allow_candidate:
        raise GeneratedStartupRejected(
            f"generated contract status is {payload.get('status')!r}, not 'final'")

    facts = runtime_compatibility_facts(vllm_config, source_path=source_path)
    if (contract_path.parent / "manifest.json").exists():
        from vllm.compilation.startup_assets import verify_bundle
        identity = dict(facts)
        identity.pop("cache_state", None)
        if not verify_bundle(contract_path.parent, identity):
            raise GeneratedStartupRejected("contract bundle integrity/compatibility mismatch")
        # A cache-state label supplied by the caller is not a runtime probe.
        # Repository contracts may not use it to authorize skipping actions.
        facts["cache_state"] = None
    compatibility = dict(payload.get("compatibility", {}))
    if not _matches(compatibility, facts):
        mismatches = {
            key: {"expected": value, "actual": facts.get(key)}
            for key, value in compatibility.items()
            if facts.get(key) != value
        }
        raise GeneratedStartupRejected(
            f"generated contract compatibility mismatch: {mismatches}")
    variants = [variant for variant in payload.get("variants", [])
                if _matches(dict(variant.get("match", {})), facts)]
    if len(variants) != 1:
        raise GeneratedStartupRejected(
            f"expected exactly one matching variant, found {len(variants)}")
    variant = variants[0]

    adapter_spec = dict(payload.get("adapter", {}))
    resolved_adapter = Path(adapter_path) if adapter_path else Path(
        adapter_spec.get("path", "startup_adapter.py"))
    if not resolved_adapter.is_absolute():
        resolved_adapter = contract_path.parent / resolved_adapter
    resolved_adapter = resolved_adapter.resolve()
    module = _load_adapter(resolved_adapter, adapter_spec.get("sha256"))
    try:
        registry = module.build_action_registry(engine_core)
    except Exception as exc:
        raise GeneratedStartupRejected(
            f"adapter registry construction failed: {exc}") from exc
    if not isinstance(registry, Mapping):
        raise GeneratedStartupRejected("adapter registry is not a mapping")

    nodes = list(variant.get("nodes", []))
    names = [str(node.get("name", "")) for node in nodes]
    if not names or any(not name for name in names):
        raise GeneratedStartupRejected("variant contains an empty node set/name")
    if len(set(names)) != len(names):
        raise GeneratedStartupRejected("variant contains duplicate node names")
    name_set = set(names)
    capacities = {
        str(key): int(value)
        for key, value in dict(payload.get("resource_capacities", {})).items()
    }
    dag = StartupDAG()
    producers: dict[str, list[str]] = {}
    initial_state = set(str(item) for item in payload.get("initial_state", []))
    initial_state.update(str(key) for key in constructor_inputs)
    for spec in nodes:
        name = str(spec["name"])
        if not bool(spec.get("executable", True)):
            raise GeneratedStartupRejected(f"node {name!r} is unresolved")
        dependencies = {str(item) for item in spec.get("dependencies", [])}
        unknown = dependencies - name_set
        if unknown:
            raise GeneratedStartupRejected(
                f"node {name!r} has unknown dependencies: {sorted(unknown)}")
        action_id = str(spec.get("action_id", ""))
        action = registry.get(action_id)
        if bool(spec.get("enabled", True)) and not callable(action):
            raise GeneratedStartupRejected(
                f"enabled node {name!r} has no callable action {action_id!r}")
        resources = {
            str(key): int(value)
            for key, value in dict(spec.get("resources", {})).items()
        }
        for resource, units in resources.items():
            capacity = capacities.get(resource, 1)
            if units < 0 or units > capacity:
                raise GeneratedStartupRejected(
                    f"node {name!r} requests {resource}={units}, capacity={capacity}")
        for state in spec.get("produced_state", []):
            producers.setdefault(str(state), []).append(name)
        dag.add_node(StageNode(
            name=name,
            action=action,
            dependencies=dependencies,
            estimated_cost_s=float(spec.get("estimated_cost_s", 0.0)),
            resources=resources,
            enabled=bool(spec.get("enabled", True)),
            category=str(spec.get("category", "generated")),
            details={
                "action_id": action_id,
                "required_state": list(spec.get("required_state", [])),
                "produced_state": list(spec.get("produced_state", [])),
                "generated_contract": str(contract_path),
            },
        ))
    order = dag.topological_order()
    ancestors: dict[str, set[str]] = {}
    for name in order:
        direct = set(dag.nodes[name].dependencies)
        ancestors[name] = direct | set().union(
            *(ancestors[item] for item in direct)) if direct else set()
    for spec in nodes:
        name = str(spec["name"])
        for state in (str(item) for item in spec.get("required_state", [])):
            if state in initial_state:
                continue
            candidates = producers.get(state, [])
            if len(candidates) != 1:
                raise GeneratedStartupRejected(
                    f"state {state!r} required by {name!r} has "
                    f"{len(candidates)} producers")
            if candidates[0] not in ancestors[name]:
                raise GeneratedStartupRejected(
                    f"state producer {candidates[0]!r} is not an ancestor of {name!r}")

    state = StartupState(
        model=str(facts.get("model", "")),
        compile_mode="generated",
        kv_profile_hit=(facts.get("cache_state") == "D2"),
        attributes=facts,
    )
    context = StartupContext(state)
    for key, value in constructor_inputs.items():
        context.put(str(key), value)
    executor = ResourceAwareDAGExecutor(
        max_workers=max(1, int(os.environ.get(
            "VLLM_STARTUP_DAG_MAX_WORKERS", "2"))),
        resource_capacities=capacities,
    )
    return GeneratedStartupPlan(
        contract_path=contract_path,
        adapter_path=resolved_adapter,
        payload=payload,
        variant=variant,
        dag=dag,
        context=context,
        executor=executor,
        facts=facts,
    )
