# SPDX-License-Identifier: Apache-2.0

"""Compatibility metadata for offline CUDA graph profiling resources."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "cuda_graph_profile_manifest.json"
SCHEMA_VERSION = 1


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item)
                for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def build_cudagraph_profile_manifest(
    *,
    model_config: Any,
    cache_config: Any,
    scheduler_config: Any,
    parallel_config: Any,
    compilation_config: Any,
    device_properties: Any,
    dtype: Any,
    torch_version: str,
    vllm_version: str,
) -> dict[str, Any]:
    """Build a path-independent identity for graph shape and cost profiles."""
    hf_config = getattr(model_config, "hf_config", None)

    def hf(name: str, default: Any = None) -> Any:
        return getattr(hf_config, name, default)

    architectures = hf("architectures", ()) or ()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "architectures": list(architectures),
            "model_type": hf("model_type"),
            "hidden_size": hf("hidden_size"),
            "num_hidden_layers": hf("num_hidden_layers"),
            "num_attention_heads": hf("num_attention_heads"),
            "num_key_value_heads": hf("num_key_value_heads"),
            "vocab_size": hf("vocab_size"),
            "quantization": getattr(model_config, "quantization", None),
        },
        "runtime": {
            "dtype": str(dtype),
            "kv_cache_dtype": getattr(cache_config, "cache_dtype", None),
            "block_size": getattr(cache_config, "block_size", None),
            "max_model_len": getattr(model_config, "max_model_len", None),
            "max_num_batched_tokens": getattr(
                scheduler_config, "max_num_batched_tokens", None),
            "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None),
            "tensor_parallel_size": getattr(
                parallel_config, "tensor_parallel_size", None),
            "pipeline_parallel_size": getattr(
                parallel_config, "pipeline_parallel_size", None),
            "full_cuda_graph": getattr(
                compilation_config, "full_cuda_graph", None),
            "cudagraph_capture_sizes": list(
                getattr(compilation_config, "cudagraph_capture_sizes", ())),
        },
        "device": {
            "name": getattr(device_properties, "name", None),
            "compute_capability": [
                getattr(device_properties, "major", None),
                getattr(device_properties, "minor", None),
            ],
            "sm_count": getattr(
                device_properties, "multi_processor_count", None),
        },
        "software": {
            "torch": torch_version,
            "vllm": vllm_version,
        },
    }
    return _normalize(manifest)


def manifest_path(directory: str | Path) -> Path:
    return Path(directory) / MANIFEST_FILENAME


def load_cudagraph_profile_manifest(
        directory: str | Path) -> dict[str, Any] | None:
    path = manifest_path(directory)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def compare_cudagraph_profile_manifests(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    """Return stable field-level incompatibilities."""
    expected_flat = _flatten(_normalize(expected))
    actual_flat = _flatten(_normalize(actual))
    mismatches = []
    for key in sorted(set(expected_flat) | set(actual_flat)):
        if expected_flat.get(key) != actual_flat.get(key):
            mismatches.append(
                f"{key}: expected={expected_flat.get(key)!r}, "
                f"found={actual_flat.get(key)!r}")
    return mismatches


def ensure_cudagraph_profile_manifest(
    directory: str | Path,
    current: dict[str, Any],
    *,
    allow_overwrite: bool = False,
) -> tuple[bool, str]:
    """Create or validate a profile manifest without silently replacing it."""
    directory_path = Path(directory)
    existing = load_cudagraph_profile_manifest(directory_path)
    legacy_resources = (
        list(directory_path.glob("*.csv")) if directory_path.is_dir() else [])
    if existing is None and legacy_resources and not allow_overwrite:
        names = ",".join(sorted(path.name for path in legacy_resources))
        return False, f"manifest-missing-for-existing-resources: {names}"
    if existing is not None:
        mismatches = compare_cudagraph_profile_manifests(current, existing)
        if not mismatches:
            return True, "manifest-match"
        if not allow_overwrite:
            return False, "; ".join(mismatches)

    directory_path.mkdir(parents=True, exist_ok=True)
    path = manifest_path(directory_path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_normalize(current), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return True, "manifest-created" if existing is None else "manifest-replaced"


def validate_cudagraph_profile_manifest(
    directory: str | Path,
    current: dict[str, Any],
) -> tuple[bool, str]:
    existing = load_cudagraph_profile_manifest(directory)
    if existing is None:
        return False, "manifest-missing"
    mismatches = compare_cudagraph_profile_manifests(current, existing)
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "manifest-match"
