import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch

import vllm.envs as envs
from vllm.compilation.startup_resources import (
    enabled_by_master,
    env_path,
    resource_file,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_STORE = resource_file("kv_profile", "kv_profile_store.json")


def enabled() -> bool:
    return enabled_by_master("VLLM_ENABLE_KV_PROFILE_CACHE")


def store_path() -> Path:
    return env_path("VLLM_KV_PROFILE_CACHE_FILE", _DEFAULT_STORE)


def _json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_model_identity(model_path: Any) -> Any:
    if not isinstance(model_path, str):
        return model_path
    try:
        from vllm.compilation.cache_reuse import canonical_model_identity
        return canonical_model_identity(model_path)
    except Exception:
        return model_path


def _device_fingerprint(device: torch.device) -> dict[str, Any]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "device_type": str(device.type),
        "gpu_name": props.name,
        "gpu_total_memory": int(props.total_memory),
        "gpu_major": int(props.major),
        "gpu_minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
    }


def build_key(vllm_config: Any, device: torch.device) -> tuple[str, dict[str, Any]]:
    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)

    from vllm import __version__ as vllm_version
    payload = {
        "schema": "kv-profile-cache-v2",
        # Do not use vllm_config.compute_hash() or envs.compute_hash() here:
        # they may include compile-cache-path or graph-cache factors that do
        # not affect the peak memory measured by profile_run(). The key below
        # is intentionally limited to model/runtime/memory/device factors.
        "vllm_version": vllm_version,
        "vllm_use_v1": envs.VLLM_USE_V1,
        "model_identity": _canonical_model_identity(
            getattr(model_config, "model", None)),
        "dtype": str(getattr(model_config, "dtype", None)),
        "max_model_len": getattr(model_config, "max_model_len", None),
        "served_model_name": str(getattr(model_config, "served_model_name", None)),
        "load_format": str(getattr(getattr(vllm_config, "load_config", None),
                                   "load_format", None)),
        "trust_remote_code": getattr(model_config, "trust_remote_code", None),
        "quantization": str(getattr(model_config, "quantization", None)),
        "quant_config": str(getattr(vllm_config, "quant_config", None)),
        "enforce_eager": getattr(model_config, "enforce_eager", None),
        "gpu_memory_utilization": getattr(cache_config,
                                          "gpu_memory_utilization", None),
        "kv_cache_dtype": str(getattr(cache_config, "cache_dtype", None)),
        "block_size": getattr(cache_config, "block_size", None),
        "swap_space": getattr(cache_config, "swap_space", None),
        "cpu_offload_gb": getattr(cache_config, "cpu_offload_gb", None),
        "max_num_batched_tokens": getattr(scheduler_config,
                                          "max_num_batched_tokens", None),
        "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None),
        "chunked_prefill_enabled": getattr(scheduler_config,
                                           "chunked_prefill_enabled", None),
        "tensor_parallel_size": getattr(parallel_config,
                                        "tensor_parallel_size", None),
        "pipeline_parallel_size": getattr(parallel_config,
                                          "pipeline_parallel_size", None),
        "data_parallel_size": getattr(parallel_config,
                                      "data_parallel_size", None),
        "disable_custom_all_reduce": getattr(parallel_config,
                                             "disable_custom_all_reduce", None),
        "compilation_level": getattr(compilation_config, "level", None),
        "use_cudagraph": getattr(compilation_config, "use_cudagraph", None),
        "full_cuda_graph": getattr(compilation_config, "full_cuda_graph", None),
        "device": _device_fingerprint(device),
    }
    return _json_hash(payload), payload


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read KV profile cache %s: %s", path, exc)
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "entries": {}}
    data.setdefault("version", 1)
    data.setdefault("entries", {})
    if not isinstance(data["entries"], dict):
        data["entries"] = {}
    return data


def lookup(vllm_config: Any, device: torch.device) -> tuple[Optional[int], str, dict[str, Any], Optional[dict[str, Any]]]:
    if not enabled():
        key, payload = build_key(vllm_config, device)
        return None, key, payload, None
    key, payload = build_key(vllm_config, device)
    path = store_path()
    data = _read_store(path)
    entry = data.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return None, key, payload, None
    value = entry.get("available_kv_cache_memory")
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None, key, payload, None
    if value <= 0:
        return None, key, payload, None
    return value, key, payload, entry


def save(vllm_config: Any, device: torch.device, available_kv_cache_memory: int,
         *, profile_result: Any = None) -> tuple[str, Path]:
    key, payload = build_key(vllm_config, device)
    path = store_path()
    if not enabled():
        return key, path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_store(path)
    entry = {
        "available_kv_cache_memory": int(available_kv_cache_memory),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "payload": payload,
    }
    if profile_result is not None:
        entry["profile"] = {
            "non_kv_cache_memory": int(getattr(profile_result,
                                               "non_kv_cache_memory", 0)),
            "torch_peak_increase": int(getattr(profile_result,
                                               "torch_peak_increase", 0)),
            "non_torch_increase": int(getattr(profile_result,
                                              "non_torch_increase", 0)),
        }
    data.setdefault("entries", {})[key] = entry
    with tempfile.NamedTemporaryFile("w", delete=False,
                                    dir=str(path.parent), encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        tmp_name = f.name
    os.replace(tmp_name, path)
    return key, path
