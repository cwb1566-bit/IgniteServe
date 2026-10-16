import os
import platform
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
from vllm.compilation import startup_assets as assets

logger = init_logger(__name__)


def enabled() -> bool:
    return enabled_by_master("VLLM_ENABLE_KV_PROFILE_CACHE")


def store_path() -> Path:
    return env_path("VLLM_KV_PROFILE_CACHE_FILE", resource_file("kv_profile", "legacy.json"))


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
        "schema": "kv-profile-cache-v3",
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
        "source": assets.source_hash("v1/worker", "model_executor/models",
                                     "model_executor/layers", "compilation"),
        "model_config": getattr(getattr(model_config, "hf_config", None),
                                "to_dict", lambda: {})(),
        "lora": str(getattr(vllm_config, "lora_config", None)),
        "speculative": str(getattr(vllm_config, "speculative_config", None)),
        "rank": getattr(parallel_config, "rank", None),
        "dp_rank": getattr(parallel_config, "data_parallel_rank", None),
        "partition": os.getenv("VLLM_PP_LAYER_PARTITION"),
        "allocator": os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""),
        "attention_backend": os.getenv("VLLM_ATTENTION_BACKEND", ""),
        "persistent": os.getenv("VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0"),
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
    payload["model_config"].pop("_name_or_path", None)
    return assets.digest(payload), payload


def lookup(vllm_config: Any, device: torch.device, *,
           weights_memory: Optional[int] = None,
           requested_memory: Optional[int] = None,
           init_snapshot: Any = None) -> tuple[Optional[int], str, dict[str, Any], Optional[dict[str, Any]]]:
    if not enabled():
        key, payload = build_key(vllm_config, device)
        return None, key, payload, None
    key, payload = build_key(vllm_config, device)
    entry = assets.load("kv_profile", payload, os.getenv("VLLM_KV_PROFILE_CACHE_FILE"))
    if not isinstance(entry, dict):
        return None, key, payload, None
    try:
        profile = entry["profile"]
        peak = profile["torch_peak_increase"]
        non_torch = profile["non_torch_increase"]
        saved_weights = entry["weights_memory"]
        if (not entry.get("complete") or weights_memory != saved_weights
                or requested_memory is None or init_snapshot is None
                or any(type(v) is not int or v < 0
                       for v in (peak, non_torch, saved_weights))):
            return None, key, payload, entry
        # Reuse measured increments, never a historical remaining budget.
        value = int(requested_memory) - weights_memory - peak - non_torch
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        reclaimable = max(0, reserved - allocated)
        margin = 64 * 1024 * 1024
        if (total != payload["device"]["gpu_total_memory"]
                or init_snapshot.free_memory < requested_memory
                or value + peak + non_torch + margin > free + reclaimable):
            return None, key, payload, entry
    except (TypeError, ValueError, KeyError):
        return None, key, payload, None
    if value <= 0:
        return None, key, payload, None
    return value, key, payload, entry


def save(vllm_config: Any, device: torch.device, available_kv_cache_memory: int,
         *, profile_result: Any = None,
         weights_memory: Optional[int] = None) -> tuple[str, Path]:
    key, payload = build_key(vllm_config, device)
    override = os.getenv("VLLM_KV_PROFILE_CACHE_FILE")
    path = Path(override) if override else assets.asset_path("kv_profile", payload)
    if not enabled():
        return key, path
    entry = {
        "available_kv_cache_memory": int(available_kv_cache_memory),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "payload": payload,
        "weights_memory": weights_memory,
        "complete": profile_result is not None and weights_memory is not None,
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
    assets.save("kv_profile", payload, entry, override)
    return key, path
