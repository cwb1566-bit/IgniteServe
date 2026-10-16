# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Startup compile/profile optimization switches and resource paths.

The default startup path is S0: compile-only plus a real KV profile probe,
without semantic caches or structural graph reuse. Set
``VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1`` to enable the S4 path and use the
shared resource stores under ``VLLM_STARTUP_RESOURCE_ROOT``.
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return default


def startup_optimizations_enabled() -> bool:
    return env_flag("VLLM_ENABLE_STARTUP_OPTIMIZATIONS", False)


STARTUP_OPTIMIZATION_DEFAULTS = {
    "VLLM_STARTUP_DAG_MODE": "execute",
    "VLLM_ENABLE_DYNAMO_FRONTEND_CACHE": "1",
    "VLLM_ENABLE_SUBPROCESS_COMPILE": "1",
    "VLLM_ENABLE_ADAPTIVE_CUDAGRAPH": "1",
    "TORCHINDUCTOR_FULL_FX_FAST_SHUTDOWN": "1",
    "VLLM_ENABLE_WEIGHT_SHARD_PIPELINE": "1",
    "VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP": "1",
    "VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM": "1",
    "VLLM_ENABLE_ASYNC_COMPILER_HASH_PREWARM": "1",
}


def apply_startup_optimization_defaults() -> dict[str, str]:
    """Resolve process-wide defaults before creating any engine workers.

    Explicit child settings win, including off/discover/generated DAG modes.
    Environment settings remain fixed for this engine process and its children.
    """
    if not startup_optimizations_enabled():
        return {}
    return {name: os.environ.setdefault(name, value)
            for name, value in STARTUP_OPTIMIZATION_DEFAULTS.items()}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _configured_root(config_file: str) -> Optional[str]:
    path = Path(config_file)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("startup_resource_root")
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"Invalid startup_resource_root in {path}")
    return value


def resource_root() -> Path:
    config_home = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")).expanduser()
    cache_home = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    value = os.getenv("VLLM_STARTUP_RESOURCE_ROOT")
    if value is None:
        value = _configured_root(str(config_home / "igniteserve/config.json"))
    return Path(value or cache_home / "igniteserve/assets").expanduser().resolve()


def resource_dir(name: str) -> Path:
    return resource_root() / name


def resource_file(name: str, filename: str) -> Path:
    return resource_dir(name) / filename


def parallel_compile_metadata_identity(cache_dir: str, rank: int,
                                       dp_rank: int, prefix: str,
                                       vllm_config=None) -> dict:
    from .startup_assets import source_hash
    identity = {"compile": Path(cache_dir).resolve().name,
                "rank": rank, "dp_rank": dp_rank, "prefix": prefix}
    if vllm_config is not None:
        import torch
        model = vllm_config.model_config
        hf = model.hf_config
        identity.pop("compile")
        identity.update({"architecture": getattr(hf, "architectures", None),
                         "model_type": getattr(hf, "model_type", None),
                         "dtype": str(model.dtype),
                         "quantization": str(model.quantization),
                         "tp": vllm_config.parallel_config.tensor_parallel_size,
                         "pp": vllm_config.parallel_config.pipeline_parallel_size,
                         "partition": os.getenv("VLLM_PP_LAYER_PARTITION"),
                         "torch": str(torch.__version__), "cuda": torch.version.cuda,
                         "source": source_hash("compilation"),
                         "device": list(torch.cuda.get_device_capability())
                         if torch.cuda.is_available() else None})
    return identity


def parallel_compile_metadata_dir(cache_dir: str, rank: int, dp_rank: int,
                                  prefix: str, vllm_config=None) -> Path:
    """Return the family-partitioned input metadata file."""
    from .startup_assets import digest
    identity = parallel_compile_metadata_identity(cache_dir, rank, dp_rank,
                                                   prefix, vllm_config)
    return resource_dir("input_metadata") / f"{digest(identity)}.json"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def enabled_by_master(name: str, default: Optional[bool] = None) -> bool:
    if default is None:
        default = startup_optimizations_enabled()
    return env_flag(name, default)


@lru_cache(maxsize=1)
def run_dir() -> Path:
    import time
    return resource_dir("runs") / f"{time.time_ns()}_{os.getpid()}"
