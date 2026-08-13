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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resource_root() -> Path:
    return Path(os.environ.get(
        "VLLM_STARTUP_RESOURCE_ROOT",
        str(repo_root() / "vllm_startup_resources"),
    ))


def resource_dir(name: str) -> Path:
    return resource_root() / name


def resource_file(name: str, filename: str) -> Path:
    return resource_dir(name) / filename


def parallel_compile_metadata_dir(cache_dir: str,
                                  rank: int,
                                  dp_rank: int,
                                  prefix: str) -> Path:
    """Return the global metadata dir for piecewise compile inputs/reuse maps."""
    compile_hash = Path(cache_dir).resolve().name
    prefix_name = prefix or "default"
    return (resource_dir("parallel_compile_metadata") / compile_hash /
            f"rank_{rank}_{dp_rank}" / prefix_name /
            "submodule_inputs_metadata")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def enabled_by_master(name: str, default: Optional[bool] = None) -> bool:
    if default is None:
        default = startup_optimizations_enabled()
    return env_flag(name, default)
