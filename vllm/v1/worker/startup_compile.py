import gc
import json
import os
import pathlib
import time
from typing import Any, Iterable

import torch
from vllm.compilation.startup_resources import startup_optimizations_enabled
from vllm.logger import init_logger
from vllm.v1.worker import best_config_cache
from vllm.v1.phase_profiler import timer

logger = init_logger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE


def startup_compile_mode() -> str:
    explicit = os.environ.get("VLLM_STARTUP_COMPILE_MODE")
    if explicit is not None:
        return explicit.strip().lower()
    return "s4" if startup_optimizations_enabled() else "s0"


def _best_config_matches(config: Any, data: dict[str, Any]) -> bool:
    if getattr(config, "num_warps", None) != data.get("num_warps"):
        return False
    if getattr(config, "num_stages", None) != data.get("num_stages"):
        return False
    for key, value in getattr(config, "kwargs", {}).items():
        if data.get(key) != value:
            return False
    return True


def build_semantic_best_map(root: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    mapping: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rootp = pathlib.Path(root)
    if not rootp.exists():
        return mapping
    for best_path in rootp.rglob("*.best_config"):
        try:
            data = json.loads(best_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        configs_hash = data.get("configs_hash")
        if not configs_hash:
            continue
        for py_path in best_path.parent.glob("*.py"):
            try:
                text = py_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            marker = "'kernel_name': '"
            start = 0
            while True:
                i = text.find(marker, start)
                if i < 0:
                    break
                j = text.find("'", i + len(marker))
                if j < 0:
                    break
                kernel_name = text[i + len(marker):j]
                mapping.setdefault((kernel_name, configs_hash), []).append(data)
                start = j + 1
    return mapping


def _unique_matching_launchers(launchers: Iterable[Any],
                               data_items: Iterable[dict[str, Any]]) -> list[Any]:
    matches = []
    for data in data_items:
        for launcher in launchers:
            if _best_config_matches(launcher.config, data):
                matches.append(launcher)

    unique = []
    seen = set()
    for launcher in matches:
        config = launcher.config
        key = (
            tuple(sorted(getattr(config, "kwargs", {}).items())),
            getattr(config, "num_warps", None),
            getattr(config, "num_stages", None),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(launcher)
    return unique


def select_semantic_best_launchers(
        tuners: Iterable[Any],
        offline_root: str | None = None,
        *,
        device: torch.device | None = None) -> dict[str, Any]:
    from torch._inductor.runtime.triton_heuristics import hash_configs

    start = time.perf_counter()
    semantic = build_semantic_best_map(offline_root) if offline_root else {}
    stats = {
        "store": str(best_config_cache.store_path()),
        "offline_root": offline_root,
        "semantic_keys": len(semantic),
        "tuners": 0,
        "selected": 0,
        "selected_from_store": 0,
        "selected_from_offline_root": 0,
        "already_single": 0,
        "no_kernel_name": 0,
        "no_configs_hash": 0,
        "cache_miss": 0,
        "no_match": 0,
        "ambiguous_match": 0,
    }
    for tuner in tuners:
        stats["tuners"] += 1
        launchers = getattr(tuner, "launchers", [])
        if len(launchers) <= 1:
            stats["already_single"] += 1
            continue
        kernel_name = getattr(tuner, "inductor_meta", {}).get("kernel_name")
        if not kernel_name:
            stats["no_kernel_name"] += 1
            continue
        try:
            configs_hash = hash_configs([launcher.config for launcher in launchers])
        except Exception:
            configs_hash = None
        if not configs_hash:
            stats["no_configs_hash"] += 1
            continue

        source = None
        data_items: list[dict[str, Any]] = []
        best_data, _, _, entry = best_config_cache.lookup(
            kernel_name, configs_hash, device)
        if best_data is not None:
            data_items = [best_data]
            source = "store"
        else:
            candidates = semantic.get((kernel_name, configs_hash))
            if candidates:
                data_items = candidates
                source = "offline_root"

        if not data_items:
            stats["cache_miss"] += 1
            continue

        unique = _unique_matching_launchers(launchers, data_items)
        if len(unique) == 1:
            tuner.launchers = [unique[0]]
            stats["selected"] += 1
            if source == "store":
                stats["selected_from_store"] += 1
            elif source == "offline_root":
                stats["selected_from_offline_root"] += 1
        elif len(unique) > 1:
            stats["ambiguous_match"] += 1
        else:
            stats["no_match"] += 1
    stats["wall_s"] = round(time.perf_counter() - start, 6)
    return stats


def _config_to_best_data(config: Any, configs_hash: str | None) -> dict[str, Any]:
    """Serialize a Triton launcher config into a persistable best_config dict.

    The layout mirrors the ``*.best_config`` files consumed by
    ``build_semantic_best_map`` / ``_best_config_matches`` so that a saved
    entry can be matched back to a launcher on the next cold start.
    """
    data: dict[str, Any] = {
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
    }
    for key, value in getattr(config, "kwargs", {}).items():
        data[key] = value
    if configs_hash:
        data["configs_hash"] = configs_hash
    return data


def compile_only_forward(model_runner: Any, *, autotune: bool = False,
                         resolve_best_config: bool = False,
                         device: torch.device | None = None,
                         offline_root: str | None = None,
                         persist_best_config: bool = False) -> dict[str, Any]:
    """Trigger Dynamo/Inductor/Triton precompile while suppressing normal launches.

    Three behaviors are available for multi-launcher Triton tuners during the
    dummy run:

    * ``autotune``: benchmark and collapse to one launcher unconditionally.
    * ``resolve_best_config``: look the kernel up in the persisted best-config
      store (default ``vllm_startup_resources/best_config``) plus the optional
      ``offline_root`` semantic map. On a hit, load the cached config and skip
      autotune; on a miss/ambiguous match, autotune to one config and -- when
      ``persist_best_config`` is set -- save the chosen config back into the
      store so the next cold start can load it directly.
    * neither: leave autotune for the first real execution stage.
    """
    from torch._inductor.runtime.triton_heuristics import (CachingAutotuner,
                                                           hash_configs)
    from torch._inductor.select_algorithm import extern_kernels

    if device is None:
        device = getattr(model_runner, "device", None)
    semantic = (build_semantic_best_map(offline_root)
                if resolve_best_config and offline_root else {})
    pending_saves: list[tuple[str, str | None, dict[str, Any]]] = []

    original_ca_run = CachingAutotuner.run
    original_mm = extern_kernels.mm
    tuners: list[Any] = []
    seen_tuners: set[int] = set()
    stats = {
        "autotune": autotune,
        "resolve_best_config": resolve_best_config,
        "tuners": 0,
        "precompile_calls": 0,
        "autotune_calls": 0,
        "autotune_s": 0.0,
        "loaded_from_cache": 0,
        "autotuned_and_saved": 0,
        "wall_s": 0.0,
    }

    def _autotune(self_ca, args, kwargs, stream) -> None:
        from vllm.compilation.weight_compile_overlap import (
            wait_for_weights_ready)
        wait_for_weights_ready()
        st = time.perf_counter()
        self_ca.autotune_to_one_config(*args, **kwargs)
        if stream is not None:
            torch.cuda.synchronize()
        stats["autotune_calls"] += 1
        stats["autotune_s"] += time.perf_counter() - st

    def _resolve_best_config(self_ca, args, kwargs, stream) -> None:
        kernel_name = getattr(self_ca, "inductor_meta", {}).get("kernel_name")
        if not kernel_name:
            _autotune(self_ca, args, kwargs, stream)
            return
        try:
            configs_hash = hash_configs(
                [launcher.config for launcher in self_ca.launchers])
        except Exception:
            configs_hash = None

        data_items: list[dict[str, Any]] = []
        if configs_hash:
            best_data, _, _, _ = best_config_cache.lookup(
                kernel_name, configs_hash, device)
            if best_data is not None:
                data_items = [best_data]
            elif semantic:
                candidates = semantic.get((kernel_name, configs_hash))
                if candidates:
                    data_items = candidates

        if data_items:
            unique = _unique_matching_launchers(self_ca.launchers, data_items)
            if len(unique) == 1:
                self_ca.launchers = [unique[0]]
                stats["loaded_from_cache"] += 1
                return

        # Cache miss or ambiguous match: autotune now and persist the winner so
        # subsequent cold starts can load it from the resource store directly.
        _autotune(self_ca, args, kwargs, stream)
        if persist_best_config and self_ca.launchers:
            data = _config_to_best_data(self_ca.launchers[0].config, configs_hash)
            pending_saves.append((kernel_name, configs_hash, data))

    def compile_only_ca_run(self_ca, *args, stream=None, benchmark_run=False, **kwargs):
        ident = id(self_ca)
        if ident not in seen_tuners:
            seen_tuners.add(ident)
            tuners.append(self_ca)
        if len(self_ca.launchers) == 0:
            self_ca.precompile()
            stats["precompile_calls"] += 1
        if resolve_best_config and len(self_ca.launchers) > 1:
            _resolve_best_config(self_ca, args, kwargs, stream)
        elif autotune and len(self_ca.launchers) > 1:
            _autotune(self_ca, args, kwargs, stream)
        return None

    def compile_only_mm(*args, **kwargs):
        a, b = args[0], args[1]
        out = kwargs.get("out")
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
        return out

    CachingAutotuner.run = compile_only_ca_run
    extern_kernels.mm = compile_only_mm
    start = time.perf_counter()
    try:
        timer.start("startup_compile_dummy_run")
        model_runner._dummy_run(model_runner.max_num_tokens, is_profile=True)
        timer.end("startup_compile_dummy_run")
        timer.start("startup_compile_cuda_sync")
        torch.cuda.synchronize()
        timer.end("startup_compile_cuda_sync")
    finally:
        CachingAutotuner.run = original_ca_run
        extern_kernels.mm = original_mm

    if pending_saves:
        timer.start("startup_best_config_save")
        model_name = None
        vllm_config = getattr(model_runner, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            model_name = getattr(model_config, "model", None)
        saved = 0
        for kernel_name, configs_hash, data in pending_saves:
            if not configs_hash:
                continue
            try:
                best_config_cache.save(kernel_name, configs_hash, "", data,
                                       source_model=model_name, device=device)
                saved += 1
            except Exception as exc:
                logger.warning(
                    "Failed to persist startup best_config for %s: %s",
                    kernel_name, exc)
        stats["autotuned_and_saved"] = saved
        timer.end("startup_best_config_save")

    stats["wall_s"] = round(time.perf_counter() - start, 6)
    stats["tuners"] = len(tuners)
    stats["autotune_s"] = round(stats["autotune_s"], 6)
    model_runner._startup_compile_tuners = tuners
    timer.start("startup_compile_cleanup")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    timer.end("startup_compile_cleanup")
    logger.info("startup compile-only phase finished: %s", stats)
    return stats


def maybe_apply_offline_best_config(model_runner: Any) -> dict[str, Any] | None:
    offline_root = os.environ.get("VLLM_OFFLINE_BEST_CONFIG_ROOT")
    tuners = getattr(model_runner, "_startup_compile_tuners", [])
    if not tuners:
        return None
    device = getattr(model_runner, "device", None)
    stats = select_semantic_best_launchers(
        tuners, offline_root, device=device)
    logger.info("startup best-config semantic load: %s", stats)
    return stats
