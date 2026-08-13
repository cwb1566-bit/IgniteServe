import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch

from vllm.compilation.startup_resources import (
    enabled_by_master,
    env_path,
    resource_file,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_STORE = resource_file("best_config", "best_config_store.json")
_SCHEMA = "triton-best-config-cache-v1"


def enabled() -> bool:
    return enabled_by_master("VLLM_ENABLE_BEST_CONFIG_CACHE")


def store_path() -> Path:
    return env_path("VLLM_BEST_CONFIG_CACHE_FILE", _DEFAULT_STORE)


def _json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _version_payload() -> dict[str, Any]:
    try:
        import triton
        triton_version = getattr(triton, "__version__", None)
    except Exception:
        triton_version = None
    try:
        from vllm import __version__ as vllm_version
    except Exception:
        vllm_version = None
    return {
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton_version": triton_version,
        "vllm_version": vllm_version,
    }


def _device_payload(device: Optional[torch.device] = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device_type": "cuda", "available": False}
    if device is None:
        idx = torch.cuda.current_device()
    else:
        idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "device_type": "cuda",
        "gpu_name": props.name,
        "gpu_major": int(props.major),
        "gpu_minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
    }


def build_key(kernel_name: str, configs_hash: str,
              device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": _SCHEMA,
        "kernel_name": kernel_name,
        "configs_hash": configs_hash,
        "device": _device_payload(device),
        "versions": _version_payload(),
    }
    return _json_hash(payload), payload


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "schema": _SCHEMA, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read best-config cache %s: %s", path, exc)
        return {"version": 1, "schema": _SCHEMA, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "schema": _SCHEMA, "entries": {}}
    data.setdefault("version", 1)
    data.setdefault("schema", _SCHEMA)
    data.setdefault("entries", {})
    if not isinstance(data["entries"], dict):
        data["entries"] = {}
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent),
                                    encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        tmp_name = f.name
    os.replace(tmp_name, path)


def load_best_config_data(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    best_path = entry.get("best_config_path")
    if best_path:
        try:
            path = Path(best_path)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Failed to load persisted best_config %s: %s",
                           best_path, exc)
    data = entry.get("best_config")
    return data if isinstance(data, dict) else None


def lookup(kernel_name: str, configs_hash: str,
           device: Optional[torch.device] = None) -> tuple[Optional[dict[str, Any]], str, dict[str, Any], Optional[dict[str, Any]]]:
    key, payload = build_key(kernel_name, configs_hash, device)
    if not enabled():
        return None, key, payload, None
    store = _read_store(store_path())
    entry = store.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return None, key, payload, None
    best = load_best_config_data(entry)
    if not isinstance(best, dict):
        return None, key, payload, entry
    return best, key, payload, entry


def save(kernel_name: str, configs_hash: str, best_config_path: str,
         best_config: dict[str, Any], *, source_root: Optional[str] = None,
         source_model: Optional[str] = None,
         device: Optional[torch.device] = None) -> tuple[str, Path]:
    key, payload = build_key(kernel_name, configs_hash, device)
    path = store_path()
    if not enabled():
        return key, path
    store = _read_store(path)
    entries = store.setdefault("entries", {})
    prev = entries.get(key) if isinstance(entries.get(key), dict) else {}
    source_models = list(prev.get("source_models", []))
    if source_model and source_model not in source_models:
        source_models.append(source_model)
    source_roots = list(prev.get("source_roots", []))
    if source_root and source_root not in source_roots:
        source_roots.append(source_root)
    entries[key] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "payload": payload,
        "best_config_path": str(best_config_path),
        "best_config": best_config,
        "source_root": source_root,
        "source_model": source_model,
        "source_roots": source_roots,
        "source_models": source_models,
    }
    _atomic_write(path, store)
    return key, path


def extract_kernel_names_from_dir(directory: Path) -> list[str]:
    names: list[str] = []
    marker = "'kernel_name': '"
    for py_path in directory.glob("*.py"):
        try:
            text = py_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        start = 0
        while True:
            i = text.find(marker, start)
            if i < 0:
                break
            j = text.find("'", i + len(marker))
            if j < 0:
                break
            name = text[i + len(marker):j]
            if name not in names:
                names.append(name)
            start = j + 1
    return names


def ingest_best_config_tree(root: str, *, source_model: Optional[str] = None,
                            device: Optional[torch.device] = None) -> dict[str, Any]:
    rootp = Path(root)
    stats = {
        "root": str(rootp),
        "source_model": source_model,
        "files": 0,
        "saved": 0,
        "missing_configs_hash": 0,
        "missing_kernel_name": 0,
        "errors": 0,
    }
    if not rootp.exists():
        stats["missing_root"] = True
        return stats
    for best_path in rootp.rglob("*.best_config"):
        stats["files"] += 1
        try:
            data = json.loads(best_path.read_text(encoding="utf-8"))
        except Exception:
            stats["errors"] += 1
            continue
        configs_hash = data.get("configs_hash")
        if not configs_hash:
            stats["missing_configs_hash"] += 1
            continue
        kernel_names = extract_kernel_names_from_dir(best_path.parent)
        if not kernel_names:
            stats["missing_kernel_name"] += 1
            continue
        for kernel_name in kernel_names:
            save(kernel_name, configs_hash, str(best_path), data,
                 source_root=str(rootp), source_model=source_model,
                 device=device)
            stats["saved"] += 1
    return stats
