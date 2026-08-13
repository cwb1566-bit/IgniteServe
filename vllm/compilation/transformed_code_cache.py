# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from types import CodeType
from typing import Any, Optional

import torch

from vllm.compilation.startup_resources import (
    enabled_by_master,
    env_path,
    resource_dir,
    resource_file,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_STORE = resource_file("transformed_code", "transformed_code_store.json")
_DEFAULT_ARTIFACT_DIR = resource_dir("transformed_code") / "artifacts"
_SCHEMA = "dynamo-transformed-code-cache-v1"


def enabled() -> bool:
    return enabled_by_master("VLLM_ENABLE_TRANSFORMED_CODE_CACHE")


def store_path() -> Path:
    return env_path("VLLM_TRANSFORMED_CODE_CACHE_FILE", _DEFAULT_STORE)


def artifact_dir() -> Path:
    return env_path("VLLM_TRANSFORMED_CODE_CACHE_DIR", _DEFAULT_ARTIFACT_DIR)


def _json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version_payload() -> dict[str, Any]:
    try:
        import depyf
        depyf_version = getattr(depyf, "__version__", None)
    except Exception:
        depyf_version = None
    try:
        from vllm import __version__ as vllm_version
    except Exception:
        vllm_version = None
    return {
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm_version": vllm_version,
        "depyf_version": depyf_version,
    }


def _const_payload(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"code_hash": _code_hash(value)}
    if isinstance(value, (tuple, list)):
        return [_const_payload(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _const_payload(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_const_payload(v) for v in value), key=repr)
    if isinstance(value, bytes):
        return {"bytes_sha256": _bytes_hash(value), "len": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _code_payload(code: CodeType) -> dict[str, Any]:
    payload = {
        "co_name": code.co_name,
        "co_argcount": code.co_argcount,
        "co_posonlyargcount": getattr(code, "co_posonlyargcount", 0),
        "co_kwonlyargcount": code.co_kwonlyargcount,
        "co_nlocals": code.co_nlocals,
        "co_stacksize": code.co_stacksize,
        "co_flags": code.co_flags,
        "co_code_sha256": _bytes_hash(code.co_code),
        "co_consts": _const_payload(code.co_consts),
        "co_names": list(code.co_names),
        "co_varnames": list(code.co_varnames),
        "co_freevars": list(code.co_freevars),
        "co_cellvars": list(code.co_cellvars),
    }
    exception_table = getattr(code, "co_exceptiontable", b"")
    if exception_table:
        payload["co_exceptiontable_sha256"] = _bytes_hash(exception_table)
    return payload


def _code_hash(code: CodeType) -> str:
    return _json_hash(_code_payload(code))


def build_key(old_code: CodeType, new_code: CodeType,
              wrapper_class: Optional[type] = None) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema": _SCHEMA,
        "old_code_hash": _code_hash(old_code),
        "new_code_hash": _code_hash(new_code),
        "versions": _version_payload(),
    }
    if wrapper_class is not None:
        payload["wrapper_class"] = {
            "module": getattr(wrapper_class, "__module__", None),
            "qualname": getattr(wrapper_class, "__qualname__", None),
        }
    return _json_hash(payload), payload


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "schema": _SCHEMA, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read transformed-code cache %s: %s", path,
                       exc)
        return {"version": 1, "schema": _SCHEMA, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "schema": _SCHEMA, "entries": {}}
    data.setdefault("version", 1)
    data.setdefault("schema", _SCHEMA)
    data.setdefault("entries", {})
    if not isinstance(data["entries"], dict):
        data["entries"] = {}
    return data


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent),
                                    encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        tmp_name = f.name
    os.replace(tmp_name, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as src_file:
        with tempfile.NamedTemporaryFile("wb", delete=False,
                                        dir=str(dst.parent)) as dst_file:
            while True:
                chunk = src_file.read(1024 * 1024)
                if not chunk:
                    break
                dst_file.write(chunk)
            tmp_name = dst_file.name
    os.replace(tmp_name, dst)


def lookup(old_code: CodeType, new_code: CodeType, target_file: str,
           wrapper_class: Optional[type] = None) -> tuple[bool, str, dict[str, Any], Optional[dict[str, Any]]]:
    key, payload = build_key(old_code, new_code, wrapper_class)
    if not enabled():
        return False, key, payload, None
    store = _read_store(store_path())
    entry = store.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return False, key, payload, None
    source = entry.get("transformed_code_path")
    if not source:
        return False, key, payload, entry
    source_path = Path(source)
    if not source_path.exists():
        return False, key, payload, entry
    expected_sha = entry.get("sha256")
    if expected_sha:
        actual_sha = _bytes_hash(source_path.read_bytes())
        if actual_sha != expected_sha:
            logger.warning("Ignoring transformed-code cache entry %s: sha256 "
                           "mismatch", key[:12])
            return False, key, payload, entry
    _atomic_copy(source_path, Path(target_file))
    return True, key, payload, entry


def save(old_code: CodeType, new_code: CodeType, transformed_file: str,
         wrapper_class: Optional[type] = None, *,
         local_cache_dir: Optional[str] = None) -> tuple[str, Path]:
    key, payload = build_key(old_code, new_code, wrapper_class)
    path = store_path()
    if not enabled():
        return key, path

    source = Path(transformed_file)
    if not source.exists():
        return key, path
    content = source.read_bytes()
    sha256 = _bytes_hash(content)

    root = artifact_dir()
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / f"{key}.py"
    if not artifact.exists() or _bytes_hash(artifact.read_bytes()) != sha256:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(root)) as f:
            f.write(content)
            tmp_name = f.name
        os.replace(tmp_name, artifact)

    store = _read_store(path)
    entries = store.setdefault("entries", {})
    prev = entries.get(key) if isinstance(entries.get(key), dict) else {}
    local_cache_dirs = list(prev.get("local_cache_dirs", []))
    if local_cache_dir and local_cache_dir not in local_cache_dirs:
        local_cache_dirs.append(local_cache_dir)
    entries[key] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "payload": payload,
        "transformed_code_path": str(artifact),
        "sha256": sha256,
        "source_transformed_code_path": str(source),
        "local_cache_dir": local_cache_dir,
        "local_cache_dirs": local_cache_dirs,
    }
    _atomic_write_json(path, store)
    return key, path
