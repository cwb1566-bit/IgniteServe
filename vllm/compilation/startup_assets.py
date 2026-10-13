# SPDX-License-Identifier: Apache-2.0
"""Small disk primitives. Callers own compatibility and fallback decisions."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from .startup_resources import resource_dir

logger = logging.getLogger(__name__)
SCHEMA = "igniteserve-asset-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest") else _stream_hash(handle)


def _stream_hash(handle) -> str:
    result = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


@lru_cache(maxsize=32)
def source_hash(*relative_paths: str) -> str:
    root = Path(__file__).resolve().parents[1]
    files = {}
    for name in relative_paths:
        path = root / name
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                files[str(child.relative_to(root))] = file_hash(child)
        else:
            files[name] = file_hash(path)
    return digest(files)


@contextmanager
def locked(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False).encode())


def record(identity: Any, data: Any) -> dict:
    return {"schema": SCHEMA, "identity": identity, "data": data,
            "checksum": digest([identity, data])}


def read_record(path: Path, identity: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value["schema"] != SCHEMA or digest(value["identity"]) != digest(identity)
                or value["checksum"] != digest([identity, value["data"]])):
            raise ValueError("schema, identity or checksum mismatch")
        return value["data"]
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Startup asset rejected %s: %s", path, exc)
        return None


def asset_path(kind: str, identity: Any) -> Path:
    return resource_dir(kind) / f"{digest(identity)}.json"


def load(kind: str, identity: Any, override: str | None = None) -> Any:
    if not override:
        return read_record(asset_path(kind, identity), identity)
    # An explicit legacy FILE remains a multi-entry store, but old unverified
    # entries are misses. New entries use the same validation as the repository.
    path = Path(override).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        entry = value["entries"][digest(identity)]
        if (entry["schema"] == SCHEMA and digest(entry["identity"]) == digest(identity)
                and entry["checksum"] == digest([identity, entry["data"]])):
            return entry["data"]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def save(kind: str, identity: Any, data: Any,
         override: str | None = None) -> Path:
    path = Path(override).expanduser() if override else asset_path(kind, identity)
    with locked(path):
        value = record(identity, data)
        if override:
            try:
                store = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(store.get("entries"), dict):
                    raise ValueError("invalid entries")
            except (OSError, ValueError, TypeError, AttributeError):
                store = {"entries": {}}
            store["entries"][digest(identity)] = value
            value = store
        write_json(path, value)
    return path


def member(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"unsafe asset member: {relative}")
    root = root.resolve()
    original = root
    for part in path.parts:
        original = original / part
        if original.is_symlink():
            raise ValueError(f"symlink in asset member: {relative}")
    resolved = original.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"asset member escapes bundle: {relative}")
    return resolved


def publish_bundle(root: Path, identity: Any, files: list[str]) -> None:
    hashes = {name: file_hash(member(root, name)) for name in files}
    with locked(root / "manifest.json"):
        write_json(root / "manifest.json", record(identity, {"files": hashes}))


def verify_bundle(root: Path, identity: Any) -> bool:
    data = read_record(root / "manifest.json", identity)
    if not isinstance(data, dict) or not data.get("files"):
        return False
    try:
        return all(file_hash(member(root, name)) == sha
                   for name, sha in data["files"].items())
    except (OSError, ValueError, TypeError, AttributeError):
        return False
