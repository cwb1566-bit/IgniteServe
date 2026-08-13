import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from vllm.compilation.startup_resources import enabled_by_master, env_flag
from vllm.logger import init_logger

logger = init_logger(__name__)

WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
}
CHUNK_SIZE = 8 * 1024 * 1024

PREWARM_PID_ENV = "VLLM_WEIGHT_PAGECACHE_PREWARM_PID"
PREWARM_PGID_ENV = "VLLM_WEIGHT_PAGECACHE_PREWARM_PGID"
PREWARM_MODEL_ENV = "VLLM_WEIGHT_PAGECACHE_PREWARM_MODEL"
PREWARM_STARTED_AT_ENV = "VLLM_WEIGHT_PAGECACHE_PREWARM_STARTED_AT"
PREWARM_ENV_KEYS = (
    PREWARM_PID_ENV,
    PREWARM_PGID_ENV,
    PREWARM_MODEL_ENV,
    PREWARM_STARTED_AT_ENV,
)


def weight_prewarm_enabled() -> bool:
    return enabled_by_master("VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM")


def _index_weight_files(model_dir: Path) -> set[Path]:
    files: set[Path] = set()
    for index_path in model_dir.glob("*.index.json"):
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            continue
        for rel_path in weight_map.values():
            if isinstance(rel_path, str):
                files.add(model_dir / rel_path)
    return files


def collect_weight_files(model_path: str) -> list[Path]:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        return [path] if path.suffix in WEIGHT_SUFFIXES else []
    if not path.is_dir():
        return []

    files = _index_weight_files(path)
    for suffix in WEIGHT_SUFFIXES:
        files.update(path.glob(f"*{suffix}"))
    return sorted(p for p in files if p.exists() and p.is_file())


def _read_file(path: Path, chunk_size: int = CHUNK_SIZE) -> int:
    total = 0
    with path.open("rb", buffering=0) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
    return total


def warm_files(files: Iterable[Path]) -> tuple[int, int, float]:
    started = time.perf_counter()
    total_bytes = 0
    total_files = 0
    for path in files:
        try:
            total_bytes += _read_file(path)
            total_files += 1
        except FileNotFoundError:
            continue
    return total_files, total_bytes, time.perf_counter() - started


def drop_files(files: Iterable[Path]) -> tuple[int, int, float]:
    started = time.perf_counter()
    total_bytes = 0
    total_files = 0
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", 4)
    for path in files:
        try:
            size = path.stat().st_size
            with path.open("rb", buffering=0) as f:
                if advise is not None:
                    advise(f.fileno(), 0, 0, dontneed)
            total_bytes += size
            total_files += 1
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Failed to drop page cache for %s: %s", path, exc)
    return total_files, total_bytes, time.perf_counter() - started


def _vmtouch_path() -> Optional[str]:
    from shutil import which
    return which("vmtouch")


def _clear_prewarm_env() -> None:
    for key in PREWARM_ENV_KEYS:
        os.environ.pop(key, None)


def _publish_prewarm_env(proc: subprocess.Popen, model_path: str) -> None:
    os.environ[PREWARM_PID_ENV] = str(proc.pid)
    # start_new_session=True makes pid == process group id.
    os.environ[PREWARM_PGID_ENV] = str(proc.pid)
    os.environ[PREWARM_MODEL_ENV] = model_path
    os.environ[PREWARM_STARTED_AT_ENV] = f"{time.time():.6f}"


def _read_proc_state(pid: int) -> Optional[str]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return "unknown"
    parts = stat.split()
    return parts[2] if len(parts) > 2 else "unknown"


def _pid_is_running_work(pid: int) -> bool:
    state = _read_proc_state(pid)
    if state is None or state == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_pid_stopped(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_is_running_work(pid):
            return True
        time.sleep(0.05)
    return not _pid_is_running_work(pid)


def start_async_weight_prewarm(model_path: str) -> Optional[subprocess.Popen]:
    _clear_prewarm_env()
    if not weight_prewarm_enabled():
        return None
    files = collect_weight_files(model_path)
    if not files:
        logger.info("[WEIGHT_PREWARM] no weight files found for %s", model_path)
        return None

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [
        sys.executable,
        "-m",
        "vllm.v1.engine.weight_pagecache",
        "warm",
        "--model",
        model_path,
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _publish_prewarm_env(proc, model_path)
    total_bytes = sum(p.stat().st_size for p in files)
    logger.info(
        "[WEIGHT_PREWARM] started async pagecache prewarm pid=%s files=%s "
        "bytes=%.2f GiB model=%s",
        proc.pid, len(files), total_bytes / (1024 ** 3), model_path)
    return proc


def stop_async_weight_prewarm(
        proc: Optional[subprocess.Popen],
        reason: str = "load_weights",
        clear_env: bool = True) -> str:
    if proc is None:
        if clear_env:
            _clear_prewarm_env()
        return "disabled"
    rc = proc.poll()
    if rc is not None:
        logger.info("[WEIGHT_PREWARM] async prewarm already finished at %s "
                    "pid=%s rc=%s", reason, proc.pid, rc)
        if clear_env:
            _clear_prewarm_env()
        return "completed"

    logger.info("[WEIGHT_PREWARM] stopping async prewarm at %s pid=%s "
                "to avoid I/O contention", reason, proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        if clear_env:
            _clear_prewarm_env()
        return "completed"
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
    if clear_env:
        _clear_prewarm_env()
    return "stopped"


def stop_async_weight_prewarm_from_env(reason: str = "load_weights") -> str:
    pid_text = os.environ.get(PREWARM_PID_ENV)
    if not pid_text:
        return "disabled"
    try:
        pid = int(pid_text)
        pgid = int(os.environ.get(PREWARM_PGID_ENV, pid_text))
    except ValueError:
        _clear_prewarm_env()
        return "invalid"

    model_path = os.environ.get(PREWARM_MODEL_ENV, "<unknown>")
    if not _pid_is_running_work(pid):
        logger.info("[WEIGHT_PREWARM] external prewarm already finished at %s "
                    "pid=%s model=%s", reason, pid, model_path)
        _clear_prewarm_env()
        return "completed"

    logger.info("[WEIGHT_PREWARM] stopping external prewarm at %s pid=%s "
                "pgid=%s model=%s to avoid I/O contention", reason, pid,
                pgid, model_path)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_prewarm_env()
        return "completed"
    except Exception as exc:
        logger.warning("[WEIGHT_PREWARM] failed to SIGTERM process group %s: %s; "
                       "falling back to pid %s", pgid, exc, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            _clear_prewarm_env()
            return "completed"

    if _wait_pid_stopped(pid, timeout=2.0):
        _clear_prewarm_env()
        return "stopped"

    logger.info("[WEIGHT_PREWARM] force killing external prewarm pid=%s pgid=%s",
                pid, pgid)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _wait_pid_stopped(pid, timeout=1.0)
    _clear_prewarm_env()
    return "killed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["warm", "drop"])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    files = collect_weight_files(args.model)
    if args.action == "warm":
        nfiles, nbytes, elapsed = warm_files(files)
    else:
        nfiles, nbytes, elapsed = drop_files(files)
    print(json.dumps({
        "action": args.action,
        "model": args.model,
        "files": nfiles,
        "bytes": nbytes,
        "elapsed_s": elapsed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
