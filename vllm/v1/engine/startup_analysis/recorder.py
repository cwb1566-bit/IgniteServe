# SPDX-License-Identifier: Apache-2.0
"""Low-overhead event collection for startup contract generation."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any, Iterator, Optional


_CURRENT_STAGE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "igniteserve_startup_stage", default=None)
_RECORDER: Optional["StartupAnalysisRecorder"] = None
_RECORDER_LOCK = threading.Lock()


def _safe_value(value: Any, limit: int = 512, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        text = value
    elif depth < 3 and isinstance(value, (list, tuple)):
        return [_safe_value(item, limit, depth + 1) for item in value]
    elif depth < 3 and isinstance(value, dict):
        return {str(key): _safe_value(item, limit, depth + 1)
                for key, item in value.items()}
    else:
        try:
            text = repr(value)
        except Exception:
            text = f"<{type(value).__name__}>"
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "..."
    return text


class StartupAnalysisRecorder:
    """Collect stage, state, call, audit, and manually supplied resource events."""

    _AUDIT_PREFIXES = (
        "open",
        "os.listdir",
        "os.scandir",
        "mmap.__new__",
        "subprocess.Popen",
        "socket.connect",
        "socket.getaddrinfo",
    )

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.pid = os.getpid()
        self.trace_path = output_dir / f"startup_trace_{self.pid}.jsonl"
        self.summary_path = output_dir / f"startup_trace_{self.pid}.summary.json"
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self._started = False
        self._flushed = False
        self._dropped_events = 0
        self._max_events = max(
            1000, int(os.environ.get("VLLM_STARTUP_ANALYSIS_MAX_EVENTS", "500000")))
        self._capture_calls = os.environ.get(
            "VLLM_STARTUP_ANALYSIS_PYTHON_CALLS", "0").strip().lower() in (
                "1", "true", "yes", "on")
        self._capture_audit = os.environ.get(
            "VLLM_STARTUP_ANALYSIS_AUDIT", "1").strip().lower() in (
                "1", "true", "yes", "on")
        self._capture_nvtx = os.environ.get(
            "VLLM_STARTUP_ANALYSIS_NVTX", "1").strip().lower() in (
                "1", "true", "yes", "on")
        self._call_min_ns = int(float(os.environ.get(
            "VLLM_STARTUP_ANALYSIS_CALL_MIN_MS", "1.0")) * 1_000_000)
        prefixes = os.environ.get(
            "VLLM_STARTUP_ANALYSIS_MODULE_PREFIXES", "vllm,torch._dynamo,torch._inductor")
        self._module_prefixes = tuple(
            item.strip() for item in prefixes.split(",") if item.strip())
        self._previous_profile = None
        self._call_ids = itertools.count(1)
        self._call_starts: dict[tuple[int, int], tuple[
            int, int, Optional[int], Optional[str], str, str, str, int]] = {}
        self._call_stacks: dict[int, list[int]] = {}

    @property
    def enabled(self) -> bool:
        return True

    def start(self) -> None:
        if self._started:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started = True
        self.record("analysis_start", output_dir=str(self.output_dir))
        if self._capture_calls:
            self._previous_profile = sys.getprofile()
            sys.setprofile(self._profile)
            threading.setprofile(self._profile)
        if self._capture_audit:
            sys.addaudithook(self._audit_hook)
        atexit.register(self.flush)

    def record(self, event: str, **payload: Any) -> None:
        if getattr(self._local, "recording", False):
            return
        self._local.recording = True
        try:
            item = {
                "schema": 1,
                "event": event,
                "timestamp_ns": time.perf_counter_ns(),
                "wall_time_ns": time.time_ns(),
                "pid": self.pid,
                "thread_id": threading.get_ident(),
                "thread_name": threading.current_thread().name,
                "stage": _CURRENT_STAGE.get(),
            }
            item.update({key: _safe_value(value) for key, value in payload.items()})
            with self._lock:
                if len(self._events) < self._max_events:
                    self._events.append(item)
                else:
                    self._dropped_events += 1
        finally:
            self._local.recording = False

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.start()
        token = _CURRENT_STAGE.set(name)
        started = time.perf_counter_ns()
        self.record("stage_start", name=name)
        nvtx_pushed = self._nvtx_push(name)
        try:
            yield
        except BaseException as exc:
            self.record("stage_error", name=name, error=repr(exc))
            raise
        finally:
            if nvtx_pushed:
                self._nvtx_pop()
            finished = time.perf_counter_ns()
            self.record("stage_end", name=name, elapsed_ns=finished - started)
            _CURRENT_STAGE.reset(token)

    def state_access(self, operation: str, key: str, **metadata: Any) -> None:
        self.record("state_access", operation=operation, key=key, **metadata)

    def resource(self, resource: str, units: int = 1, **metadata: Any) -> None:
        self.record("resource_evidence", resource=resource, units=units,
                    **metadata)

    def flush(self) -> Optional[Path]:
        if not self._started or self._flushed:
            return self.trace_path if self._flushed else None
        self._flushed = True
        if self._capture_calls:
            sys.setprofile(self._previous_profile)
            threading.setprofile(None)
        self.record("analysis_end", dropped_events=self._dropped_events)
        with self._lock:
            events = list(self._events)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temp = self.trace_path.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True, default=str))
                handle.write("\n")
        os.replace(temp, self.trace_path)
        summary = {
            "schema": "igniteserve-startup-trace-summary-v1",
            "pid": self.pid,
            "trace": str(self.trace_path),
            "event_count": len(events),
            "dropped_events": self._dropped_events,
            "capture_calls": self._capture_calls,
            "capture_audit": self._capture_audit,
            "capture_nvtx": self._capture_nvtx,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return self.trace_path

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        if event not in ("call", "return", "exception"):
            return
        module = str(frame.f_globals.get("__name__", ""))
        if module.startswith("vllm.v1.engine.startup_analysis"):
            return
        if self._module_prefixes and not module.startswith(self._module_prefixes):
            return
        key = (threading.get_ident(), id(frame))
        if event == "call":
            code = frame.f_code
            thread_id = threading.get_ident()
            stack = self._call_stacks.setdefault(thread_id, [])
            call_id = next(self._call_ids)
            parent_call_id = stack[-1] if stack else None
            stack.append(call_id)
            started_ns = time.perf_counter_ns()
            self._call_starts[key] = (
                started_ns, call_id, parent_call_id,
                _CURRENT_STAGE.get(), module,
                getattr(code, "co_qualname", code.co_name),
                code.co_filename, code.co_firstlineno)
            return
        call = self._call_starts.pop(key, None)
        if call is None:
            return
        (started, call_id, parent_call_id, stage, module, qualname, filename,
         lineno) = call
        stack = self._call_stacks.get(threading.get_ident(), [])
        if stack and stack[-1] == call_id:
            stack.pop()
        elif call_id in stack:
            stack.remove(call_id)
        elapsed = time.perf_counter_ns() - started
        if elapsed < self._call_min_ns:
            return
        self.record(
            "call_span",
            call_id=call_id,
            parent_call_id=parent_call_id,
            started_ns=started,
            stage_at_call=stage,
            module=module,
            qualname=qualname,
            filename=filename,
            lineno=lineno,
            elapsed_ns=elapsed,
            outcome="exception" if event == "exception" else "return",
        )

    def _audit_hook(self, event: str, args: tuple[Any, ...]) -> None:
        if not event.startswith(self._AUDIT_PREFIXES):
            return
        if _CURRENT_STAGE.get() is None:
            return
        self.record("audit", audit_event=event,
                    arguments=[_safe_value(value, 256) for value in args[:4]])

    def _nvtx_push(self, name: str) -> bool:
        if not self._capture_nvtx:
            return False
        try:
            from torch.cuda import nvtx
            nvtx.range_push(f"igniteserve.startup::{name}")
            return True
        except Exception as exc:
            self.record("nvtx_unavailable", error=repr(exc))
            self._capture_nvtx = False
            return False

    def _nvtx_pop(self) -> None:
        try:
            from torch.cuda import nvtx
            nvtx.range_pop()
        except Exception as exc:
            self.record("nvtx_pop_error", error=repr(exc))


def get_startup_recorder() -> Optional[StartupAnalysisRecorder]:
    """Return the process recorder when offline analysis is explicitly enabled."""
    output = os.environ.get("VLLM_STARTUP_OFFLINE_ANALYSIS_DIR")
    if not output:
        return None
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = StartupAnalysisRecorder(Path(output))
            _RECORDER.start()
        return _RECORDER


def current_startup_stage() -> Optional[str]:
    return _CURRENT_STAGE.get()
