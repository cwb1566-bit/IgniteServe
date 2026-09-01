# SPDX-License-Identifier: Apache-2.0
"""Build root-level startup discovery evidence without a seed DAG."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .pyspy import parse_speedscope_profiles


def _events(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            result.append(json.loads(line))
    return result


def _source_digest(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_resource(event: str) -> Optional[str]:
    if event.startswith(("open", "os.listdir", "os.scandir", "mmap.")):
        return "disk_io"
    if event.startswith("subprocess."):
        return "process"
    if event.startswith("socket."):
        return "network_io"
    return None


class StartupDiscoveryBuilder:
    """Aggregate native sequential starts into boundary and resource evidence."""

    def __init__(
        self,
        trace_paths: Iterable[Path],
        *,
        call_profile_paths: Iterable[Path] = (),
        nsight_summary_paths: Iterable[Path] = (),
        source_path: Optional[Path] = None,
        root_stage: str = "engine_startup.root",
    ) -> None:
        self.trace_paths = [Path(path) for path in trace_paths]
        if not self.trace_paths:
            raise ValueError("At least one root startup trace is required")
        self.call_profile_paths = [Path(path) for path in call_profile_paths]
        self.nsight_summary_paths = [Path(path) for path in nsight_summary_paths]
        self.source_path = Path(source_path) if source_path else None
        self.root_stage = root_stage

    def generate(self) -> dict[str, Any]:
        started = time.perf_counter()
        root_durations = []
        call_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = (
            defaultdict(list))
        call_by_run: list[dict[int, dict[str, Any]]] = []
        original_orders: list[list[dict[str, Any]]] = []
        call_edges: Counter[tuple[tuple[str, str, int], tuple[str, str, int]]] = (
            Counter())
        audit_events = []
        compatibility_observations = []
        resources: Counter[str] = Counter()

        for trace_path in self.trace_paths:
            events = _events(trace_path)
            run_calls: dict[int, dict[str, Any]] = {}
            for event in events:
                stage = event.get("stage") or event.get("name")
                kind = event.get("event")
                if (kind == "stage_end" and stage == self.root_stage):
                    root_durations.append(float(event.get("elapsed_ns", 0)) / 1e9)
                elif kind == "call_span" and (
                        event.get("stage_at_call") == self.root_stage):
                    key = (
                        str(event.get("module", "")),
                        str(event.get("qualname", "")),
                        str(event.get("filename", "")),
                        int(event.get("lineno", 0)),
                    )
                    call_groups[key].append(event)
                    call_id = event.get("call_id")
                    if call_id is not None:
                        run_calls[int(call_id)] = event
                elif kind == "audit" and stage == self.root_stage:
                    audit_events.append({
                        "trace": str(trace_path),
                        "audit_event": str(event.get("audit_event", "")),
                        "arguments": list(event.get("arguments", [])),
                    })
                    resource = _audit_resource(str(event.get("audit_event", "")))
                    if resource:
                        resources[resource] += 1
                elif kind == "startup_compatibility":
                    facts = event.get("facts")
                    if isinstance(facts, dict):
                        compatibility_observations.append(dict(facts))
            call_by_run.append(run_calls)
            ordered = sorted(run_calls.values(), key=lambda item: int(
                item.get("started_ns", int(item.get("timestamp_ns", 0))
                         - int(item.get("elapsed_ns", 0)))))
            original_orders.append([{
                "call_id": item.get("call_id"),
                "parent_call_id": item.get("parent_call_id"),
                "qualname": str(item.get("qualname", "")),
                "filename": str(item.get("filename", "")),
                "lineno": int(item.get("lineno", 0)),
                "started_ns": int(item.get(
                    "started_ns", int(item.get("timestamp_ns", 0))
                    - int(item.get("elapsed_ns", 0)))),
                "elapsed_s": float(item.get("elapsed_ns", 0)) / 1e9,
                "thread_id": item.get("thread_id"),
                "thread_name": str(item.get("thread_name", "")),
            } for item in ordered])

        for run_calls in call_by_run:
            for event in run_calls.values():
                parent = event.get("parent_call_id")
                if parent is None or int(parent) not in run_calls:
                    continue
                parent_event = run_calls[int(parent)]
                parent_key = (
                    str(parent_event.get("qualname", "")),
                    str(parent_event.get("filename", "")),
                    int(parent_event.get("lineno", 0)),
                )
                child_key = (
                    str(event.get("qualname", "")),
                    str(event.get("filename", "")),
                    int(event.get("lineno", 0)),
                )
                call_edges[(parent_key, child_key)] += 1

        calls = []
        for (module, qualname, filename, lineno), observed in call_groups.items():
            elapsed = [float(item.get("elapsed_ns", 0)) / 1e9 for item in observed]
            calls.append({
                "module": module,
                "qualname": qualname,
                "filename": filename,
                "lineno": lineno,
                "count": len(elapsed),
                "total_s": sum(elapsed),
                "median_s": statistics.median(elapsed),
                "max_s": max(elapsed),
                "threads": sorted({str(item.get("thread_name", ""))
                                   for item in observed}),
                "evidence": "sys.setprofile",
            })
        calls.sort(key=lambda item: item["total_s"], reverse=True)

        sampled = parse_speedscope_profiles(
            self.call_profile_paths,
            excluded_file_markers=("/vllm/usage/",),
            excluded_qualnames=("_report_usage_worker", "_report_continuous_usage"),
            required_stack_qualnames=(
                "EngineCore.__init__", "_initialize_engine_core"),
            max_profile_s=None,
        ) if self.call_profile_paths else []

        nsight = []
        for path in self.nsight_summary_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = dict(payload.get("stages", {}).get(self.root_stage, {}))
            if root:
                nsight.append({"source": str(path), **root})
                for resource in root.get("resources", []):
                    resources[str(resource)] += 1

        compatibility = {}
        if compatibility_observations:
            common_keys = set(compatibility_observations[0])
            for facts in compatibility_observations[1:]:
                common_keys &= set(facts)
            for key in sorted(common_keys - {"cache_state", "model"}):
                values = [facts[key] for facts in compatibility_observations]
                if all(value == values[0] for value in values[1:]):
                    compatibility[key] = values[0]

        return {
            "schema": "igniteserve-startup-discovery-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "root_stage": self.root_stage,
            "source": ({
                "path": str(self.source_path),
                "sha256": _source_digest(self.source_path),
            } if self.source_path else None),
            "compatibility": compatibility,
            "compatibility_observations": compatibility_observations,
            "traces": [str(path) for path in self.trace_paths],
            "root_cost_s": {
                "observations": len(root_durations),
                "median": statistics.median(root_durations)
                if root_durations else 0.0,
                "minimum": min(root_durations, default=0.0),
                "maximum": max(root_durations, default=0.0),
            },
            "call_boundaries": calls,
            "original_call_orders": original_orders,
            "sampled_call_boundaries": sampled,
            "call_edges": [{
                "caller": {"qualname": parent[0], "filename": parent[1],
                           "lineno": parent[2]},
                "callee": {"qualname": child[0], "filename": child[1],
                           "lineno": child[2]},
                "observations": count,
            } for (parent, child), count in call_edges.most_common()],
            "audit_events": audit_events,
            "resource_evidence": dict(sorted(resources.items())),
            "nsight_root_evidence": nsight,
            "generation_elapsed_s": time.perf_counter() - started,
        }

    def write(self, output: Path) -> dict[str, Any]:
        payload = self.generate()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True),
                          encoding="utf-8")
        return payload
