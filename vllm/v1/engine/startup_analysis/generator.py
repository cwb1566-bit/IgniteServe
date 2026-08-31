# SPDX-License-Identifier: Apache-2.0
"""Generate a reviewable startup contract from observed startup traces."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .pyspy import parse_speedscope_profiles


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trace event at {path}:{lineno}") from exc
            if isinstance(event, dict):
                events.append(event)
    events.sort(key=lambda item: int(item.get("timestamp_ns", 0)))
    return events


def _seed_nodes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("nodes"), list):
        return list(payload["nodes"])
    plan = payload.get("plan")
    if isinstance(plan, Mapping) and isinstance(plan.get("nodes"), list):
        return list(plan["nodes"])
    return []


def _audit_resource(name: str) -> Optional[str]:
    if name.startswith(("open", "os.listdir", "os.scandir", "mmap.")):
        return "disk_io"
    if name.startswith("subprocess."):
        return "process"
    if name.startswith("socket."):
        return "network_io"
    return None


class StartupContractGenerator:
    """Synthesize candidate nodes, edges, and resources with evidence scores."""

    def __init__(
        self,
        trace_paths: Iterable[Path],
        seed_plan: Optional[Path] = None,
        include_seed_edges: bool = True,
        call_profile_paths: Iterable[Path] = (),
    ) -> None:
        self.trace_paths = [Path(path) for path in trace_paths]
        if not self.trace_paths:
            raise ValueError("At least one startup trace is required")
        self.seed_plan = Path(seed_plan) if seed_plan else None
        self.include_seed_edges = include_seed_edges
        self.call_profile_paths = [Path(path) for path in call_profile_paths]

    def generate(self) -> dict[str, Any]:
        started = time.perf_counter()
        runs = [(path, _read_events(path)) for path in self.trace_paths]
        seed_payload = _read_json(self.seed_plan) if self.seed_plan else {}
        seed_nodes = _seed_nodes(seed_payload)

        stage_durations: dict[str, list[float]] = defaultdict(list)
        stage_occurrences: dict[str, int] = defaultdict(int)
        produced: dict[str, set[str]] = defaultdict(set)
        required: dict[str, set[str]] = defaultdict(set)
        resources: dict[str, dict[str, int]] = defaultdict(dict)
        resource_evidence: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list))
        edges: dict[tuple[str, str], dict[str, Any]] = {}
        ordering_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
        call_stats: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        observed_stages: set[str] = set()

        for path, events in runs:
            last_writer: dict[str, str] = {}
            readers_since_write: dict[str, set[str]] = defaultdict(set)
            for event in events:
                kind = str(event.get("event", ""))
                stage = event.get("stage") or event.get("name")
                if kind == "stage_start" and stage:
                    observed_stages.add(str(stage))
                    stage_occurrences[str(stage)] += 1
                elif kind == "stage_end" and stage:
                    observed_stages.add(str(stage))
                    elapsed = float(event.get("elapsed_ns", 0)) / 1e9
                    stage_durations[str(stage)].append(elapsed)
                elif kind == "state_access" and stage:
                    stage = str(stage)
                    key = str(event.get("key", ""))
                    operation = str(event.get("operation", ""))
                    if not key:
                        continue
                    if operation in ("get", "require", "read"):
                        required[stage].add(key)
                        writer = last_writer.get(key)
                        if writer and writer != stage:
                            self._merge_edge(
                                edges, writer, stage, 0.98,
                                f"observed RAW state dependency {key} in {path.name}")
                        readers_since_write[key].add(stage)
                    elif operation in ("put", "write"):
                        produced[stage].add(key)
                        previous = last_writer.get(key)
                        if previous and previous != stage:
                            candidate_key = (previous, stage, key)
                            ordering_candidates[candidate_key] = {
                                "source": previous,
                                "target": stage,
                                "state": key,
                                "kind": "WAW",
                                "confidence": 0.75,
                                "evidence": [f"two stages write {key} in {path.name}"],
                            }
                        for reader in readers_since_write.get(key, set()):
                            if reader != stage:
                                candidate_key = (reader, stage, key)
                                ordering_candidates[candidate_key] = {
                                    "source": reader,
                                    "target": stage,
                                    "state": key,
                                    "kind": "WAR",
                                    "confidence": 0.65,
                                    "evidence": [
                                        f"write follows an observed read of {key} "
                                        f"in {path.name}"
                                    ],
                                }
                        readers_since_write[key].clear()
                        last_writer[key] = stage
                elif kind == "audit" and stage:
                    stage = str(stage)
                    resource = _audit_resource(str(event.get("audit_event", "")))
                    if resource:
                        resources[stage][resource] = max(
                            resources[stage].get(resource, 0), 1)
                        resource_evidence[stage][resource].append(
                            f"audit:{event.get('audit_event')}")
                elif kind == "resource_evidence" and stage:
                    stage = str(stage)
                    resource = str(event.get("resource", ""))
                    if resource:
                        units = max(1, int(event.get("units", 1)))
                        resources[stage][resource] = max(
                            resources[stage].get(resource, 0), units)
                        source = str(event.get("source", "runtime"))
                        resource_evidence[stage][resource].append(source)
                elif kind == "call_span":
                    module = str(event.get("module", ""))
                    qualname = str(event.get("qualname", ""))
                    filename = str(event.get("filename", ""))
                    lineno = int(event.get("lineno", 0))
                    key = (module, qualname, filename, lineno)
                    stat = call_stats.setdefault(key, {
                        "module": module,
                        "qualname": qualname,
                        "filename": filename,
                        "lineno": lineno,
                        "elapsed_s": [],
                        "stages": set(),
                    })
                    stat["elapsed_s"].append(
                        float(event.get("elapsed_ns", 0)) / 1e9)
                    observed_stage = event.get("stage_at_call") or event.get("stage")
                    if observed_stage:
                        stat["stages"].add(str(observed_stage))

        seed_by_name = {
            str(node.get("name")): node for node in seed_nodes if node.get("name")
        }
        all_stages = observed_stages | set(seed_by_name)
        if self.include_seed_edges:
            for target, node in seed_by_name.items():
                for source in node.get("dependencies", []):
                    self._merge_edge(
                        edges, str(source), target, 0.50,
                        "conservative edge from the existing adapter contract")

        contract_nodes: list[dict[str, Any]] = []
        for name in sorted(all_stages):
            seed = seed_by_name.get(name, {})
            durations = stage_durations.get(name, [])
            node_resources = dict(resources.get(name, {}))
            for resource, units in dict(seed.get("resources", {})).items():
                if resource not in node_resources:
                    node_resources[str(resource)] = int(units)
                    resource_evidence[name][str(resource)].append(
                        "conservative adapter annotation")
            node_edges = sorted(
                source for (source, target) in edges if target == name)
            edge_confidences = [
                float(edges[(source, name)]["confidence"])
                for source in node_edges
            ]
            contract_nodes.append({
                "name": name,
                "dependencies": node_edges,
                "required_state": sorted(required.get(name, set())),
                "produced_state": sorted(produced.get(name, set())),
                "resources": node_resources,
                "estimated_cost_s": (
                    float(statistics.median(durations)) if durations else
                    float(seed.get("estimated_cost_s", 0.0))),
                "confidence": min(edge_confidences, default=0.5),
                "category": str(seed.get("category", "observed_startup")),
                "observations": len(durations),
                "resource_evidence": {
                    resource: sorted(set(evidence))
                    for resource, evidence in resource_evidence[name].items()
                },
            })

        edge_payload = sorted(
            ({"source": source, "target": target, **details}
             for (source, target), details in edges.items()),
            key=lambda item: (item["target"], item["source"]),
        )
        calls = []
        for stat in call_stats.values():
            elapsed = stat.pop("elapsed_s")
            stages = stat.pop("stages")
            calls.append({
                **stat,
                "count": len(elapsed),
                "total_s": sum(elapsed),
                "median_s": statistics.median(elapsed),
                "max_s": max(elapsed),
                "observed_in_stages": sorted(stages),
            })
        calls.sort(key=lambda item: item["total_s"], reverse=True)
        sampled_calls = parse_speedscope_profiles(self.call_profile_paths)
        if sampled_calls:
            calls = sampled_calls + calls

        return {
            "schema": "igniteserve-startup-contract-v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_traces": [str(path) for path in self.trace_paths],
            "call_profiles": [str(path) for path in self.call_profile_paths],
            "seed_plan": str(self.seed_plan) if self.seed_plan else None,
            "generation_elapsed_s": time.perf_counter() - started,
            "nodes": contract_nodes,
            "edges": edge_payload,
            "ordering_candidates": sorted(
                ordering_candidates.values(),
                key=lambda item: (item["target"], item["source"], item["state"])),
            "candidate_stage_boundaries": calls[:500],
            "summary": {
                "trace_count": len(runs),
                "node_count": len(contract_nodes),
                "edge_count": len(edge_payload),
                "state_derived_edge_count": sum(
                    1 for edge in edge_payload if edge["confidence"] >= 0.9),
                "conservative_edge_count": sum(
                    1 for edge in edge_payload if edge["confidence"] < 0.9),
                "ordering_candidate_count": len(ordering_candidates),
                "call_boundary_candidate_count": len(calls),
            },
        }

    def write(self, output: Path) -> dict[str, Any]:
        contract = self.generate()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(contract, indent=2, sort_keys=True),
                          encoding="utf-8")
        return contract

    @staticmethod
    def _merge_edge(
        edges: dict[tuple[str, str], dict[str, Any]],
        source: str,
        target: str,
        confidence: float,
        evidence: str,
    ) -> None:
        if source == target:
            return
        key = (source, target)
        details = edges.setdefault(key, {"confidence": 0.0, "evidence": []})
        details["confidence"] = max(float(details["confidence"]), confidence)
        if evidence not in details["evidence"]:
            details["evidence"].append(evidence)
