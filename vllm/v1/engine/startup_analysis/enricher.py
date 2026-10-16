# SPDX-License-Identifier: Apache-2.0
"""Attach node-level runtime evidence to an extracted startup contract."""

from __future__ import annotations

import copy
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    events = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def _audit_resource(event: str) -> Optional[str]:
    if event.startswith(("open", "os.listdir", "os.scandir", "mmap.")):
        return "disk_io"
    if event.startswith("subprocess."):
        return "process"
    if event.startswith("socket."):
        return "network_io"
    return None


class StartupEvidenceEnricher:
    """Create cache-state variants from line/NVTX-attributed startup traces."""

    def __init__(
        self,
        contract_path: Path,
        traces_by_variant: Mapping[str, Iterable[Path]],
        *,
        matches_by_variant: Optional[Mapping[str, Mapping[str, Any]]] = None,
        nsight_by_variant: Optional[Mapping[str, Iterable[Path]]] = None,
    ) -> None:
        self.contract_path = Path(contract_path)
        self.contract = _read_json(self.contract_path)
        if self.contract.get("schema") != "igniteserve-startup-contract-v2":
            raise ValueError("enrichment requires a v2 startup contract")
        self.traces_by_variant = {
            str(name): [Path(path) for path in paths]
            for name, paths in traces_by_variant.items()
        }
        self.matches_by_variant = {
            str(name): dict(match)
            for name, match in (matches_by_variant or {}).items()
        }
        self.nsight_by_variant = {
            str(name): [Path(path) for path in paths]
            for name, paths in (nsight_by_variant or {}).items()
        }

    def _evidence(self, name: str, node_names: set[str]) -> dict[str, Any]:
        events = _read_events(self.traces_by_variant[name])
        costs: dict[str, list[float]] = defaultdict(list)
        resources: dict[str, set[str]] = defaultdict(set)
        puts: dict[str, list[str]] = defaultdict(list)
        requires: dict[str, list[str]] = defaultdict(list)
        for event in events:
            stage = str(event.get("stage") or event.get("name") or "")
            if stage not in node_names:
                continue
            kind = event.get("event")
            if kind == "stage_end":
                costs[stage].append(float(event.get("elapsed_ns", 0)) / 1e9)
            elif kind == "audit":
                resource = _audit_resource(str(event.get("audit_event", "")))
                if resource:
                    resources[stage].add(resource)
            elif kind == "resource_evidence":
                resource = str(event.get("resource", ""))
                if resource:
                    resources[stage].add(resource)
            elif kind == "state_access":
                key = str(event.get("key", ""))
                operation = str(event.get("operation", ""))
                if key and operation == "put":
                    puts[key].append(stage)
                elif key and operation in ("get", "require"):
                    requires[key].append(stage)

        nsight_stages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in self.nsight_by_variant.get(name, []):
            payload = _read_json(path)
            for stage, evidence in payload.get("stages", {}).items():
                if stage in node_names:
                    nsight_stages[stage].append(dict(evidence))
                    resources[stage].update(
                        str(item) for item in evidence.get("resources", []))

        return {
            "costs": costs,
            "resources": resources,
            "puts": puts,
            "requires": requires,
            "nsight": nsight_stages,
        }

    def generate(self) -> dict[str, Any]:
        started = time.perf_counter()
        base_variant = self.contract.get("variants", [{}])[0]
        base_nodes = list(base_variant.get("nodes", []))
        node_names = {str(node.get("name", "")) for node in base_nodes}
        variants = []
        review_constraints = []

        for variant_name, paths in self.traces_by_variant.items():
            if not paths:
                raise ValueError(f"variant {variant_name!r} has no traces")
            evidence = self._evidence(variant_name, node_names)
            nodes = copy.deepcopy(base_nodes)
            edges = copy.deepcopy(list(base_variant.get("edges", [])))
            edge_keys = {
                (str(edge.get("source")), str(edge.get("target")),
                 str(edge.get("state", ""))) for edge in edges
            }
            produced_by: dict[str, str] = {}
            for key, producers in evidence["puts"].items():
                for producer in producers:
                    produced_by[key] = producer
            for key, consumers in evidence["requires"].items():
                producer = produced_by.get(key)
                if not producer:
                    continue
                for consumer in consumers:
                    edge_key = (producer, consumer, key)
                    if edge_key not in edge_keys:
                        edges.append({
                            "source": producer,
                            "target": consumer,
                            "kind": "runtime-state-raw",
                            "state": key,
                            "confidence": 0.98,
                            "protected": True,
                        })
                        edge_keys.add(edge_key)

            dependencies: dict[str, set[str]] = {
                str(node["name"]): set(node.get("dependencies", []))
                for node in nodes
            }
            for edge in edges:
                dependencies[str(edge["target"])].add(str(edge["source"]))

            for node in nodes:
                stage = str(node["name"])
                observations = list(evidence["costs"].get(stage, []))
                if observations:
                    ordered = sorted(observations)
                    node["estimated_cost_s"] = statistics.median(ordered)
                    node["cost_distribution_s"] = {
                        "observations": len(ordered),
                        "minimum": ordered[0],
                        "median": statistics.median(ordered),
                        "maximum": ordered[-1],
                    }
                discovered_resources = sorted(evidence["resources"].get(
                    stage, set()))
                node["resources"] = {
                    resource: max(1, int(node.get("resources", {}).get(
                        resource, 1))) for resource in discovered_resources
                }
                node["dependencies"] = sorted(dependencies[stage])
                produced = set(node.get("produced_state", []))
                required = set(node.get("required_state", []))
                for key, producers in evidence["puts"].items():
                    if stage in producers:
                        produced.add(key)
                for key, consumers in evidence["requires"].items():
                    if stage in consumers:
                        required.add(key)
                node["produced_state"] = sorted(produced)
                node["required_state"] = sorted(required)
                nsight = evidence["nsight"].get(stage, [])
                if nsight:
                    node["gpu_evidence"] = nsight
                if (node.get("category") == "opaque"
                        and set(discovered_resources)
                        & {"cuda_runtime", "gpu_compute", "copy_engine",
                           "gpu_memory", "gpu_sync", "process"}):
                    review_constraints.append({
                        "id": f"opaque-resource:{variant_name}:{stage}",
                        "mandatory": True,
                        "status": "pending",
                        "variant": variant_name,
                        "node": stage,
                        "resources": discovered_resources,
                        "reason": "opaque source region owns externally visible resources",
                    })

            variants.append({
                "name": variant_name,
                "match": dict(self.matches_by_variant.get(
                    variant_name, {"cache_state": variant_name})),
                "source_traces": [str(path) for path in paths],
                "nodes": nodes,
                "edges": edges,
            })

        result = copy.deepcopy(self.contract)
        result["status"] = "candidate"
        result["variants"] = variants
        result["evidence_generation"] = {
            "source_contract": str(self.contract_path),
            "elapsed_s": time.perf_counter() - started,
        }
        result["manual_review_suggestions"] = {
            "opaque_constraints": review_constraints,
        }
        return result

    def write(self, output: Path, *, review_dir: Optional[Path] = None
              ) -> dict[str, Any]:
        payload = self.generate()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True),
                          encoding="utf-8")
        if review_dir is not None:
            review_path = Path(review_dir) / "opaque_constraints.json"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            existing = ({"schema": "igniteserve-manual-review-v1",
                         "kind": "opaque_constraints", "items": []})
            if review_path.exists():
                existing = _read_json(review_path)
            known = {str(item.get("id")) for item in existing.get("items", [])}
            for item in payload.get("manual_review_suggestions", {}).get(
                    "opaque_constraints", []):
                if str(item.get("id")) not in known:
                    existing.setdefault("items", []).append(item)
            review_path.write_text(json.dumps(existing, indent=2,
                                              sort_keys=True),
                                   encoding="utf-8")
        return payload
