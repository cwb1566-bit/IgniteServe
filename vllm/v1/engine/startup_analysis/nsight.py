# SPDX-License-Identifier: Apache-2.0
"""Import stage-scoped CUDA evidence from an Nsight Systems SQLite export."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional


_PREFIX = "igniteserve.startup::"


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def _string_map(connection: sqlite3.Connection,
                tables: set[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }


def _stage_ranges(
    connection: sqlite3.Connection,
    tables: set[str],
    strings: dict[int, str],
) -> list[dict[str, Any]]:
    if "NVTX_EVENTS" not in tables:
        return []
    cols = _columns(connection, "NVTX_EVENTS")
    selected = [name for name in (
        "start", "end", "text", "textId", "globalTid", "endGlobalTid")
                if name in cols]
    rows = connection.execute(
        "SELECT " + ", ".join(selected) + " FROM NVTX_EVENTS "
        "WHERE start IS NOT NULL AND end IS NOT NULL")
    ranges = []
    for raw in rows:
        row = dict(zip(selected, raw))
        text = row.get("text")
        if not text and row.get("textId") is not None:
            text = strings.get(int(row["textId"]), "")
        text = str(text or "")
        if not text.startswith(_PREFIX):
            continue
        ranges.append({
            "stage": text[len(_PREFIX):],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "global_tid": row.get("globalTid"),
        })
    return ranges


def _rows_in_range(
    connection: sqlite3.Connection,
    table: str,
    start: int,
    end: int,
    columns: Iterable[str],
    global_tid: Optional[int] = None,
) -> list[dict[str, Any]]:
    available = _columns(connection, table)
    selected = [name for name in columns if name in available]
    if "start" not in selected:
        selected.insert(0, "start")
    conditions = ["start >= ?", "start <= ?"]
    params: list[Any] = [start, end]
    if global_tid is not None and "globalTid" in available:
        conditions.append("globalTid = ?")
        params.append(global_tid)
        if "globalTid" not in selected:
            selected.append("globalTid")
    query = ("SELECT " + ", ".join(selected) + f' FROM "{table}" WHERE ' +
             " AND ".join(conditions))
    return [dict(zip(selected, row))
            for row in connection.execute(query, params)]


def extract_nsight_stage_resources(sqlite_path: Path) -> dict[str, Any]:
    """Return resource evidence correlated through CUDA API correlation IDs."""
    started = time.perf_counter()
    connection = sqlite3.connect(str(sqlite_path))
    try:
        tables = _tables(connection)
        strings = _string_map(connection, tables)
        ranges = _stage_ranges(connection, tables, strings)
        stages: dict[str, dict[str, Any]] = {}
        runtime_table = "CUPTI_ACTIVITY_KIND_RUNTIME"
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
        memcpy_table = "CUPTI_ACTIVITY_KIND_MEMCPY"
        memset_table = "CUPTI_ACTIVITY_KIND_MEMSET"
        sync_table = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"

        for stage_range in ranges:
            name = stage_range["stage"]
            stage = stages.setdefault(name, {
                "range_count": 0,
                "range_duration_ns": 0,
                "cuda_api_calls": 0,
                "kernel_count": 0,
                "kernel_duration_ns": 0,
                "memcpy_count": 0,
                "memcpy_bytes": 0,
                "memcpy_duration_ns": 0,
                "memset_count": 0,
                "synchronization_count": 0,
                "stream_ids": set(),
                "context_ids": set(),
                "device_ids": set(),
                "api_names": set(),
                "resources": set(),
            })
            stage["range_count"] += 1
            stage["range_duration_ns"] += (
                stage_range["end"] - stage_range["start"])
            runtime_rows: list[dict[str, Any]] = []
            if runtime_table in tables:
                runtime_rows = _rows_in_range(
                    connection, runtime_table, stage_range["start"],
                    stage_range["end"],
                    ("start", "end", "correlationId", "nameId", "globalTid"),
                    stage_range.get("global_tid"))
            correlations = {
                int(row["correlationId"])
                for row in runtime_rows if row.get("correlationId") is not None
            }
            stage["cuda_api_calls"] += len(runtime_rows)
            if runtime_rows:
                stage["resources"].add("cuda_runtime")
            for row in runtime_rows:
                if row.get("nameId") is not None:
                    api_name = strings.get(int(row["nameId"]), "")
                    if api_name:
                        stage["api_names"].add(api_name)
                        lowered = api_name.lower()
                        if "synchron" in lowered or "wait" in lowered:
                            stage["resources"].add("gpu_sync")
                        if "malloc" in lowered or "alloc" in lowered:
                            stage["resources"].add("gpu_memory")

            for table, count_key, duration_key, resource in (
                (kernel_table, "kernel_count", "kernel_duration_ns", "gpu_compute"),
                (memcpy_table, "memcpy_count", "memcpy_duration_ns", "copy_engine"),
            ):
                if table not in tables or not correlations:
                    continue
                cols = _columns(connection, table)
                selected = [column for column in (
                    "start", "end", "correlationId", "streamId", "contextId",
                    "deviceId", "bytes") if column in cols]
                placeholders = ",".join("?" for _ in correlations)
                query = ("SELECT " + ", ".join(selected) + f' FROM "{table}" '
                         f"WHERE correlationId IN ({placeholders})")
                rows = [dict(zip(selected, row)) for row in connection.execute(
                    query, sorted(correlations))]
                stage[count_key] += len(rows)
                stage[duration_key] += sum(
                    max(0, int(row.get("end", 0)) - int(row.get("start", 0)))
                    for row in rows)
                if rows:
                    stage["resources"].add(resource)
                if table == memcpy_table:
                    stage["memcpy_bytes"] += sum(
                        int(row.get("bytes", 0) or 0) for row in rows)
                for row in rows:
                    for column, target in (
                        ("streamId", "stream_ids"),
                        ("contextId", "context_ids"),
                        ("deviceId", "device_ids"),
                    ):
                        if row.get(column) is not None:
                            stage[target].add(int(row[column]))

            if memset_table in tables and correlations:
                cols = _columns(connection, memset_table)
                if "correlationId" in cols:
                    placeholders = ",".join("?" for _ in correlations)
                    count = connection.execute(
                        f'SELECT COUNT(*) FROM "{memset_table}" '
                        f"WHERE correlationId IN ({placeholders})",
                        sorted(correlations)).fetchone()[0]
                    stage["memset_count"] += int(count)
                    if count:
                        stage["resources"].add("gpu_memory")

            if sync_table in tables:
                sync_rows = _rows_in_range(
                    connection, sync_table, stage_range["start"],
                    stage_range["end"], ("start", "end", "streamId"))
                stage["synchronization_count"] += len(sync_rows)
                if sync_rows:
                    stage["resources"].add("gpu_sync")

        for stage in stages.values():
            for key in (
                "stream_ids", "context_ids", "device_ids", "api_names",
                "resources"):
                stage[key] = sorted(stage[key])
        return {
            "schema": "igniteserve-nsight-stage-resources-v1",
            "source": str(sqlite_path),
            "analysis_elapsed_s": time.perf_counter() - started,
            "stage_count": len(stages),
            "stages": stages,
        }
    finally:
        connection.close()


def append_nsight_resource_events(
    trace_path: Path,
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """Copy a startup trace and append stage-scoped resource evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        with trace_path.open("r", encoding="utf-8") as source:
            for line in source:
                output.write(line)
        timestamp = time.perf_counter_ns()
        for stage_name, stage in analysis.get("stages", {}).items():
            for resource in stage.get("resources", []):
                event = {
                    "schema": 1,
                    "event": "resource_evidence",
                    "timestamp_ns": timestamp,
                    "wall_time_ns": time.time_ns(),
                    "pid": 0,
                    "thread_id": 0,
                    "thread_name": "nsight-import",
                    "stage": stage_name,
                    "resource": resource,
                    "units": 1,
                    "source": "nsight-cupti",
                }
                output.write(json.dumps(event, sort_keys=True))
                output.write("\n")
