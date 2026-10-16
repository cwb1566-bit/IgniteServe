import math
import os
import time
import csv
import threading
import uuid
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from vllm.logger import init_logger
from vllm.v1.worker.cudagraph_profile import validate_cudagraph_profile_manifest


logger = init_logger(__name__)


@dataclass
class BatchSizeStats:
    """批次大小统计"""

    size: int
    hit_count: int = 0
    total_tokens: int = 0
    first_access_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    capture_priority: float = 0.0

    @property
    def frequency(self) -> float:
        elapsed = time.time() - self.first_access_time
        return self.hit_count / max(elapsed, 1.0)

    @property
    def throughput_contribution(self) -> float:
        return self.total_tokens / max(self.hit_count, 1)

    @property
    def recency_score(self) -> float:
        elapsed = time.time() - self.last_access_time
        return math.exp(-0.001 * elapsed)


class HotspotAnalyzer:
    """Per-engine observations and capture eligibility; never submits work."""

    def __init__(
        self,
        *,
        min_hits=1,
        output_dir=None,
        auto_plot=False,
        analysis_thresholds=None,
    ):
        self.min_hits = max(1, int(min_hits))
        self._lock = threading.Lock()
        self.stats = {}
        self.actual_stats = {}
        self.total_requests = 0
        self.analysis_thresholds = analysis_thresholds or [1000, 5000, 10000]
        self.triggered_thresholds = set()
        self.auto_plot = auto_plot
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.exporter_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.profile_manifest = None
        self.has_exported_final = False

    def observe(self, raw_shape, target_shape):
        # CSV keys retain registered target shapes for precapture compatibility.
        self.actual_stats[raw_shape] = self.actual_stats.get(raw_shape, 0) + 1
        self.record_batch(target_shape)

    def record_target(self, record):
        """Update the manager-owned record rather than duplicating its counter."""
        record.hit_count += 1
        now = time.time()
        record.first_seen_at = record.first_seen_at or now
        record.last_seen_at = now

    def should_capture(self, record):
        return record.hit_count >= self.min_hits

    def select_precapture(self, registered, profile_dir, manifest, *, adaptive):
        from io import StringIO
        from vllm.compilation.startup_assets import read_text

        registered = sorted(set(registered))
        if os.getenv("VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return registered
        candidates = []
        if os.getenv("VLLM_CUDA_GRAPH_HOTSPOT_CSV"):
            candidates.append(Path(os.environ["VLLM_CUDA_GRAPH_HOTSPOT_CSV"]))
        if os.getenv("VLLM_CUDA_GRAPH_HOTSPOT_DIR"):
            candidates.append(
                Path(os.environ["VLLM_CUDA_GRAPH_HOTSPOT_DIR"])
                / "cuda_graph_hits.csv"
            )
        candidates.extend(
            (
                Path(profile_dir) / "cuda_graph_hits.csv",
                Path("analysis_results/cuda_graph_hits.csv"),
            )
        )
        for path in candidates:
            if not path.is_file():
                continue
            valid, reason = validate_cudagraph_profile_manifest(
                path.parent, manifest
            )
            if not valid:
                logger.warning("Ignoring hotspot profile %s: %s", path, reason)
                continue
            try:
                rows = list(csv.DictReader(StringIO(read_text(path))))
                records = [
                    (
                        int(r["batch_size"]),
                        int(r["hit_count"]),
                        float(r.get("capture_priority", 0)),
                    )
                    for r in rows
                ]
                if any(
                    s <= 0 or h < 0 or not math.isfinite(p)
                    for s, h, p in records
                ):
                    raise ValueError("invalid hotspot record")
                records.sort(key=lambda r: (r[1], r[2]), reverse=True)
                top_k = os.getenv("VLLM_CUDA_GRAPH_HOTSPOT_TOP_K")
                if adaptive and top_k is None:
                    total = sum(h for _, h, _ in records)
                    cumulative = 0
                    selected = []
                    minimum = int(
                        os.getenv("VLLM_CUDA_GRAPH_PRECAPTURE_MIN_HITS", "10")
                    )
                    for size, hits, _ in records:
                        if cumulative * 100 >= total * 99:
                            break
                        cumulative += hits
                        if hits >= minimum and size in registered:
                            selected.append(size)
                    policy = f"offline-top99-then-hit-threshold>={minimum}"
                else:
                    selected = [s for s, _, _ in records if s in registered][
                        : int(top_k or 10)
                    ]
                    policy = f"top-k={top_k or 10}"
                if adaptive:
                    mandatory = {
                        int(s)
                        for s in os.getenv(
                            "VLLM_ADAPTIVE_CUDAGRAPH_MANDATORY_SHAPES", "1,2"
                        ).split(",")
                        if s.strip()
                    }
                    selected.extend((mandatory | {1, 2}) & set(registered))
                if not selected:
                    raise ValueError("no eligible shapes")
                selected = sorted(set(selected))
                logger.info(
                    "Selected CUDA graph shapes using %s: %s", policy, selected
                )
                return selected
            except (ValueError, KeyError, OSError):
                logger.exception(
                    "Invalid hotspot profile %s; using registered shapes", path
                )
                return registered
        logger.info("No compatible hotspot profile; using registered shapes")
        return registered

    def record_batch(self, num_tokens: int):
        """仅负责记录批次 (供 ModelRunner 调用)"""
        with self._lock:  # 如果有多个 GPU worker 并发，加上锁更安全
            self.total_requests += 1

            # 压测重启时，重置落盘状态
            self.has_exported_final = False

            if num_tokens not in self.stats:
                self.stats[num_tokens] = BatchSizeStats(size=num_tokens)

            stat = self.stats[num_tokens]
            stat.hit_count += 1
            stat.total_tokens += num_tokens
            stat.last_access_time = time.time()

            # 更新优先级
            stat.capture_priority = (
                stat.frequency
                * stat.throughput_contribution
                * stat.recency_score
            )

            if self.auto_plot:
                self._check_analysis_trigger()

    def check_and_export(self, is_queue_empty: bool):
        """监控队列状态并决定是否落盘 (供 Scheduler 或 Engine 调用)"""
        if (
            self.output_dir is not None
            and is_queue_empty
            and self.total_requests > 0
            and not self.has_exported_final
        ):
            with self._lock:
                if not self.has_exported_final:  # Double check
                    logger.debug("Exporting hotspot observations for %d batches",
                                 self.total_requests)
                    self.export_to_csv()
                    if self.auto_plot:
                        self.plot_hit_statistics(suffix="_final")
                    self.has_exported_final = True

    def _check_analysis_trigger(self):
        """到达特定请求量时的定期报告"""
        for threshold in self.analysis_thresholds:
            if (
                self.total_requests >= threshold
                and threshold not in self.triggered_thresholds
            ):
                self.triggered_thresholds.add(threshold)
                if self.auto_plot:
                    self.plot_hit_statistics(suffix=f"_at_{threshold}")

    def export_to_csv(
        self, filename: str = "cuda_graph_hits.csv", top_percent: float = 0.99
    ):
        """Export this engine generation and atomically merge all generations."""
        if not self.stats or self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        expected_manifest = self.profile_manifest
        if expected_manifest is not None:
            compatible, reason = validate_cudagraph_profile_manifest(
                self.output_dir, expected_manifest
            )
            if not compatible:
                warning = (str(self.output_dir), reason)
                if getattr(self, "_profile_export_warning", None) != warning:
                    logger.warning(
                        "Not exporting CUDA graph hotspots into incompatible "
                        "directory %s: %s; use a configuration-specific "
                        "VLLM_CUDA_GRAPH_HOTSPOT_DIR.",
                        self.output_dir,
                        reason,
                    )
                    self._profile_export_warning = warning
                return
            self._profile_export_warning = None

        header = [
            "batch_size",
            "hit_count",
            "total_tokens",
            "frequency",
            "capture_priority",
        ]
        base_name, ext = os.path.splitext(filename)
        part_path = self.output_dir / (
            f".{base_name}.part-{self.exporter_id}{ext}"
        )
        lock_path = self.output_dir / f".{base_name}.lock"

        # A process-local analyzer is cumulative. Replacing its own part makes
        # repeated exports idempotent, while distinct engine generations retain
        # separate parts that can be summed after unload/reload.
        part_tmp = part_path.with_suffix(part_path.suffix + ".tmp")
        with open(part_tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for size in sorted(self.stats):
                stat = self.stats[size]
                writer.writerow(
                    [
                        size,
                        stat.hit_count,
                        stat.total_tokens,
                        stat.hit_count / max(self.total_requests, 1),
                        stat.capture_priority,
                    ]
                )
        os.replace(part_tmp, part_path)

        # Multiple EngineCore generations may exit concurrently and target the
        # same workload directory. Serialize aggregation across processes.
        import fcntl

        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            aggregate: Dict[int, Dict[str, float]] = {}
            for candidate in self.output_dir.glob(f".{base_name}.part-*{ext}"):
                with open(candidate, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        size = int(row["batch_size"])
                        entry = aggregate.setdefault(
                            size,
                            {
                                "hit_count": 0,
                                "total_tokens": 0,
                                "capture_priority": 0.0,
                            },
                        )
                        entry["hit_count"] += int(row["hit_count"])
                        entry["total_tokens"] += int(row["total_tokens"])
                        entry["capture_priority"] = max(
                            entry["capture_priority"],
                            float(row["capture_priority"]),
                        )

            total_hits = sum(
                int(entry["hit_count"]) for entry in aggregate.values()
            )
            filepath = self.output_dir / filename
            filepath_tmp = filepath.with_suffix(filepath.suffix + ".tmp")
            with open(filepath_tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for size in sorted(aggregate):
                    entry = aggregate[size]
                    writer.writerow(
                        [
                            size,
                            int(entry["hit_count"]),
                            int(entry["total_tokens"]),
                            entry["hit_count"] / max(total_hits, 1),
                            entry["capture_priority"],
                        ]
                    )
            os.replace(filepath_tmp, filepath)

            sorted_stats = sorted(
                aggregate.items(),
                key=lambda item: (
                    item[1]["hit_count"],
                    item[1]["capture_priority"],
                ),
                reverse=True,
            )

            target_hits = total_hits * top_percent

            top_stats = []
            accumulated_hits = 0

            for size, entry in sorted_stats:
                top_stats.append((size, entry))
                accumulated_hits += int(entry["hit_count"])
                if accumulated_hits >= target_hits:
                    break

            top_filename = f"{base_name}_top{int(top_percent * 100)}{ext}"
            top_filepath = self.output_dir / top_filename
            top_tmp = top_filepath.with_suffix(top_filepath.suffix + ".tmp")

            # Ascending shapes let capture code iterate in reverse and reuse
            # the largest graph's memory pool.
            with open(top_tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for size, entry in sorted(top_stats):
                    writer.writerow(
                        [
                            size,
                            int(entry["hit_count"]),
                            int(entry["total_tokens"]),
                            entry["hit_count"] / max(total_hits, 1),
                            entry["capture_priority"],
                        ]
                    )
            os.replace(top_tmp, top_filepath)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        print(f"[+] CUDA 图全量命中统计已导出至: {filepath}")
        print(
            f"[+] CUDA 图 Top {int(top_percent * 100)}% 优选统计已导出至: {top_filepath}"
        )
        print(
            f"    (有效剔除了长尾数据，共选中 {len(top_stats)} 个核心批次大小)"
        )

    def plot_hit_statistics(self, suffix: str = ""):
        # The serving path disables plotting. Import plotting dependencies only
        # for an explicit offline visualization request.
        import matplotlib.pyplot as plt
        import numpy as np

        """极其干净的学术风分布与 CDF 图"""
        if not self.stats:
            return

        # 整理数据
        sizes = sorted(self.stats.keys())
        counts = [self.stats[s].hit_count for s in sizes]
        cdf = np.cumsum(counts) / sum(counts)

        plt.rcParams.update({"font.size": 12, "axes.labelsize": 14})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.25)

        # --- 子图1：命中频率分布 (柱状图) ---
        ax1.bar(
            sizes,
            counts,
            color="#3F88C5",
            edgecolor="black",
            linewidth=0.5,
            width=max(1, (max(sizes) - min(sizes)) / 100),
        )
        ax1.set_xlabel("Batch Size (Tokens)", fontweight="bold")
        ax1.set_ylabel("Hit Count", fontweight="bold")
        ax1.set_title("CUDA Graph Hit Distribution", fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # --- 子图2：CDF 累积分布曲线 ---
        ax2.plot(sizes, cdf, color="#D5A03A", linewidth=3)
        ax2.fill_between(sizes, 0, cdf, color="#D5A03A", alpha=0.2)
        ax2.axhline(y=0.9, color="red", linestyle=":", label="P90 Threshold")
        ax2.set_xlabel("Batch Size (Tokens)", fontweight="bold")
        ax2.set_ylabel("CDF", fontweight="bold")
        ax2.set_title(
            "Cumulative Distribution Function (CDF)", fontweight="bold"
        )
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower right", frameon=False)
        ax2.grid(True, linestyle="--", alpha=0.6)

        filename = self.output_dir / f"hit_statistics{suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
