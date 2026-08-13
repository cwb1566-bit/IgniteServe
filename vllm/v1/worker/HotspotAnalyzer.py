import math
import os
import time
import csv
import threading
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from vllm.compilation.startup_resources import resource_dir


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


@dataclass
class AnalysisReport:
    """精简版分析报告"""
    timestamp: float
    total_requests: int
    unique_sizes: int
    top_k_stats: List[tuple[int, BatchSizeStats]]

    def __str__(self) -> str:
        res = (
            f"\n{'=' * 60}\n"
            f" CUDA Graph 热点批次报告 | 总请求: {self.total_requests} | 唯一批次: {self.unique_sizes}\n"
            f"{'-' * 60}\n"
            f"{'排名':<6}{'批次大小':<12}{'命中次数':<12}{'频率':<12}\n"
            f"{'-' * 60}\n"
        )
        for i, (size, stat) in enumerate(self.top_k_stats[:10]):
            freq = stat.hit_count / max(self.total_requests, 1)
            res += f"{i + 1:<6}{size:<12}{stat.hit_count:<12}{freq:<12.4f}\n"
        res += f"{'=' * 60}\n"
        return res


class HotspotAnalyzer:
    """CUDA 图热点批次分析器 (全局单例版)"""

    _instance = None
    _lock = threading.Lock()  # 线程安全锁，防止并发初始化引发的竞争
    _initialized = False  # 防止 __init__ 被多次调用

    def __new__(cls, *args, **kwargs):
        """控制实例的创建，确保全局唯一"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(HotspotAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(
            self,
            window_size: int = 10000,
            capture_threshold: float = 0.05,
            analysis_thresholds: Optional[List[int]] = None,
            auto_plot: bool = True,
            output_dir: str = "./hotspot_analysis"
    ):
        """控制初始化，确保只执行一次"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # 初始化核心变量
            self.stats: Dict[int, BatchSizeStats] = {}
            self.recent_sizes: deque = deque(maxlen=window_size)
            self.total_requests = 0
            self.capture_threshold = capture_threshold

            self.analysis_thresholds = analysis_thresholds or [1000, 5000, 10000]
            self.triggered_thresholds = set()
            self.pending_captures = set() # 记录尚未捕获的热点批次
            self.auto_plot = auto_plot
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.has_exported_final = False  # 避免压测结束后重复疯狂落盘

            self._initialized = True

    # 触发条件---命中次数超过最小阈值，或者频率超过阈值。
    def should_capture(self, num_tokens: int) -> bool:
        with self._lock:
            if num_tokens not in self.stats:
                return False

            # 如果已经在待捕获队列里了，就不需要再触发
            if num_tokens in self.pending_captures:
                return False

            stat = self.stats[num_tokens]
            # 这里的逻辑可以自定义，例如：命中超过 10 次，或者频率大于预设阈值
            MIN_HITS_REQUIRED = 10
            # if stat.hit_count >= MIN_HITS_REQUIRED or stat.frequency >= self.capture_threshold:
            if stat.hit_count >= MIN_HITS_REQUIRED:
                return True
            return False

    def mark_for_capture(self, num_tokens: int):
        """将热点批次加入待捕获队列"""
        with self._lock:
            self.pending_captures.add(num_tokens)

    def get_and_clear_pending_captures(self) -> list[int]:
        """获取所有待捕获的批次，并清空队列（移交给 Graph_Manager）"""
        with self._lock:
            if not self.pending_captures:
                return []
            pending_list = list(self.pending_captures)
            self.pending_captures.clear()
            return pending_list

    def record_batch(self, num_tokens: int):
        """仅负责记录批次 (供 ModelRunner 调用)"""
        with self._lock:  # 如果有多个 GPU worker 并发，加上锁更安全
            self.total_requests += 1
            self.recent_sizes.append(num_tokens)

            # 压测重启时，重置落盘状态
            self.has_exported_final = False

            if num_tokens not in self.stats:
                self.stats[num_tokens] = BatchSizeStats(size=num_tokens)

            stat = self.stats[num_tokens]
            stat.hit_count += 1
            stat.total_tokens += num_tokens
            stat.last_access_time = time.time()

            # 更新优先级
            stat.capture_priority = (stat.frequency * stat.throughput_contribution * stat.recency_score)

            self._check_analysis_trigger()

    def check_and_export(self, is_queue_empty: bool):
        """监控队列状态并决定是否落盘 (供 Scheduler 或 Engine 调用)"""
        if is_queue_empty and self.total_requests > 0 and not self.has_exported_final:
            with self._lock:
                if not self.has_exported_final:  # Double check
                    print(f"\n[+] 推理任务队列已清空，累计捕获 {self.total_requests} 个请求，开始生成全量分析数据...")
                    self.export_to_csv()
                    if self.auto_plot:
                        self.plot_hit_statistics(suffix="_final")
                    self.has_exported_final = True

    def _check_analysis_trigger(self):
        """到达特定请求量时的定期报告"""
        for threshold in self.analysis_thresholds:
            if self.total_requests >= threshold and threshold not in self.triggered_thresholds:
                self.triggered_thresholds.add(threshold)
                # print(self.analyze_and_report())
                if self.auto_plot:
                    self.plot_hit_statistics(suffix=f"_at_{threshold}")

    def analyze_and_report(self, top_k: int = 10) -> AnalysisReport:
        top_k_stats = sorted(self.stats.items(), key=lambda x: x[1].hit_count, reverse=True)[:top_k]
        return AnalysisReport(
            timestamp=time.time(),
            total_requests=self.total_requests,
            unique_sizes=len(self.stats),
            top_k_stats=top_k_stats
        )

    def export_to_csv(self, filename: str = "cuda_graph_hits.csv", top_percent: float = 0.99):
        """将命中统计落盘为 CSV，同时额外导出一个占据前 N% 命中总数的优选批次集合"""
        import os
        import csv

        if not self.stats:
            return

        # ==========================================
        # 1. 导出全量 CSV 数据 (原始逻辑)
        # ==========================================
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_size', 'hit_count', 'total_tokens', 'frequency', 'capture_priority'])

            for size in sorted(self.stats.keys()):
                stat = self.stats[size]
                freq = stat.hit_count / max(self.total_requests, 1)
                writer.writerow([size, stat.hit_count, stat.total_tokens, freq, stat.capture_priority])

        print(f"[+] CUDA 图全量命中统计已导出至: {filepath}")

        # ==========================================
        # 2. 核心逻辑：提取覆盖 Top N% 命中的最小批次集合
        # ==========================================
        # 最优优先级排序方案：
        # 第一关键字：hit_count (命中次数降序)
        # 第二关键字：capture_priority (综合了吞吐量贡献和时间最近性，用于同命中次数时的平局决胜)
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: (s.hit_count, s.capture_priority),
            reverse=True
        )

        total_hits = sum(s.hit_count for s in sorted_stats)
        target_hits = total_hits * top_percent

        top_stats = []
        accumulated_hits = 0

        for stat in sorted_stats:
            top_stats.append(stat)
            accumulated_hits += stat.hit_count
            # 当累加的命中次数达到目标覆盖率时，停止收集
            if accumulated_hits >= target_hits:
                break

        # ==========================================
        # 3. 将达到目标覆盖率的数据导出为新的 CSV
        # ==========================================
        base_name, ext = os.path.splitext(filename)
        top_filename = f"{base_name}_top{int(top_percent * 100)}{ext}"
        top_filepath = self.output_dir / top_filename

        # 写入时，按 batch_size 从小到大升序排列，方便 vLLM 底层预捕获时 reversed 从大到小复用显存
        with open(top_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_size', 'hit_count', 'total_tokens', 'frequency', 'capture_priority'])

            for stat in sorted(top_stats, key=lambda s: s.size):
                freq = stat.hit_count / max(self.total_requests, 1)
                writer.writerow([stat.size, stat.hit_count, stat.total_tokens, freq, stat.capture_priority])

        print(f"[+] CUDA 图 Top {int(top_percent * 100)}% 优选统计已导出至: {top_filepath}")
        print(f"    (有效剔除了长尾数据，共选中 {len(top_stats)} 个核心批次大小)")

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

        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.25)

        # --- 子图1：命中频率分布 (柱状图) ---
        ax1.bar(sizes, counts, color='#3F88C5', edgecolor='black', linewidth=0.5,
                width=max(1, (max(sizes) - min(sizes)) / 100))
        ax1.set_xlabel('Batch Size (Tokens)', fontweight='bold')
        ax1.set_ylabel('Hit Count', fontweight='bold')
        ax1.set_title('CUDA Graph Hit Distribution', fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # --- 子图2：CDF 累积分布曲线 ---
        ax2.plot(sizes, cdf, color='#D5A03A', linewidth=3)
        ax2.fill_between(sizes, 0, cdf, color='#D5A03A', alpha=0.2)
        ax2.axhline(y=0.9, color='red', linestyle=':', label='P90 Threshold')
        ax2.set_xlabel('Batch Size (Tokens)', fontweight='bold')
        ax2.set_ylabel('CDF', fontweight='bold')
        ax2.set_title('Cumulative Distribution Function (CDF)', fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right', frameon=False)
        ax2.grid(True, linestyle='--', alpha=0.6)

        filename = self.output_dir / f"hit_statistics{suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

HotspotAnalyzer = HotspotAnalyzer(analysis_thresholds=[1000, 5000, 10000],
            auto_plot=False,
            output_dir=os.environ.get(
                "VLLM_CUDA_GRAPH_HOTSPOT_DIR",
                str(resource_dir("cuda_graph_hotspots")),
            )
)
