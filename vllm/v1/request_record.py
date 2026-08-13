import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass, asdict


@dataclass
class RequestMetrics:
    """单个请求的指标"""
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    arrival_time: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None

    @property
    def ttft(self) -> Optional[float]:
        """首token延迟 (Time to First Token)"""
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """总延迟"""
        if self.completion_time:
            return self.completion_time - self.arrival_time
        return None

    @property
    def generation_latency(self) -> Optional[float]:
        """生成延迟"""
        if self.completion_time and self.first_token_time:
            return self.completion_time - self.first_token_time
        return None

    @property
    def throughput(self) -> Optional[float]:
        """吞吐量 (tokens/s)"""
        if self.generation_latency and self.generation_latency > 0:
            return self.generated_tokens / self.generation_latency
        return None


class MetricsRecorder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def __init(self, csv_path: str = None):
        # CSV 文件路径
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"vllm_metrics_{timestamp}.csv"

        self.csv_path = Path(csv_path)

        # 当前活跃的请求
        self.active_requests: Dict[str, RequestMetrics] = {}

        # 初始化 CSV 文件（写入表头）
        self._init_csv()

        # print(f"✅ 指标记录器已初始化，数据将保存到: {self.csv_path}")

    def _init_csv(self):
        """初始化 CSV 文件"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                'request_id',
                'prompt_tokens',
                'generated_tokens',
                'arrival_time',
                'first_token_time',
                'completion_time',
                'ttft',
                'total_latency',
                'generation_latency',
                'throughput'
            ])

    def start_request(self, request_id: str, prompt_tokens: int):
        """记录请求开始"""
        metrics = RequestMetrics(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=0,
            arrival_time=time.time()
        )
        self.active_requests[request_id] = metrics
        return metrics

    def record_first_token(self, request_id: str):
        """记录首个token生成"""
        if request_id in self.active_requests:
            self.active_requests[request_id].first_token_time = time.time()

    def record_token(self, request_id: str):
        """记录token生成"""
        if request_id in self.active_requests:
            metrics = self.active_requests[request_id]

            # 如果是第一个token，记录首token时间
            if metrics.first_token_time is None:
                metrics.first_token_time = time.time()

            metrics.generated_tokens += 1

    def complete_request(self, request_id: str):
        """记录请求完成并保存到CSV"""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests[request_id]
        metrics.completion_time = time.time()

        # 立即保存到 CSV
        self._save_to_csv(metrics)

        # 从活跃请求中移除
        del self.active_requests[request_id]

    def _save_to_csv(self, metrics: RequestMetrics):
        """保存单条记录到CSV"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.request_id,
                metrics.prompt_tokens,
                metrics.generated_tokens,
                metrics.arrival_time,
                metrics.first_token_time,
                metrics.completion_time,
                metrics.ttft,
                metrics.total_latency,
                metrics.generation_latency,
                metrics.throughput
            ])


recorder = MetricsRecorder()