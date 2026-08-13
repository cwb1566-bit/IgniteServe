import time
import os
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Set, Optional
import numpy as np
from pathlib import Path
import threading
import torch
from vllm.logger import init_logger
logger = init_logger(__name__)
from vllm.distributed.parallel_state import graph_capture


def dynamic_cuda_graph_capture_disabled() -> bool:
    value = os.environ.get("VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE", "")
    return value.lower() in ("1", "true", "yes", "on")


def dynamic_cuda_graph_capture_no_replay() -> bool:
    value = os.environ.get("VLLM_DYNAMIC_CUDA_GRAPH_CAPTURE_NO_REPLAY", "")
    return value.lower() in ("1", "true", "yes", "on")


class DynamicGraphManager:
    """CUDA 图动态捕获管理器 (Worker进程级单例)"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DynamicGraphManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_runner=None):
        if getattr(self, '_initialized', False):
            return

        with self._lock:
            if getattr(self, '_initialized', False):
                return

            # if model_runner is None:
            #     raise ValueError("[DynamicGraphManager] 首次初始化必须提供 model_runner!")

            self.runner = model_runner
            self.capture_queue = set()
            self._initialized = True
            logger.info("[DynamicGraphManager] 初始化完成，已挂载至 Worker 进程。")

    def submit_to_queue(self, batch_size: int):
        with self._lock:
            self.capture_queue.add(batch_size)

    def _is_capturable_shape(self, batch_size: int) -> bool:
        return batch_size in getattr(self.runner, "cudagraph_batch_sizes", [])

    def _mark_captured(self, batch_size: int) -> None:
        if dynamic_cuda_graph_capture_no_replay():
            logger.info(
                "Dynamic CUDA graph shape %s captured but not registered "
                "for replay because VLLM_DYNAMIC_CUDA_GRAPH_CAPTURE_NO_REPLAY is set.",
                batch_size)
            return
        dynamic_captured = getattr(self.runner, "dynamic_cuda_graph_captured", None)
        if dynamic_captured is not None:
            dynamic_captured.add(batch_size)
        if batch_size not in self.runner.cuda_graph_captured:
            self.runner.cuda_graph_captured.append(batch_size)
            self.runner.cuda_graph_captured.sort()

        graph_matcher = getattr(self.runner, "graphMatcher", None)
        if graph_matcher is not None and batch_size not in graph_matcher.captured_sizes:
            graph_matcher.captured_sizes.append(batch_size)
            graph_matcher.captured_sizes.sort()

    # 在系统空闲时捕获
    def _runner_has_live_requests(self) -> bool:
        if self.runner is None:
            return False
        requests = getattr(self.runner, "requests", {})
        input_batch = getattr(self.runner, "input_batch", None)
        req_ids = getattr(input_batch, "req_ids", []) if input_batch is not None else []
        return bool(requests) or bool(req_ids)

    def capture_pending_graphs(self):
        """执行真正的动态捕获（安全点触发）"""
        if dynamic_cuda_graph_capture_disabled():
            return

        if self.runner is None:
            logger.warning("[DynamicGraphManager] runner is not initialized; skip pending CUDA graph capture.")
            return
        if self._runner_has_live_requests():
            return

        with self._lock:
            if not self.capture_queue:
                return
            shapes_to_capture = list(self.capture_queue)
            self.capture_queue.difference_update(shapes_to_capture)

        # 从大到小排序以复用显存池
        shapes_to_capture.sort(reverse=True)
        logger.info(f"系统空闲，开始执行动态按需捕获 CUDA 图，Shapes: {shapes_to_capture}")
        start_time = time.perf_counter()

        with self.runner.manager.fast_capture_scope(), graph_capture(device=self.runner.device):
            full_cg = self.runner.full_cuda_graph
            for num_tokens in shapes_to_capture:
                if num_tokens in self.runner.cuda_graph_captured:
                    continue
                if not self._is_capturable_shape(num_tokens):
                    logger.warning(
                        "Skip dynamic CUDA graph capture for shape %s because it is not in cudagraph_capture_sizes.",
                        num_tokens)
                    continue
                logger.info(f"  -> 正在动态捕获 Shape: {num_tokens} ...")
                # Warmup
                for _ in range(self.runner.compilation_config.cudagraph_num_of_warmups):
                    self.runner._dummy_run(num_tokens, capture_attn_cudagraph=full_cg, skip_eplb=True)
                # 真正的捕获
                self.runner._dummy_run(num_tokens, capture_attn_cudagraph=full_cg, skip_eplb=True)
                self._mark_captured(num_tokens)

        total_time = time.perf_counter() - start_time
        logger.info(f"捕获完成，总耗时: {total_time:.4f}s\n")

    # 立即捕获
    def capture_graph_for_batch_size(self, batch_size):
        if dynamic_cuda_graph_capture_disabled():
            return

        if self.runner is None:
            logger.warning("[DynamicGraphManager] runner is not initialized; skip CUDA graph capture for %s.",
                           batch_size)
            return
        if self._runner_has_live_requests():
            self.submit_to_queue(batch_size)
            return

        with self._lock:
            if batch_size in self.capture_queue:
                self.capture_queue.remove(batch_size)

        if batch_size in self.runner.cuda_graph_captured:
            return
        if not self._is_capturable_shape(batch_size):
            logger.warning(
                "Skip dynamic CUDA graph capture for shape %s because it is not in cudagraph_capture_sizes.",
                batch_size)
            return

        with self.runner.manager.fast_capture_scope(), graph_capture(device=self.runner.device):
            full_cg = self.runner.full_cuda_graph
            self.runner._dummy_run(batch_size,
                            capture_attn_cudagraph=full_cg,
                            skip_eplb=True)
            self._mark_captured(batch_size)
        logger.info(f"捕获批次: {batch_size}\n")

DynamicGraphManager = DynamicGraphManager()
