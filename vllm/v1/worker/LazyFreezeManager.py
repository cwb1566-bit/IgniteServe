import gc
import torch
from contextlib import contextmanager
import time


class LazyFreezeManager:
    _instance = None
    _is_globally_frozen = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def prepare_for_capture(self):
        """
        在模型加载完成、KV Cache 分配好后调用一次。
        将模型参数等长寿对象永久冻结。
        """
        if not self._is_globally_frozen:
            gc.collect()  # 仅在此处执行一次重型回收
            gc.freeze()  # 冻结当前所有模型参数
            self._is_globally_frozen = True
            print("Model state globally frozen for fast CUDA graph capture.")

    def release_for_model_rebind(self) -> bool:
        """Return frozen objects to normal GC before dropping a model."""
        if not self._is_globally_frozen:
            return False
        gc.unfreeze()
        self._is_globally_frozen = False
        return True

    @contextmanager
    def fast_capture_scope(self):
        """
        轻量级捕获上下文：跳过 gc.collect()，仅关闭 GC 开关。
        """
        # 确保全局已冻结
        self.prepare_for_capture()

        # 暂时关闭自动回收，防止捕获期间触发 Generation 0 的回收
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()
