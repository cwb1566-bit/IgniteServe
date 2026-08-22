# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import functools
import hashlib
import os
import threading
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch._inductor.compile_fx
import torch.fx as fx

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.utils import is_torch_equal_or_newer

from .inductor_pass import pass_context
from vllm.logger import init_logger
logger = init_logger(__name__)


class CompilerInterface:
    """
    The interface for a compiler that can be used by vLLM.
    """
    # The name of the compiler, e.g. inductor.
    # This is a class-level attribute.
    name: str

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = ""):
        """
        when the vLLM process uses `cache_dir` as the cache directory,
        the compiler should initialize itself with the cache directory,
        e.g. by re-directing its own cache directory to a sub-directory.

        prefix can be used in combination with cache_dir to figure out the base
        cache directory, e.g. there're multiple parts of model being compiled,
        but we want to share the same cache directory for all of them.

        e.g.
        cache_dir = "/path/to/dir/backbone", prefix = "backbone"
        cache_dir = "/path/to/dir/eagle_head", prefix = "eagle_head"
        """
        pass

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """
        Gather all the relevant information from the vLLM config,
        to compute a hash so that we can cache the compiled model.

        See [`VllmConfig.compute_hash`][vllm.config.VllmConfig.compute_hash]
        to check what information
        is already considered by default. This function should only
        consider the information that is specific to the compiler.
        """
        return ""

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
        graph_index: int = 0,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        """
        Compile the graph with the given example inputs and compiler config,
        with a runtime shape. If the `runtime_shape` is None, it means
        the `example_inputs` have a dynamic shape. Otherwise, the
        `runtime_shape` specifies the shape of the inputs. Right now we only
        support one variable shape for all inputs, which is the batchsize
        (number of tokens) during inference.

        Dynamo will make sure `graph(*example_inputs)` is valid.

        The function should return a compiled callable function, as well as
        a handle that can be used to directly load the compiled function.

        The handle should be a plain Python object, preferably a string or a
        file path for readability.

        If the compiler doesn't support caching, it should return None for the
        handle. If the compiler fails to compile the graph, it should return
        None for the compiled function as well.

        `key` is required for StandaloneInductorAdapter, it specifies where to
        save the compiled artifact. The compiled artifact gets saved to
        `cache_dir/key`.
        """
        return None, None

    def load(self,
             handle: Any,
             graph: fx.GraphModule,
             example_inputs: list[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Callable:
        """
        Load the compiled function from the handle.
        Raises an error if the handle is invalid.

        The handle is the second return value of the `compile` function.
        """
        raise NotImplementedError("caching is not supported")


class AlwaysHitShapeEnv:
    """
    Why do we need this class:

    For normal `torch.compile` usage, every compilation will have
    one Dynamo bytecode compilation and one Inductor compilation.
    The Inductor compilation happens under the context of the
    Dynamo bytecode compilation, and that context is used to
    determine the dynamic shape information, etc.

    For our use case, we only run Dynamo bytecode compilation once,
    and run Inductor compilation multiple times with different shapes
    plus a general shape. The compilation for specific shapes happens
    outside of the context of the Dynamo bytecode compilation. At that
    time, we don't have shape environment to provide to Inductor, and
    it will fail the Inductor code cache lookup.

    By providing a dummy shape environment that always hits, we can
    make the Inductor code cache lookup always hit, and we can
    compile the graph for different shapes as needed.

    The following dummy methods are obtained by trial-and-error
    until it works.
    """

    def __init__(self) -> None:
        self.guards: list[Any] = []

    def evaluate_guards_expression(self, *args, **kwargs):
        return True

    def get_pruned_guards(self, *args, **kwargs):
        return []

    def produce_guards_expression(self, *args, **kwargs):
        return ""


_inductor_factor_prewarm_thread: Optional[threading.Thread] = None


@functools.lru_cache(maxsize=1)
def _get_inductor_factors_cached() -> list[Any]:
    factors: list[Any] = []
    # summarize system state
    from torch._inductor.codecache import CacheBase
    system_factors = CacheBase.get_system()
    factors.append(system_factors)

    # summarize pytorch state
    from torch._inductor.codecache import torch_key
    torch_factors = torch_key()
    factors.append(torch_factors)
    return factors


def get_inductor_factors() -> list[Any]:
    thread = _inductor_factor_prewarm_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join()
    return _get_inductor_factors_cached()


def start_inductor_factor_prewarm() -> None:
    """Compute the compiler identity while model weights are loading."""
    global _inductor_factor_prewarm_thread
    thread = _inductor_factor_prewarm_thread
    if thread is not None:
        return
    thread = threading.Thread(
        target=get_inductor_factors,
        name="inductor-factor-prewarm",
        daemon=True,
    )
    _inductor_factor_prewarm_thread = thread
    thread.start()


class InductorStandaloneAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler.
    Requires PyTorch 2.8+.
    This is not on by default yet, but we plan to turn it on by default for
    PyTorch 2.8.

    Use VLLM_USE_STANDALONE_COMPILE to toggle this on or off.
    """
    name = "inductor_standalone"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        factors = get_inductor_factors()
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = ""):
        self.cache_dir = cache_dir

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
        graph_index: int = 0,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        compilation_counter.num_inductor_compiles += 1
        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)
        set_inductor_config(current_config, runtime_shape)

        if isinstance(runtime_shape, int):
            dynamic_shapes = "from_example_inputs"
        else:
            dynamic_shapes = "from_tracing_context"

        from torch._inductor import standalone_compile
        with pass_context(runtime_shape):
            compiled_graph = standalone_compile(
                graph,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                options={"config_patches": current_config})

        # Save the compiled artifact to disk in the specified path
        assert key is not None
        path = os.path.join(self.cache_dir, key)
        compiled_graph.save(path=path, format="unpacked")
        return compiled_graph, (key, path)

    def load(self,
             handle: Any,
             graph: fx.GraphModule,
             example_inputs: list[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Callable:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        path = handle[1]
        inductor_compiled_graph = torch._inductor.CompiledArtifact.load(
            path=path, format="unpacked")
        from torch._inductor.compile_fx import graph_returns_tuple
        returns_tuple = graph_returns_tuple(graph)

        def compiled_graph_wrapper(*args):
            graph_output = inductor_compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

class ThreadSafeInductorAdaptor(CompilerInterface):
    """
    Thread-safe Inductor adaptor used by the parallel compile backend.

    All subgraphs share one cache root, while PyTorch/Inductor still derives
    per-graph cache entries from the graph hash.
    """
    name = "inductor"

    def __init__(self):
        self._thread_local = threading.local()
        self._init_lock = threading.Lock()
        self._cache_initialized = False
        self.cache_dir: Optional[str] = None
        self.prefix: Optional[str] = None
        self.base_cache_dir: Optional[str] = None
        self.disable_cache: bool = False
        self.inductor_cache_dir: Optional[str] = None
        self.triton_cache_dir: Optional[str] = None

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """生成当前 Inductor 环境对应的 cache key"""
        factors = get_inductor_factors()
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def initialize_cache(
        self,
        cache_dir: str,
        disable_cache: bool = False,
        prefix: str = ""
    ):
        """
        懒初始化缓存目录（双重检查锁定）

        所有子图共享同一个缓存目录：
        - base_cache_dir/inductor_cache
        - base_cache_dir/triton_cache
        """
        # 快速路径：已初始化，无锁直接返回
        if self._cache_initialized:
            return

        # 慢速路径：需要初始化
        with self._init_lock:
            # 双重检查：防止多个线程同时初始化
            if self._cache_initialized:
                return

            thread_id = threading.get_ident()
            logger.debug(f"[Thread-{thread_id}] Initializing shared cache: {cache_dir}")

            self.cache_dir = cache_dir
            self.prefix = prefix
            self.base_cache_dir = cache_dir[:-len(prefix)] if prefix else cache_dir
            self.disable_cache = disable_cache

            if not disable_cache:
                # 创建共享的缓存目录（所有子图使用）
                self.inductor_cache_dir = os.path.join(self.base_cache_dir, "inductor_cache")
                self.triton_cache_dir = os.path.join(self.base_cache_dir, "triton_cache")

                os.makedirs(self.inductor_cache_dir, exist_ok=True)
                os.makedirs(self.triton_cache_dir, exist_ok=True)

                # 设置环境变量（全局，所有子图共享）
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = self.inductor_cache_dir
                os.environ["TRITON_CACHE_DIR"] = self.triton_cache_dir

                logger.info(
                    f"Shared cache directories initialized:\n"
                    f"  - Inductor: {self.inductor_cache_dir}\n"
                    f"  - Triton: {self.triton_cache_dir}"
                )

            # 标记已初始化
            self._cache_initialized = True

    def _init_thread_local_vars(self):
        """初始化线程本地变量"""
        if not hasattr(self._thread_local, 'thread_id'):
            self._thread_local.thread_id = threading.get_ident()

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
        graph_index: int = 0,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        """
        线程安全的编译方法（共享缓存目录版本）

        关键点：
        1. 所有子图使用相同的缓存目录
        2. Inductor 内部通过 graph hash 区分不同子图
        3. 无需为每个子图设置不同的环境变量
        """
        # 初始化线程本地变量
        self._init_thread_local_vars()
        thread_id = self._thread_local.thread_id

        logger.debug(f"[Thread-{thread_id}] Starting compilation for graph_{graph_index}")

        compilation_counter.num_inductor_compiles += 1

        # 深拷贝 graph，避免多线程修改同一个对象
        graph = copy.deepcopy(graph)

        # 准备配置
        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)

        current_config["fx_graph_cache"] = True
        current_config["fx_graph_remote_cache"] = False

        # 设置 Inductor 配置（无需锁，PyTorch 内部处理）
        set_inductor_config(current_config, runtime_shape)

        # 【关键】不修改环境变量，使用全局共享的缓存路径
        # Inductor 会根据 graph hash 自动在缓存目录中创建子目录

        try:
            # 根据 PyTorch 版本选择编译方法
            if torch.__version__ >= "2.6":
                compiled_graph, hash_str, file_path = self._compile_with_native_concurrent(
                    graph, example_inputs, current_config, runtime_shape, graph_index
                )
            else:
                # PyTorch 2.5 降级方案
                compiled_graph, hash_str, file_path = self._compile_with_minimal_patch(
                    graph, example_inputs, current_config, runtime_shape, graph_index
                )

        except Exception as e:
            logger.error(
                f"[Thread-{thread_id}] Compilation failed for graph_{graph_index}: {e}",
                exc_info=True
            )
            raise

        # 验证
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            if hash_str is None:
                raise RuntimeError(
                    f"[Thread-{thread_id}] Failed to compile graph_{graph_index}. "
                    "The most likely reason for this is that a previous compilation "
                    "failed, leading to a corrupted compilation artifact. "
                    "We recommend trying to "
                    "remove ~/.cache/vllm/torch_compile_cache and try again "
                    "to see the real issue.")
            assert file_path is not None, (
                f"[Thread-{thread_id}] Failed to get the file path of the compiled graph")

        logger.debug(
            f"[Thread-{thread_id}] Compilation successful for graph_{graph_index}:\n"
            f"  - Hash: {hash_str}\n"
            f"  - File: {file_path}"
        )

        return compiled_graph, (hash_str, file_path)

    def _compile_with_native_concurrent(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        config: dict[str, Any],
        runtime_shape: Optional[int],
        graph_index: int
    ) -> tuple[Callable, str, str]:
        """
        使用 PyTorch 2.6+ 的原生并发支持（无需全局 Patch 锁）

        所有子图共享相同的缓存目录，Inductor 内部通过 graph hash 区分
        """
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.codecache import FxGraphCache
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

        thread_id = self._thread_local.thread_id
        logger.debug(f"[Thread-{thread_id}] Using PyTorch 2.6+ native concurrent compile")

        # 初始化用于捕获信息的变量
        hash_str = None
        file_path = None

        # 定义劫持函数（仅用于捕获信息，不影响并发）
        def hijacked_compile_fx_inner(*args, **kwargs):
            nonlocal hash_str, file_path
            output = torch._inductor.compile_fx.compile_fx_inner(*args, **kwargs)

            if output is not None:
                # 从返回值中提取信息
                hash_str = output._fx_graph_cache_key
                compiled_fn = output.current_callable
                file_path = compiled_fn.__code__.co_filename

                # 查找真实的文件路径
                if self.base_cache_dir and not file_path.startswith(self.base_cache_dir):
                    for cell in compiled_fn.__closure__ or []:
                        if callable(cell.cell_contents):
                            code = cell.cell_contents.__code__
                            if code.co_filename.startswith(self.base_cache_dir):
                                file_path = code.co_filename
                                break

                logger.debug(
                    f"[Thread-{thread_id}] Captured compilation info for graph_{graph_index}:\n"
                    f"  - Hash: {hash_str}\n"
                    f"  - File: {file_path}"
                )

            return output

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        def _check_can_cache(*args, **kwargs):
            return

        # 【关键】不使用全局锁，直接 Patch（PyTorch 内部处理并发）
        with ExitStack() as stack:
            # Patch shape 环境
            stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      _get_shape_env))

            # Patch AOT 缓存的 shape 环境
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        _get_shape_env))

            # 强制允许缓存
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache))

            # Dynamo 指标上下文
            stack.enter_context(self.metrics_context())

            # 禁用远程缓存
            stack.enter_context(
                torch._inductor.config.patch(fx_graph_remote_cache=False))
            stack.enter_context(
                torch._functorch.config.patch(enable_remote_autograd_cache=False))

            # 执行编译（并发安全，无全局锁）
            with pass_context(runtime_shape):
                compiled_graph = compile_fx(
                    graph,
                    example_inputs,
                    inner_compile=hijacked_compile_fx_inner,
                    config_patches=config
                )

        return compiled_graph, hash_str, file_path

    def _compile_with_minimal_patch(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        config: dict[str, Any],
        runtime_shape: Optional[int],
        graph_index: int
    ) -> tuple[Callable, str, str]:
        """
        PyTorch 2.5 降级方案（使用最小化的 Patch）

        注意：PyTorch 2.5 对并发支持有限，建议升级到 2.6+
        """
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.codecache import FxGraphCache, compiled_fx_graph_hash
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

        thread_id = self._thread_local.thread_id
        logger.warning(
            f"[Thread-{thread_id}] Using PyTorch 2.5 (limited concurrency support). "
            "Consider upgrading to PyTorch 2.6+ for better parallel compilation."
        )

        # 初始化用于捕获信息的变量
        hash_str = None
        file_path = None

        # 定义劫持函数
        original_load = FxGraphCache.load
        original_load_name = "torch._inductor.codecache.FxGraphCache.load"

        def hijack_load(*args, **kwargs):
            nonlocal file_path
            inductor_compiled_graph = original_load(*args, **kwargs)
            compiled_fn = inductor_compiled_graph.current_callable
            file_path = compiled_fn.__code__.co_filename

            if self.base_cache_dir and not file_path.startswith(self.base_cache_dir):
                for cell in compiled_fn.__closure__ or []:
                    if not callable(cell.cell_contents):
                        continue
                    if cell.cell_contents.__code__.co_filename.startswith(
                            self.base_cache_dir):
                        file_path = cell.cell_contents.__code__.co_filename
                        break

            return inductor_compiled_graph

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            nonlocal hash_str
            out = compiled_fx_graph_hash(*args, **kwargs)
            hash_str = out[0]
            return out

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        def _check_can_cache(*args, **kwargs):
            return

        hijacked_compile_fx_inner = torch._inductor.compile_fx.compile_fx_inner

        # PyTorch 2.5 使用最小化的 Patch（无全局锁）
        with ExitStack() as stack:
            stack.enter_context(patch(original_load_name, hijack_load))

            stack.enter_context(
                patch("torch._inductor.codecache.compiled_fx_graph_hash",
                      hijack_compiled_fx_graph_hash))

            stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      _get_shape_env))

            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        _get_shape_env))

            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache))

            stack.enter_context(self.metrics_context())

            # 执行编译
            with pass_context(runtime_shape):
                compiled_graph = compile_fx(
                    graph,
                    example_inputs,
                    inner_compile=hijacked_compile_fx_inner,
                    config_patches=config
                )

        return compiled_graph, hash_str, file_path

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None
    ) -> Callable:
        """加载缓存的编译图（线程安全）"""
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)

        hash_str = handle[0]
        thread_id = threading.get_ident()

        logger.debug(f"[Thread-{thread_id}] Loading cached graph: hash={hash_str}")

        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor.codecache import FxGraphCache

        with ExitStack() as exit_stack:
            exit_stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      lambda *args, **kwargs: AlwaysHitShapeEnv()))

            if hasattr(AOTAutogradCache, "_get_shape_env"):
                exit_stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        lambda *args, **kwargs: AlwaysHitShapeEnv()))

            exit_stack.enter_context(self.metrics_context())

            if torch.__version__.startswith("2.5"):
                inductor_compiled_graph = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, False)
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove "
                    "the cache directory and try again.")

            elif torch.__version__ >= "2.6":
                from torch._inductor.output_code import CompiledFxGraphConstantsWithGm
                constants = CompiledFxGraphConstantsWithGm(graph)
                inductor_compiled_graph, _ = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, None, constants)
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove "
                    "the cache directory and try again.")

        # 检查原始图是否返回元组
        from torch._inductor.compile_fx import graph_returns_tuple
        returns_tuple = graph_returns_tuple(graph)

        # 返回包装后的可调用对象
        def compiled_graph(*args):
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph

    def metrics_context(self) -> contextlib.AbstractContextManager:
        """获取 Dynamo 指标上下文"""
        if is_torch_equal_or_newer("2.6"):
            import torch._dynamo.utils
            return torch._dynamo.utils.get_metrics_context()
        else:
            return contextlib.nullcontext()

class InductorAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler, version 2.5, 2.6, 2.7.
    """
    name = "inductor"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        factors = get_inductor_factors()
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = ""):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.base_cache_dir = cache_dir[:-len(prefix)] if prefix else cache_dir
        if disable_cache:
            return
        # redirect the cache directory to a sub-directory
        # set flags so that Inductor and Triton store their cache
        # in the cache_dir, then users only need to copy the cache_dir
        # to another machine to reuse the cache.
        inductor_cache = os.path.join(self.base_cache_dir, "inductor_cache")
        os.makedirs(inductor_cache, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        triton_cache = os.path.join(self.base_cache_dir, "triton_cache")
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
        graph_index: int = 0,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        compilation_counter.num_inductor_compiles += 1
        from torch._inductor.compile_fx import compile_fx
        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)

        # disable remote cache
        current_config["fx_graph_cache"] = True
        current_config["fx_graph_remote_cache"] = False

        set_inductor_config(current_config, runtime_shape)

        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.

        hash_str, file_path = None, None
        from torch._inductor.codecache import (FxGraphCache,
                                               compiled_fx_graph_hash)
        if torch.__version__.startswith("2.5"):
            original_load = FxGraphCache.load
            original_load_name = "torch._inductor.codecache.FxGraphCache.load"

            def hijack_load(*args, **kwargs):
                inductor_compiled_graph = original_load(*args, **kwargs)
                nonlocal file_path
                compiled_fn = inductor_compiled_graph.current_callable
                file_path = compiled_fn.__code__.co_filename  # noqa
                if not file_path.startswith(self.base_cache_dir):
                    # hooked in the align_inputs_from_check_idxs function
                    # in torch/_inductor/utils.py
                    for cell in compiled_fn.__closure__:
                        if not callable(cell.cell_contents):
                            continue
                        if cell.cell_contents.__code__.co_filename.startswith(
                                self.base_cache_dir):
                            # this is the real file path compiled from Inductor
                            file_path = cell.cell_contents.__code__.co_filename
                            break
                return inductor_compiled_graph

            hijacked_compile_fx_inner = torch._inductor.compile_fx.compile_fx_inner  # noqa
        elif torch.__version__ >= "2.6":
            # function renamed in 2.6
            original_load_name = None

            def hijacked_compile_fx_inner(*args, **kwargs):
                output = torch._inductor.compile_fx.compile_fx_inner(
                    *args, **kwargs)
                nonlocal hash_str
                inductor_compiled_graph = output
                if inductor_compiled_graph is not None:
                    nonlocal file_path
                    compiled_fn = inductor_compiled_graph.current_callable
                    file_path = compiled_fn.__code__.co_filename  # noqa
                    if not file_path.startswith(self.base_cache_dir):
                        # hooked in the align_inputs_from_check_idxs function
                        # in torch/_inductor/utils.py
                        for cell in compiled_fn.__closure__:
                            if not callable(cell.cell_contents):
                                continue
                            code = cell.cell_contents.__code__
                            if code.co_filename.startswith(
                                    self.base_cache_dir):
                                # this is the real file path
                                # compiled from Inductor
                                file_path = code.co_filename
                                break
                    hash_str = inductor_compiled_graph._fx_graph_cache_key
                return output

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            out = compiled_fx_graph_hash(*args, **kwargs)
            nonlocal hash_str
            hash_str = out[0]
            return out

        def _check_can_cache(*args, **kwargs):
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For vLLM, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        with ExitStack() as stack:
            # hijack to get the compiled graph itself
            if original_load_name is not None:
                stack.enter_context(patch(original_load_name, hijack_load))

            # for hijacking the hash of the compiled graph
            stack.enter_context(
                patch("torch._inductor.codecache.compiled_fx_graph_hash",
                      hijack_compiled_fx_graph_hash))

            # for providing a dummy shape environment
            stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      _get_shape_env))

            from torch._functorch._aot_autograd.autograd_cache import (
                AOTAutogradCache)

            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        _get_shape_env))

            # for forcing the graph to be cached
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache))

            # Dynamo metrics context, see method for more details.
            stack.enter_context(self.metrics_context())

            # Disable remote caching. When these are on, on remote cache-hit,
            # the monkey-patched functions never actually get called.
            # vLLM today assumes and requires the monkey-patched functions to
            # get hit.
            # TODO(zou3519): we're going to replace this all with
            # standalone_compile sometime.
            if is_torch_equal_or_newer("2.6"):
                stack.enter_context(
                    torch._inductor.config.patch(fx_graph_remote_cache=False))
                stack.enter_context(
                    torch._functorch.config.patch(
                        enable_remote_autograd_cache=False))

            with pass_context(runtime_shape):
                compiled_graph = compile_fx(
                    graph,
                    example_inputs,
                    inner_compile=hijacked_compile_fx_inner,
                    config_patches=current_config)

        # We treat VLLM_DISABLE_COMPILE_CACHE as the overall switch for torch
        # compilation cache. So turn off the checks if we disable the
        # compilation cache.
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            if hash_str is None:
                raise RuntimeError(
                    "vLLM failed to compile the model. The most "
                    "likely reason for this is that a previous compilation "
                    "failed, leading to a corrupted compilation artifact. "
                    "We recommend trying to "
                    "remove ~/.cache/vllm/torch_compile_cache and try again "
                    "to see the real issue. ")
            assert file_path is not None, (
                "failed to get the file path of the compiled graph")
        return compiled_graph, (hash_str, file_path)

    def load(self,
             handle: Any,
             graph: fx.GraphModule,
             example_inputs: list[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Callable:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        hash_str = handle[0]

        from torch._functorch._aot_autograd.autograd_cache import (
            AOTAutogradCache)
        from torch._inductor.codecache import FxGraphCache
        with ExitStack() as exit_stack:
            exit_stack.enter_context(
                patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                      lambda *args, **kwargs: AlwaysHitShapeEnv()))
            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                exit_stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        lambda *args, **kwargs: AlwaysHitShapeEnv()))

            # Dynamo metrics context, see method for more details.
            exit_stack.enter_context(self.metrics_context())

            if torch.__version__.startswith("2.5"):
                inductor_compiled_graph = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, False)
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove"
                    f"the cache directory and try again."  # noqa
                )
            elif torch.__version__ >= "2.6":
                from torch._inductor.output_code import (
                    CompiledFxGraphConstantsWithGm)
                constants = CompiledFxGraphConstantsWithGm(graph)
                inductor_compiled_graph, _ = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, None, constants)
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove"
                    f"the cache directory and try again."  # noqa
                )

        # Inductor calling convention (function signature):
        # f(list) -> tuple
        # Dynamo calling convention (function signature):
        # f(*args) -> Any

        # need to know if the graph returns a tuple
        from torch._inductor.compile_fx import graph_returns_tuple
        returns_tuple = graph_returns_tuple(graph)

        # this is the callable we return to Dynamo to run
        def compiled_graph(*args):
            # convert args to list
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            # unpack the tuple if needed
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph

    def metrics_context(self) -> contextlib.AbstractContextManager:
        """
        This method returns the Dynamo metrics context (if it exists,
        otherwise a null context). It is used by various compile components.
        Present in torch>=2.6, it's used inside FxGraphCache in
        torch==2.6 (but not after). It might also be used in various other
        torch.compile internal functions.

        Because it is re-entrant, we always set it (even if entering via Dynamo
        and the context was already entered). We might want to revisit if it
        should be set at a different level of compilation.

        This is likely a bug in PyTorch: public APIs should not rely on
        manually setting up internal contexts. But we also rely on non-public
        APIs which might not provide these guarantees.
        """
        if is_torch_equal_or_newer("2.6"):
            import torch._dynamo.utils
            return torch._dynamo.utils.get_metrics_context()
        else:
            return contextlib.nullcontext()


def set_inductor_config(config, runtime_shape):
    if isinstance(runtime_shape, int):
        # for a specific batchsize, tuning triton kernel parameters
        # can be beneficial
        config["max_autotune"] = True
        config["coordinate_descent_tuning"] = True


class EagerAdaptor(CompilerInterface):
    name = "eager"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
        graph_index: int = 0,
    ) -> tuple[Optional[Callable], Optional[Any]]:
        compilation_counter.num_eager_compiles += 1
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        return graph, None
