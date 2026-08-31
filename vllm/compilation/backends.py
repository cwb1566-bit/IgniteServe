# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from dataclasses import dataclass
import functools
import importlib.util
import os
import pickle
import pprint
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
import operator
from typing import Any, Callable, Optional, Dict, List, Tuple

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher

import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer, resolve_obj_by_qualname

from .compiler_interface import (CompilerInterface, EagerAdaptor,
                                 InductorAdaptor, InductorStandaloneAdaptor,
                                 ThreadSafeInductorAdaptor)
from .cache_reuse import (GlobalSubgraphCache, graph_semantic_key,
                          normalized_model_paths)
from .startup_resources import (enabled_by_master, env_flag,
                                parallel_compile_metadata_dir, resource_file)
from .counter import compilation_counter
from .inductor_pass import InductorPass
from .pass_manager import PostGradPassManager

# 增
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from vllm.v1.phase_profiler import timer

logger = init_logger(__name__)


def _dump_dynamo_fx_artifact_probe(graph: fx.GraphModule,
                                   example_inputs: list[Any],
                                   traced_files: Sequence[str],
                                   runtime_identity: dict[str, Any]) -> None:
    """Persist the pre-backend FX graph with PyTorch's wire serializer."""
    output_dir = (os.getenv("VLLM_DYNAMO_FRONTEND_CACHE_DIR")
                  or os.getenv("VLLM_DYNAMO_ARTIFACT_PROBE_DIR"))
    if not output_dir:
        return

    probe_dir = Path(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    stem = f"dynamo_fx_rank_{rank}_{os.getpid()}"
    started = time.perf_counter()
    import hashlib
    traced_file_sha256 = {}
    for path in traced_files:
        if path == "<string>" or not os.path.isfile(path):
            continue
        with open(path, "rb") as source_file:
            traced_file_sha256[path] = hashlib.sha256(
                source_file.read()).hexdigest()
    metadata = {
        "nodes": len(list(graph.graph.nodes)),
        "placeholders": sum(node.op == "placeholder"
                            for node in graph.graph.nodes),
        "node_ops": {
            op: sum(node.op == op for node in graph.graph.nodes)
            for op in sorted({node.op for node in graph.graph.nodes})
        },
        "call_module_targets": sorted(
            str(node.target) for node in graph.graph.nodes
            if node.op == "call_module"),
        "example_inputs": len(example_inputs),
        "example_input_types": [
            f"{type(value).__module__}.{type(value).__qualname__}"
            for value in example_inputs
        ],
        "traced_files": sorted(traced_files),
        "traced_file_sha256": traced_file_sha256,
        "runtime_identity": runtime_identity,
    }
    structural_started = time.perf_counter()
    try:
        graph_payload = _serialize_graph_module_for_subprocess(graph)
        graph_bytes = pickle.dumps(graph_payload, protocol=pickle.HIGHEST_PROTOCOL)
        graph_path = probe_dir / f"{stem}.fxstruct.pkl"
        graph_path.write_bytes(graph_bytes)

        from torch.fx._graph_pickler import GraphPickler
        from torch.fx.experimental.symbolic_shapes import free_symbols
        direct_symbols = tuple(
            (node.name, node.meta["example_value"])
            for node in graph.graph.nodes
            if isinstance(node.meta.get("example_value"),
                          (torch.SymInt, torch.SymFloat)))
        direct_symbol_bytes = GraphPickler.dumps(direct_symbols)
        direct_symbol_path = (
            probe_dir / f"{stem}.direct_symbols.graphpickle")
        direct_symbol_path.write_bytes(direct_symbol_bytes)
        symbol_dependencies = {}
        for node in graph.graph.nodes:
            value = node.meta.get("example_value")
            if value is None:
                continue
            symbols = sorted(str(symbol) for symbol in free_symbols(value))
            if symbols:
                symbol_dependencies[node.name] = symbols

        metadata.update({
            "fxstruct_bytes": len(graph_bytes),
            "fxstruct_path": str(graph_path),
            "direct_symbols": len(direct_symbols),
            "direct_symbol_graphpickle_bytes": len(direct_symbol_bytes),
            "direct_symbol_graphpickle_path": str(direct_symbol_path),
            "symbol_dependencies": symbol_dependencies,
            "structural_seconds": time.perf_counter() - structural_started,
        })
    except Exception as exc:
        metadata.update({
            "structural_error": f"{type(exc).__name__}: {exc}",
            "structural_seconds": time.perf_counter() - structural_started,
        })
    (probe_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def structural_reuse_enabled() -> bool:
    if env_flag("VLLM_DISABLE_STRUCTURAL_REUSE", False):
        return False
    return enabled_by_master("VLLM_ENABLE_STRUCTURAL_REUSE")


def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    if compilation_config.use_inductor:
        if envs.VLLM_USE_STANDALONE_COMPILE and is_torch_equal_or_newer(
                "2.8.0.dev"):
            logger.debug("Using InductorStandaloneAdaptor")
            return InductorStandaloneAdaptor()
        else:
            logger.debug("Using ThreadSafeInductorAdaptor")
            return ThreadSafeInductorAdaptor()
    else:
        logger.debug("Using EagerAdaptor")
        return EagerAdaptor()


class CompilerManager:
    """
    Manage compilation, cache loading, and compile-artifact reuse.

    `cache` stores compiler handles keyed by
    `(runtime_shape, graph_index, backend_name)`.
    `compile_metadata_for_reuse` stores canonical reusable graph indices for the
    same key so follower graphs can alias cached or in-memory compiled graphs.
    """

    def __init__(self, compilation_config: CompilationConfig):
        self.cache: dict[tuple[Optional[int], int, str], Any] = {}
        self.compiled_graph_list: dict[tuple[Optional[int], int], Any] = {}
        self.compile_metadata_for_reuse: dict[
            tuple[Optional[int], int, str], tuple[int, Any]
        ] = {}
        self.is_cache_updated = False
        self.is_cache_metadata_updated = False
        self.disable_cache = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)
        self.global_cache: Optional[GlobalSubgraphCache] = None

    def _cache_key(self,
                   runtime_shape: Optional[int],
                   graph_index: int) -> tuple[Optional[int], int, str]:
        return runtime_shape, graph_index, self.compiler.name

    def _compiled_key(self, runtime_shape: Optional[int],
                      graph_index: int) -> tuple[Optional[int], int]:
        return runtime_shape, graph_index

    def _maybe_start_timer(self, graph_index: int, track_metrics: bool) -> None:
        if graph_index == 0 and track_metrics:
            global compilation_start_time
            compilation_start_time = time.time()

    def _maybe_finish_timer(self,
                            graph_index: int,
                            num_graphs: int,
                            runtime_shape: Optional[int],
                            track_metrics: bool,
                            action: str) -> None:
        if not track_metrics or graph_index != num_graphs - 1:
            return
        elapsed = time.time() - compilation_start_time
        self.compilation_config.compilation_time += elapsed
        if runtime_shape is None:
            logger.info('%s a graph for general shape takes %.2f s', action,
                        elapsed)
        else:
            logger.info('%s a graph for shape %s takes %.2f s', action,
                        runtime_shape, elapsed)

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = ''):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, 'vllm_compile_cache.py')
        self.cache_metadata_file_path = os.path.join(
            cache_dir, 'vllm_compile_reuse_cache.py')

        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())

        if not disable_cache and os.path.exists(self.cache_metadata_file_path):
            with open(self.cache_metadata_file_path) as f:
                self.compile_metadata_for_reuse = ast.literal_eval(f.read())

        if (not disable_cache
                and os.environ.get(
                    'VLLM_ENABLE_GLOBAL_SUBGRAPH_CACHE', '').lower()
                in {'1', 'true', 'yes', 'on'}):
            self.global_cache = GlobalSubgraphCache(cache_dir, prefix)
        else:
            self.global_cache = None

        self.compiler.initialize_cache(cache_dir=cache_dir,
                                       disable_cache=disable_cache,
                                       prefix=prefix)

    def initialize_cache_model_meta(self, model_name: str):
        del model_name

    def save_to_file(self):
        if not self.disable_cache and self.is_cache_updated:
            printer = pprint.PrettyPrinter(indent=4)
            with open(self.cache_file_path, 'w') as f:
                f.write(printer.pformat(self.cache))

        if not self.disable_cache and self.is_cache_metadata_updated:
            printer = pprint.PrettyPrinter(indent=4)
            with open(self.cache_metadata_file_path, 'w') as f:
                f.write(printer.pformat(self.compile_metadata_for_reuse))

    def has_cached_graph(self,
                         graph_index: int,
                         runtime_shape: Optional[int] = None) -> bool:
        return self._cache_key(runtime_shape, graph_index) in self.cache

    def get_cached_handle(self,
                          graph_index: int,
                          runtime_shape: Optional[int] = None) -> Any:
        return self.cache.get(self._cache_key(runtime_shape, graph_index))

    def get_compiled_graph(self,
                           graph_index: int,
                           runtime_shape: Optional[int] = None) -> Any:
        return self.compiled_graph_list.get(
            self._compiled_key(runtime_shape, graph_index))

    def get_reuse_source(self,
                         graph_index: int,
                         runtime_shape: Optional[int] = None) -> int:
        reuse_entry = self.compile_metadata_for_reuse.get(
            self._cache_key(runtime_shape, graph_index))
        if reuse_entry is None:
            return graph_index
        return reuse_entry[0]

    def alias_graph(self,
                    graph_index: int,
                    source_index: int,
                    runtime_shape: Optional[int] = None) -> None:
        source_handle = self.get_cached_handle(source_index, runtime_shape)
        source_graph = self.get_compiled_graph(source_index, runtime_shape)
        if source_handle is not None:
            self.cache[self._cache_key(runtime_shape, graph_index)] = source_handle
            self.compile_metadata_for_reuse[
                self._cache_key(runtime_shape, graph_index)
            ] = (source_index, source_handle)
            self.is_cache_updated = True
            self.is_cache_metadata_updated = True
        if source_graph is not None:
            self.compiled_graph_list[
                self._compiled_key(runtime_shape, graph_index)
            ] = source_graph

    def _record_compiled_artifact(self,
                                  graph: fx.GraphModule,
                                  graph_index: int,
                                  runtime_shape: Optional[int],
                                  compiled_graph: Optional[Any],
                                  handle: Any) -> None:
        if compiled_graph is not None:
            self.compiled_graph_list[
                self._compiled_key(runtime_shape, graph_index)
            ] = compiled_graph

        if handle is None:
            return

        cache_key = self._cache_key(runtime_shape, graph_index)
        self.cache[cache_key] = handle
        self.is_cache_updated = True

        reuse_source = self.get_reuse_source(graph_index, runtime_shape)
        if reuse_source == graph_index:
            for (cached_shape, cached_graph_index,
                 compiler_name), cached_handle in self.cache.items():
                if (cached_shape == runtime_shape
                        and compiler_name == self.compiler.name
                        and cached_graph_index != graph_index
                        and cached_handle == handle):
                    reuse_source = self.get_reuse_source(cached_graph_index,
                                                         runtime_shape)
                    break

        self.compile_metadata_for_reuse[cache_key] = (reuse_source, handle)
        self.is_cache_metadata_updated = True
        if self.global_cache is not None:
            self.global_cache.register(
                graph_semantic_key(graph, runtime_shape, self.compiler.name),
                handle,
                self.cache_dir,
                graph_index,
                runtime_shape,
            )

    def supports_external_compile_handles(self) -> bool:
        return (not self.disable_cache and isinstance(
            self.compiler,
            (InductorAdaptor, ThreadSafeInductorAdaptor,
             InductorStandaloneAdaptor)))

    def record_compiled_handle(self,
                               graph_index: int,
                               runtime_shape: Optional[int],
                               handle: Any) -> None:
        if handle is None:
            raise RuntimeError(
                'Subprocess compile requires a persistent compiler handle.')
        cache_key = self._cache_key(runtime_shape, graph_index)
        self.cache[cache_key] = handle
        self.compile_metadata_for_reuse[cache_key] = (graph_index, handle)
        self.is_cache_updated = True
        self.is_cache_metadata_updated = True

    def load(self,
             graph: fx.GraphModule,
             example_inputs: list[Any],
             graph_index: int,
             num_graphs: int = 1,
             runtime_shape: Optional[int] = None,
             track_metrics: bool = True,
             count_backend_compilations: bool = True) -> Optional[Callable]:
        self._maybe_start_timer(graph_index, track_metrics)

        handle = self.get_cached_handle(graph_index, runtime_shape)
        global_hit = False
        if (handle is None and self.global_cache is not None
                and os.environ.get(
                    'VLLM_ENABLE_GLOBAL_SUBGRAPH_DIRECT_LOOKUP', '').lower()
                in {'1', 'true', 'yes', 'on'}):
            handle = self.global_cache.lookup(
                graph_semantic_key(graph, runtime_shape, self.compiler.name))
            global_hit = handle is not None
        if handle is None:
            return None
        if global_hit:
            cache_key = self._cache_key(runtime_shape, graph_index)
            self.cache[cache_key] = handle
            self.compile_metadata_for_reuse[cache_key] = (graph_index, handle)
            self.is_cache_updated = True
            self.is_cache_metadata_updated = True
            logger.info('Loaded graph %s for shape %s from global subgraph cache.',
                        graph_index, str(runtime_shape))
        if count_backend_compilations:
            compilation_counter.num_backend_compilations += 1

        reuse_source = self.get_reuse_source(graph_index, runtime_shape)
        if reuse_source != graph_index:
            reusable_graph = self.get_compiled_graph(reuse_source,
                                                     runtime_shape)
            if reusable_graph is not None:
                self.alias_graph(graph_index, reuse_source, runtime_shape)
                logger.debug('Reuse loaded graph %s for graph %s on shape %s.',
                             reuse_source, graph_index, str(runtime_shape))
                self._maybe_finish_timer(graph_index, num_graphs,
                                         runtime_shape, track_metrics,
                                         'Loading')
                return reusable_graph

        compiled_graph = self.compiler.load(handle, graph, example_inputs,
                                            graph_index, runtime_shape)
        self.compiled_graph_list[
            self._compiled_key(runtime_shape, graph_index)
        ] = compiled_graph
        logger.debug(
            'Loaded graph %s for shape %s from %s via handle %s.',
            graph_index, str(runtime_shape), self.compiler.name, handle)
        self._maybe_finish_timer(graph_index, num_graphs, runtime_shape,
                                 track_metrics, 'Loading')
        return compiled_graph

    def compile(self,
                graph: fx.GraphModule,
                example_inputs,
                additional_inductor_config,
                compilation_config: CompilationConfig,
                graph_index: int = 0,
                num_graphs: int = 1,
                runtime_shape: Optional[int] = None,
                track_metrics: bool = True) -> Any:
        del compilation_config
        self._maybe_start_timer(graph_index, track_metrics)
        compilation_counter.num_backend_compilations += 1

        compiled_graph = self.load(graph,
                                   example_inputs,
                                   graph_index,
                                   num_graphs=num_graphs,
                                   runtime_shape=runtime_shape,
                                   track_metrics=False,
                                   count_backend_compilations=False)
        if compiled_graph is not None:
            self._maybe_finish_timer(graph_index, num_graphs, runtime_shape,
                                     track_metrics, 'Loading')
            return compiled_graph

        reuse_source = self.get_reuse_source(graph_index, runtime_shape)
        if reuse_source != graph_index:
            reusable_graph = self.get_compiled_graph(reuse_source,
                                                     runtime_shape)
            if reusable_graph is not None:
                self.alias_graph(graph_index, reuse_source, runtime_shape)
                logger.debug(
                    'Reuse compiled graph %s for graph %s on shape %s.',
                    reuse_source, graph_index, str(runtime_shape))
                self._maybe_finish_timer(graph_index, num_graphs,
                                         runtime_shape, track_metrics,
                                         'Compiling')
                return reusable_graph

        if isinstance(self.compiler,
                      (InductorAdaptor, ThreadSafeInductorAdaptor)):
            maybe_key = None
        else:
            maybe_key = f'artifact_shape_{runtime_shape}_subgraph_{graph_index}'

        compiled_graph, handle = self.compiler.compile(
            graph,
            example_inputs,
            additional_inductor_config,
            runtime_shape,
            maybe_key,
            graph_index=graph_index,
        )
        assert compiled_graph is not None, 'Failed to compile the graph'

        self._record_compiled_artifact(graph, graph_index, runtime_shape,
                                       compiled_graph, handle)
        if handle is not None:
            if graph_index == 0:
                logger.info('Cache the graph of shape %s for later use',
                            str(runtime_shape))
            logger.debug(
                'Stored graph %s for shape %s from %s via handle %s.',
                graph_index, str(runtime_shape), self.compiler.name, handle)

        self._maybe_finish_timer(graph_index, num_graphs, runtime_shape,
                                 track_metrics, 'Compiling')
        return compiled_graph


@dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(graph: fx.GraphModule, ops: list[str]) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0


@dataclass
class TensorMetadata:
    """张量元信息（用于重建 FakeTensor）"""
    shape: List[Any]  # 可能包含符号维度（用字符串表示）
    dtype: str  # e.g., "torch.float16"
    device: str  # e.g., "cuda:0"
    stride: Optional[List[int]] = None
    requires_grad: bool = False
    is_sparse: bool = False

    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            'shape': [str(s) for s in self.shape],  # 转换 SymInt 为字符串
            'dtype': self.dtype,
            'device': self.device,
            'stride': [str(s) for s in self.stride] if self.stride else None,
            'requires_grad': self.requires_grad,
            'is_sparse': self.is_sparse,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'TensorMetadata':
        """从字典恢复（保持字符串格式，延迟解析）"""
        return TensorMetadata(
            # ========== 关键修复：不 eval，保持字符串 ==========
            # shape=[eval(s) if isinstance(s, str) else s for s in data['shape']],
            shape=data['shape'],
            dtype=data['dtype'],
            device=data['device'],
            # stride=[eval(s) if isinstance(s, str) else s for s in data['stride']] if data['stride'] else None,
            stride=data['stride'] if data['stride'] else None,
            requires_grad=data.get('requires_grad', False),
            is_sparse=data.get('is_sparse', False),
        )


@dataclass
class SymIntMetadata:
    """SymInt 元信息"""
    expr: str  # 符号表达式，e.g., "s0", "s1 * 2"
    hint: Optional[int] = None  # 提示值（用于推导）

    def to_dict(self) -> Dict:
        return {
            'expr': self.expr,
            'hint': self.hint,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'SymIntMetadata':
        return SymIntMetadata(
            expr=data['expr'],
            hint=data.get('hint'),
        )


@dataclass
class SubmoduleInputMetadata:
    """单个子模块的输入元信息"""
    submod_name: str
    index: int
    args_metadata: List[Any]  # TensorMetadata | SymIntMetadata | 原始值
    kwargs_metadata: Dict[str, Any]
    sym_shape_indices: List[int]

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            'submod_name': self.submod_name,
            'index': self.index,
            'args_metadata': [
                arg.to_dict() if hasattr(arg, 'to_dict') else arg
                for arg in self.args_metadata
            ],
            'kwargs_metadata': {
                k: v.to_dict() if hasattr(v, 'to_dict') else v
                for k, v in self.kwargs_metadata.items()
            },
            'sym_shape_indices': self.sym_shape_indices,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'SubmoduleInputMetadata':
        """从字典恢复"""

        def restore_arg(arg):
            if isinstance(arg, dict):
                if 'shape' in arg and 'dtype' in arg:
                    return TensorMetadata.from_dict(arg)
                elif 'expr' in arg:
                    return SymIntMetadata.from_dict(arg)
            return arg

        return SubmoduleInputMetadata(
            submod_name=data['submod_name'],
            index=data['index'],
            args_metadata=[restore_arg(arg) for arg in data['args_metadata']],
            kwargs_metadata={
                k: restore_arg(v) for k, v in data['kwargs_metadata'].items()
            },
            sym_shape_indices=data['sym_shape_indices'],
        )


# ========== 序列化工具类 ==========
class SubmoduleInputSerializer:
    """子模块输入的序列化/反序列化工具"""

    def __init__(self, metadata_cache_dir: str, model_name):
        self.metadata_cache_dir = Path(metadata_cache_dir)
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.metadata_cache_dir / f"submodule_inputs_metadata_for_{model_name}.json"
        # Reuse the same symbolic object for the same textual expression so
        # reconstructed fake inputs preserve cross-argument shape relations.
        self.symbol_table: Dict[str, Any] = {}
        self.symbol_hints: Dict[str, int] = {}

    def seed_symbol_table_from_arg(self, arg: Any) -> None:
        """Seed symbolic dims from the current Dynamo/FakeTensor inputs."""
        if isinstance(arg, torch.Tensor):
            for dim in arg.shape:
                if isinstance(dim, torch.SymInt):
                    self.symbol_table.setdefault(str(dim), dim)
            try:
                for stride in arg.stride():
                    if isinstance(stride, torch.SymInt):
                        self.symbol_table.setdefault(str(stride), stride)
            except Exception:
                pass
            return
        if isinstance(arg, torch.SymInt):
            self.symbol_table.setdefault(str(arg), arg)
            return
        if isinstance(arg, (list, tuple)):
            for item in arg:
                self.seed_symbol_table_from_arg(item)
            return
        if isinstance(arg, dict):
            for item in arg.values():
                self.seed_symbol_table_from_arg(item)

    def seed_symbol_table_from_args(self, args: tuple[Any, ...]) -> None:
        for arg in args:
            self.seed_symbol_table_from_arg(arg)

    def extract_tensor_metadata(self, tensor: torch.Tensor) -> TensorMetadata:
        """从 Tensor 或 FakeTensor 提取元信息"""
        # 处理 SymInt 维度
        shape = []
        for s in tensor.shape:
            if isinstance(s, torch.SymInt):
                # 保存符号表达式
                shape.append(str(s))
            else:
                shape.append(int(s))

        # 处理 stride
        stride = None
        if hasattr(tensor, 'stride'):
            stride = []
            for s in tensor.stride():
                if isinstance(s, torch.SymInt):
                    stride.append(str(s))
                else:
                    stride.append(int(s))

        return TensorMetadata(
            shape=shape,
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            stride=stride,
            requires_grad=tensor.requires_grad,
            is_sparse=tensor.is_sparse,
        )

    def extract_symint_metadata(self, sym_int: torch.SymInt) -> SymIntMetadata:
        """从 SymInt 提取元信息"""
        # 提取符号表达式
        expr = str(sym_int)

        # 尝试获取提示值
        hint = None
        if hasattr(sym_int, 'node') and hasattr(sym_int.node, 'hint'):
            hint = sym_int.node.hint

        return SymIntMetadata(expr=expr, hint=hint)

    def extract_arg_metadata(self, arg: Any) -> Any:
        """提取单个参数的元信息"""
        if isinstance(arg, torch.Tensor):
            return self.extract_tensor_metadata(arg)
        elif isinstance(arg, torch.SymInt):
            return self.extract_symint_metadata(arg)
        elif isinstance(arg, (list, tuple)):
            return type(arg)([self.extract_arg_metadata(a) for a in arg])
        elif isinstance(arg, dict):
            return {k: self.extract_arg_metadata(v) for k, v in arg.items()}
        else:
            # 原始类型（int, float, str, None 等）
            return arg

    def extract_submodule_metadata(
            self,
            submod_name: str,
            args: Tuple,
            kwargs: Dict,
            sym_shape_indices: List[int],
            index: int,
    ) -> SubmoduleInputMetadata:
        """提取子模块输入的元信息"""
        args_metadata = [self.extract_arg_metadata(arg) for arg in args]
        kwargs_metadata = {k: self.extract_arg_metadata(v) for k, v in kwargs.items()}

        return SubmoduleInputMetadata(
            submod_name=submod_name,
            index=index,
            args_metadata=args_metadata,
            kwargs_metadata=kwargs_metadata,
            sym_shape_indices=sym_shape_indices,
        )

    def save_all_metadata(
            self,
            metadata_list: List[SubmoduleInputMetadata]
    ) -> None:
        """保存所有子模块的元信息到 JSON 文件"""
        data = {
            'version': '1.0',
            'num_submodules': len(metadata_list),
            'submodules': [meta.to_dict() for meta in metadata_list],
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Saved metadata for {len(metadata_list)} submodules to "
            f"{self.metadata_file}"
        )

    def load_all_metadata(self) -> List[SubmoduleInputMetadata]:
        """从 JSON 文件加载所有子模块的元信息"""
        if not self.metadata_file.exists():
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            return []

        with open(self.metadata_file, 'r') as f:
            data = json.load(f)

        metadata_list = [
            SubmoduleInputMetadata.from_dict(submod_data)
            for submod_data in data['submodules']
        ]

        logger.info(
            f"Loaded metadata for {len(metadata_list)} submodules from "
            f"{self.metadata_file}"
        )
        return metadata_list

    def reconstruct_fake_tensor(
            self,
            metadata: TensorMetadata,
            fake_mode,
    ) -> torch.Tensor:
        """从元信息重建 FakeTensor"""
        # 解析 dtype
        dtype = getattr(torch, metadata.dtype.split('.')[-1])

        # 解析 device
        device = torch.device(metadata.device)

        # 解析 shape（可能包含 SymInt）
        shape = []
        for s in metadata.shape:
            if isinstance(s, str):
                # 检查是否是纯数字
                if s.isdigit():
                    shape.append(int(s))
                else:
                    # 这是一个符号表达式（如 "s0", "s1 * 2"）
                    # 创建新的 SymInt（使用默认提示值）
                    symint = self._create_symint(s, fake_mode)
                    shape.append(symint)
            else:
                shape.append(s)

        # 创建 FakeTensor
        with fake_mode:
            fake_tensor = torch.empty(
                shape,
                dtype=dtype,
                device=device,
                requires_grad=metadata.requires_grad,
            )

        return fake_tensor

    def _create_symint(
            self,
            expr: str,
            fake_mode,
            hint: Optional[int] = None,
    ) -> torch.SymInt:
        """
        创建 SymInt（从符号表达式）

        Args:
            expr: 符号表达式，如 "s0", "s1 * 2"
            fake_mode: FakeTensorMode
            hint: 提示值（默认从 expr 中提取或使用 1024）

        Returns:
            torch.SymInt
        """
        cached_symbol = self.symbol_table.get(expr)
        if cached_symbol is not None:
            return cached_symbol

        if os.environ.get("VLLM_SUBPROCESS_CONCRETE_SYMINT", "").lower() in {
                "1", "true", "yes", "on"}:
            value = hint or self.symbol_hints.get(expr) or 1024
            self.symbol_table[expr] = value
            return value

        if expr.isdigit():
            value = int(expr)
            self.symbol_table[expr] = value
            return value

        # ========== 提取符号名称和提示值 ==========
        # 处理简单符号（如 "s0"）
        if expr.startswith('s') and expr[1:].isdigit():
            sym_name = expr
            hint = hint or self.symbol_hints.get(expr) or 1024  # 默认提示值
        else:
            # 处理复杂表达式（如 "s0 * 2"）
            # 简化处理：提取第一个符号
            import re
            match = re.search(r's\d+', expr)
            if match:
                sym_name = match.group()
                hint = hint or self.symbol_hints.get(sym_name) or 1024
            else:
                # 无法解析，使用默认值
                logger.warning(f"Cannot parse symbol expression: {expr}")
                value = hint or 1024
                self.symbol_table[expr] = value
                return value

        if expr != sym_name:
            try:
                symbol_names = set(re.findall(r's\d+', expr))
                local_symbols = {
                    name: self._create_symint(name, fake_mode, hint)
                    for name in symbol_names
                }
                evaluated = eval(expr, {"__builtins__": {}}, local_symbols)
                self.symbol_table[expr] = evaluated
                return evaluated
            except Exception:
                logger.debug("Falling back to base symbol reconstruction for %s",
                             expr)

        # ========== 从 ShapeEnv 获取或创建符号 ==========
        shape_env = fake_mode.shape_env

        # ========== 方法 1：实验性使用带 hint 的 unspecified/backed SymInt ==========
        if os.environ.get("VLLM_SUBPROCESS_UNSPECIFIED_SYMINT", "").lower() in {
                "1", "true", "yes", "on"}:
            try:
                from torch.fx.experimental.symbolic_shapes import DimDynamic
                from torch._dynamo.source import ConstantSource

                symint = shape_env.create_unspecified_symint_and_symbol(
                    int(hint),
                    source=ConstantSource(f"vllm_{sym_name}"),
                    dynamic_dim=DimDynamic.DYNAMIC,
                )
                logger.debug(
                    f"Created unspecified SymInt for {sym_name} with hint {hint}")
                self.symbol_table[expr] = symint
                self.symbol_table.setdefault(sym_name, symint)
                return symint
            except Exception as exc:
                logger.warning(
                    "Failed to create unspecified SymInt for %s: %s; "
                    "falling back to unbacked SymInt", sym_name, exc)

        # ========== 方法 2：使用 create_unbacked_symint（原始实现）==========
        try:
            symint = shape_env.create_unbacked_symint()
            logger.debug(f"Created unbacked SymInt for {sym_name}")
            self.symbol_table[expr] = symint
            self.symbol_table.setdefault(sym_name, symint)
            return symint

        except AttributeError:
            pass

        value = hint or 1024
        self.symbol_table[expr] = value
        self.symbol_table.setdefault(sym_name, value)
        return value

    def reconstruct_symint(
            self,
            metadata: SymIntMetadata,
            fake_mode,
    ) -> torch.SymInt:
        """从元信息重建 SymInt"""
        return self._create_symint(
            metadata.expr,
            fake_mode,
            hint=metadata.hint,
        )

    def reconstruct_arg(
            self,
            arg_metadata: Any,
            fake_mode,
    ) -> Any:
        """重建单个参数"""
        if isinstance(arg_metadata, TensorMetadata):
            return self.reconstruct_fake_tensor(arg_metadata, fake_mode)
        elif isinstance(arg_metadata, SymIntMetadata):
            return self.reconstruct_symint(arg_metadata, fake_mode)
        elif isinstance(arg_metadata, list):
            return [self.reconstruct_arg(a, fake_mode) for a in arg_metadata]
        elif isinstance(arg_metadata, tuple):
            return tuple([self.reconstruct_arg(a, fake_mode) for a in arg_metadata])
        elif isinstance(arg_metadata, dict):
            return {k: self.reconstruct_arg(v, fake_mode) for k, v in arg_metadata.items()}
        else:
            return arg_metadata

    def _collect_symbol_hints(self, arg_metadata: Any) -> None:
        if isinstance(arg_metadata, SymIntMetadata):
            if arg_metadata.hint is not None:
                self.symbol_hints[arg_metadata.expr] = int(arg_metadata.hint)
        elif isinstance(arg_metadata, (list, tuple)):
            for item in arg_metadata:
                self._collect_symbol_hints(item)
        elif isinstance(arg_metadata, dict):
            for item in arg_metadata.values():
                self._collect_symbol_hints(item)

    def reconstruct_submodule_inputs(
            self,
            metadata: SubmoduleInputMetadata,
            fake_mode,
    ) -> Tuple[Tuple, Dict]:
        """重建子模块的输入"""
        for arg_meta in metadata.args_metadata:
            self._collect_symbol_hints(arg_meta)
        for v_meta in metadata.kwargs_metadata.values():
            self._collect_symbol_hints(v_meta)

        args = tuple([
            self.reconstruct_arg(arg_meta, fake_mode)
            for arg_meta in metadata.args_metadata
        ])

        kwargs = {
            k: self.reconstruct_arg(v_meta, fake_mode)
            for k, v_meta in metadata.kwargs_metadata.items()
        }

        return args, kwargs


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(self,
                 module: torch.fx.GraphModule,
                 compile_submod_names: list[str],
                 vllm_config: VllmConfig,
                 graph_pool,
                 vllm_backend: "VllmBackend",
                 metadata_cache_dir: Optional[str] = None):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        if self.fake_mode is None:
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            self.fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        self.extra_traceback = False

        model_name = vllm_config.model_config.model.split('/')[-1]
        self.serializer = SubmoduleInputSerializer(metadata_cache_dir, model_name)
        self.restore_input_metadata = enabled_by_master(
            "VLLM_ENABLE_PARALLEL_COMPILE_METADATA")
        self.cached_metadata = (
            self._try_load_cached_metadata()
            if self.restore_input_metadata else None)
        if (not self.restore_input_metadata
                and self.serializer.metadata_file.exists()):
            logger.info(
                "Parallel compile input metadata exists at %s, but restore is "
                "disabled outside the S4 parallel-compile-metadata path.",
                self.serializer.metadata_file)
        self.fake_run_results_metadata: dict[int, Any] = {}
        self.fake_run_map_filepath = (
                self.serializer.metadata_cache_dir /
                f'fake_run_map_metadata_for_{model_name}.json')
        self.fake_run_map_list = self._load_or_build_reuse_map()

    def _normalize_reuse_map(self,
                             raw_map: dict[Any, Any]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        max_index = len(self.compile_submod_names)
        for index in range(max_index):
            raw_value = raw_map.get(str(index), raw_map.get(index, index))
            try:
                mapped_index = int(raw_value)
            except (TypeError, ValueError):
                mapped_index = index
            if mapped_index < 0 or mapped_index >= max_index:
                mapped_index = index
            normalized[str(index)] = mapped_index
        return normalized

    def _normalize_target(self, target: Any) -> str:
        return re.sub(r'\d+', '#', str(target))

    def _submodule_signature(self, submod: torch.fx.GraphModule) -> tuple[Any, ...]:
        signature: list[tuple[str, str]] = []
        for node in submod.graph.nodes:
            target = self._normalize_target(node.target)
            signature.append((node.op, target))
        return tuple(signature)

    def _build_structural_reuse_map(self) -> dict[str, int]:
        signature_to_index: dict[tuple[Any, ...], int] = {}
        reuse_map: dict[str, int] = {}
        for index, submod_name in enumerate(self.compile_submod_names):
            submod = self.fetch_attr(submod_name)
            signature = self._submodule_signature(submod)
            canonical_index = signature_to_index.setdefault(signature, index)
            reuse_map[str(index)] = canonical_index
        return reuse_map

    def _load_or_build_reuse_map(self) -> dict[str, int]:
        if not structural_reuse_enabled():
            return self._normalize_reuse_map({
                str(index): index
                for index in range(len(self.compile_submod_names))
            })
        if self.fake_run_map_filepath.exists():
            try:
                return self._normalize_reuse_map(self._load_fake_run_metadata())
            except Exception as exc:
                logger.warning('Failed to load fake-run reuse metadata: %s', exc)
        return self._normalize_reuse_map(self._build_structural_reuse_map())

    def _try_load_cached_metadata(self) -> Optional[Dict[str, SubmoduleInputMetadata]]:
        try:
            metadata_list = self.serializer.load_all_metadata()
            if metadata_list:
                return {meta.submod_name: meta for meta in metadata_list}
        except Exception as exc:
            logger.warning('Failed to load cached metadata: %s', exc)
        return None

    def run(self, *args):
        self.vllm_backend.reuse_index_map = {
            int(index): int(mapped_index)
            for index, mapped_index in self.fake_run_map_list.items()
        }
        if self.cached_metadata is not None and not args:
            logger.info('Restoring submodule inputs from cache without '
                        'replaying full-graph example inputs...')
            self._restore_from_cache()
            return None

        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]

        if self.cached_metadata is not None:
            logger.info('Restoring submodule inputs from cache...')
            self.serializer.seed_symbol_table_from_args(tuple(fake_args))
            self._restore_from_cache()
            return None

        logger.info('Running fake mode inference...')
        with self.fake_mode, enable_python_dispatcher():
            result = super().run(*fake_args)

        self._save_metadata()
        return result

    def _load_fake_run_metadata(self) -> dict[str, int]:
        with self.fake_run_map_filepath.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _record_submodule_metadata(self,
                                   target: str,
                                   args: tuple[Any, ...],
                                   kwargs: dict[str, Any]) -> int:
        submod = self.fetch_attr(target)
        sym_shape_indices = [
            i for i, arg in enumerate(args) if isinstance(arg, torch.SymInt)
        ]
        index = self.compile_submod_names.index(target)
        self.vllm_backend.meta_data[target] = {
            'submod': submod,
            'args': args,
            'kwargs': kwargs,
            'sym_shape_indices': sym_shape_indices,
            'index': index,
        }
        return index

    def _restore_from_cache(self):
        for submod_name in self.compile_submod_names:
            if submod_name not in self.cached_metadata:
                logger.warning('No cached metadata for %s', submod_name)
                continue

            meta = self.cached_metadata[submod_name]
            args, kwargs = self.serializer.reconstruct_submodule_inputs(
                meta, self.fake_mode)
            submod = self.fetch_attr(submod_name)
            self.vllm_backend.meta_data[submod_name] = {
                'submod': submod,
                'args': args,
                'kwargs': kwargs,
                'sym_shape_indices': meta.sym_shape_indices,
                'index': meta.index,
            }

    def _save_metadata(self):
        metadata_list = []
        for submod_name in self.compile_submod_names:
            if submod_name not in self.vllm_backend.meta_data:
                logger.warning('No metadata for %s', submod_name)
                continue
            data = self.vllm_backend.meta_data[submod_name]
            metadata_list.append(
                self.serializer.extract_submodule_metadata(
                    submod_name=submod_name,
                    args=data['args'],
                    kwargs=data['kwargs'],
                    sym_shape_indices=data['sym_shape_indices'],
                    index=data['index'],
                ))

        self.serializer.save_all_metadata(metadata_list)
        with self.fake_run_map_filepath.open('w', encoding='utf-8') as f:
            json.dump(self.fake_run_map_list, f, indent=2)

    def call_module(self, target: torch.fx.node.Target,
                    args: tuple[torch.fx.node.Argument, ...],
                    kwargs: dict[str, Any]) -> Any:
        assert isinstance(target, str)
        if target not in self.compile_submod_names:
            return super().call_module(target, args, kwargs)

        index = self._record_submodule_metadata(target, args, kwargs)
        mapped_index = int(self.fake_run_map_list.get(str(index), index))
        if mapped_index != index and mapped_index in self.fake_run_results_metadata:
            reused_output = self.fake_run_results_metadata[mapped_index]
            self.fake_run_results_metadata[index] = reused_output
            return reused_output

        output = super().call_module(target, args, kwargs)
        self.fake_run_results_metadata[index] = output
        if mapped_index not in self.fake_run_results_metadata:
            self.fake_run_map_list[str(index)] = index
        return output


# the tag for the part of model being compiled,
# e.g. backbone/eagle_head
model_tag: str = "backbone"


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert tag != model_tag, \
        f"Model tag {tag} is the same as the current tag {model_tag}."
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


def _runtime_shape_label(runtime_shape: Optional[int]) -> str:
    return "dynamic" if runtime_shape is None else str(runtime_shape)


_FX_NODE_REF = "__fx_node_ref__"
_FX_TUPLE = "__fx_tuple__"
_FX_LIST = "__fx_list__"
_FX_DICT = "__fx_dict__"
_FX_SLICE = "__fx_slice__"
_FX_SIZE = "__fx_size__"
_FX_TARGET = "__fx_target__"
_FX_SUBPROCESS_MODULE_CLASS = "FxModule"


def _get_module_attr_by_qualname(root: torch.nn.Module, target: str) -> Any:
    obj: Any = root
    for atom in target.split("."):
        obj = getattr(obj, atom)
    return obj


def _set_module_attr_by_qualname(root: torch.nn.Module, target: str,
                                 value: Any) -> None:
    atoms = target.split(".")
    parent = root
    for atom in atoms[:-1]:
        child = getattr(parent, atom, None)
        if child is None:
            child = torch.nn.Module()
            parent.add_module(atom, child)
        parent = child

    leaf = atoms[-1]
    if isinstance(value, torch.nn.Module):
        parent.add_module(leaf, value)
    elif isinstance(value, torch.nn.Parameter):
        parent.register_parameter(leaf, value)
    elif isinstance(value, torch.Tensor):
        parent.register_buffer(leaf, value)
    else:
        setattr(parent, leaf, value)


def _encode_fx_value(value: Any) -> Any:
    if isinstance(value, fx.Node):
        return {_FX_NODE_REF: value.name}
    if isinstance(value, torch.Size):
        return {_FX_SIZE: [_encode_fx_value(v) for v in value]}
    if isinstance(value, slice):
        return {
            _FX_SLICE: [
                _encode_fx_value(value.start),
                _encode_fx_value(value.stop),
                _encode_fx_value(value.step),
            ]
        }
    if isinstance(value, tuple):
        return {_FX_TUPLE: [_encode_fx_value(v) for v in value]}
    if isinstance(value, list):
        return {_FX_LIST: [_encode_fx_value(v) for v in value]}
    if isinstance(value, dict):
        return {
            _FX_DICT: [[_encode_fx_value(k), _encode_fx_value(v)]
                       for k, v in value.items()]
        }
    return value


def _decode_fx_value(value: Any, env: dict[str, fx.Node]) -> Any:
    if isinstance(value, dict):
        if _FX_NODE_REF in value:
            return env[value[_FX_NODE_REF]]
        if _FX_SIZE in value:
            return torch.Size(
                [_decode_fx_value(v, env) for v in value[_FX_SIZE]])
        if _FX_SLICE in value:
            start, stop, step = value[_FX_SLICE]
            return slice(
                _decode_fx_value(start, env),
                _decode_fx_value(stop, env),
                _decode_fx_value(step, env),
            )
        if _FX_TUPLE in value:
            return tuple(_decode_fx_value(v, env) for v in value[_FX_TUPLE])
        if _FX_LIST in value:
            return [_decode_fx_value(v, env) for v in value[_FX_LIST]]
        if _FX_DICT in value:
            return {
                _decode_fx_value(k, env): _decode_fx_value(v, env)
                for k, v in value[_FX_DICT]
            }
    return value


def _serialize_fx_target(target: Any) -> Any:
    if isinstance(target, str):
        return target

    qualified_name: Optional[str] = None
    try:
        qualified_name = fx.node._get_qualified_name(target)
    except Exception:
        qualified_name = None

    if qualified_name is None:
        module_name = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if module_name and qualname:
            qualified_name = f"{module_name}.{qualname}"

    if qualified_name is None:
        raise TypeError(f"Unsupported FX target for subprocess transport: {target!r}")

    return {_FX_TARGET: qualified_name}


def _deserialize_fx_target(target: Any) -> Any:
    if isinstance(target, str):
        return target

    if isinstance(target, dict) and _FX_TARGET in target:
        qualified_name = target[_FX_TARGET]
        try:
            return resolve_obj_by_qualname(qualified_name)
        except Exception:
            namespace = {
                "torch": torch,
                "operator": operator,
            }
            return eval(qualified_name, {"__builtins__": {}}, namespace)

    return target


def _serialize_graph_module_for_subprocess(
        graph_module: fx.GraphModule) -> dict[str, Any]:
    attr_targets = sorted(
        {
            str(node.target)
            for node in graph_module.graph.nodes
            if node.op == "get_attr"
        },
        key=lambda target: (target.count("."), target),
    )

    payload = {
        "version": 1,
        "attrs": {
            target: _get_module_attr_by_qualname(graph_module, target)
            for target in attr_targets
        },
        "nodes": [],
    }

    for node in graph_module.graph.nodes:
        payload["nodes"].append({
            "op": node.op,
            "name": node.name,
            "target": _serialize_fx_target(node.target),
            "args": _encode_fx_value(node.args),
            "kwargs": _encode_fx_value(node.kwargs),
        })

    return payload


def _deserialize_graph_module_for_subprocess(
        payload: dict[str, Any]) -> fx.GraphModule:
    root = torch.nn.Module()
    for target, value in sorted(
            payload.get("attrs", {}).items(),
            key=lambda item: (item[0].count("."), item[0])):
        _set_module_attr_by_qualname(root, target, value)

    graph = fx.Graph()
    env: dict[str, fx.Node] = {}
    for node_payload in payload["nodes"]:
        node = graph.create_node(
            node_payload["op"],
            _deserialize_fx_target(node_payload["target"]),
            _decode_fx_value(node_payload["args"], env),
            _decode_fx_value(node_payload["kwargs"], env),
            name=node_payload["name"],
        )
        env[node.name] = node

    graph.lint()
    return fx.GraphModule(root, graph)


def _save_graph_module_for_subprocess(graph_module: fx.GraphModule,
                                      graph_path: str) -> None:
    shutil.rmtree(graph_path, ignore_errors=True)
    os.makedirs(graph_path, exist_ok=True)
    payload_path = os.path.join(graph_path, "graph_module.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump(_serialize_graph_module_for_subprocess(graph_module), f)


def _load_graph_module_for_subprocess(graph_path: str) -> fx.GraphModule:
    payload_path = os.path.join(graph_path, "graph_module.pkl")
    if os.path.exists(payload_path):
        with open(payload_path, "rb") as f:
            return _deserialize_graph_module_for_subprocess(pickle.load(f))

    init_file = os.path.join(graph_path, "__init__.py")
    if not os.path.exists(init_file):
        raise FileNotFoundError(
            f"FX graph payload not found at {payload_path}")

    package_name = (
        f"_vllm_fx_subprocess_{abs(hash(graph_path))}_{time.time_ns()}")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[graph_path],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to create import spec for FX package at {graph_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)

    graph_cls = getattr(module, _FX_SUBPROCESS_MODULE_CLASS)
    return graph_cls()


def _validate_saved_graph_for_subprocess(graph_path: str) -> Optional[str]:
    try:
        _load_graph_module_for_subprocess(graph_path)
        return None
    except Exception as exc:
        return repr(exc)


class ParallelCompileManager:
    """Coordinate parallel compile, parallel cache load, and graph reuse."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(os.cpu_count() or 1, 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.subprocess_graph_blacklist: set[int] = set()
        self._blacklist_cache_dir: Optional[str] = None
        logger.info(
            'ParallelCompileManager initialized with %s workers '
            '(thread cache load + optional process compile)',
            self.max_workers)
        if os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
                "1", "true", "yes", "on"}:
            os.environ.setdefault('TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS',
                                  str(self.max_workers))
            os.environ.setdefault('TORCHINDUCTOR_FX_COMPILE_INNER_THREADS', '1')
            os.environ.setdefault('TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA', '1')
            worker_imports = os.environ.get(
                'TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS', '')
            required_imports = (
                'vllm._custom_ops',
                'vllm.compilation.pass_manager',
            )
            import_names = [name for name in worker_imports.split(',') if name]
            for module_name in required_imports:
                if module_name not in import_names:
                    import_names.append(module_name)
            os.environ['TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS'] = ','.join(
                import_names)
            try:
                from torch._inductor.compile_fx_full_subproc import (
                    FullFxCompilePool,
                )
                FullFxCompilePool.prewarm_workers()
                logger.info(
                    'Started and queued import prewarm for Full-FX compiler '
                    'workers during model init.')
            except Exception as exc:
                logger.warning(
                    'Failed to pre-start Full-FX compiler worker pool: %s',
                    exc)

    def _subprocess_blacklist_path(self,
                                   compiler_manager: CompilerManager) -> str:
        return os.path.join(compiler_manager.cache_dir,
                            'subprocess_compile_blacklist.json')

    def _load_subprocess_blacklist(self,
                                   compiler_manager: CompilerManager) -> None:
        cache_dir = compiler_manager.cache_dir
        if not cache_dir:
            return
        if self._blacklist_cache_dir == cache_dir:
            return

        blacklist_path = self._subprocess_blacklist_path(compiler_manager)
        blacklist: set[int] = set()
        if os.path.exists(blacklist_path):
            try:
                with open(blacklist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                blacklist = {
                    int(index) for index in data.get('graph_indices', [])
                }
            except Exception as exc:
                logger.warning(
                    'Failed to load subprocess compile blacklist from %s: %s',
                    blacklist_path,
                    exc,
                )

        self.subprocess_graph_blacklist = blacklist
        self._blacklist_cache_dir = cache_dir

    def _save_subprocess_blacklist(self,
                                   compiler_manager: CompilerManager) -> None:
        cache_dir = compiler_manager.cache_dir
        if not cache_dir:
            return
        blacklist_path = self._subprocess_blacklist_path(compiler_manager)
        with open(blacklist_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'graph_indices': sorted(self.subprocess_graph_blacklist)},
                f,
                indent=2,
            )

    def _resolve_reuse_source(self, graph_index: int,
                              reuse_index_map: dict[int, int]) -> int:
        source_index = graph_index
        visited: set[int] = set()
        while True:
            next_index = reuse_index_map.get(source_index, source_index)
            if next_index == source_index or next_index in visited:
                return source_index
            visited.add(source_index)
            source_index = next_index

    def _completed_future(self, value: tuple[Any, str]) -> Future:
        future: Future = Future()
        future.set_result(value)
        return future

    def _load_cached_source_graph(self,
                                  meta: Dict[str, Any],
                                  compiler_manager: CompilerManager,
                                  graph_index: int,
                                  num_graphs: int,
                                  runtime_shape: Optional[int]) -> tuple[Any, str]:
        compiled_graph = compiler_manager.load(
            meta['submod'],
            meta['args'],
            graph_index=graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
        )
        return compiled_graph, 'load'

    def _compile_source_graph_serial(
            self,
            meta: Dict[str, Any],
            compiler_manager: CompilerManager,
            compilation_config: CompilationConfig,
            graph_index: int,
            num_graphs: int,
            runtime_shape: Optional[int]) -> tuple[Any, str]:
        compiled_graph = compiler_manager.compile(
            meta['submod'],
            meta['args'],
            compilation_config.inductor_compile_config,
            compilation_config,
            graph_index=graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
        )
        return compiled_graph, 'compile'

    def _load_freshly_compiled_source_graph(
            self,
            meta: Dict[str, Any],
            compiler_manager: CompilerManager,
            graph_index: int,
            num_graphs: int,
            runtime_shape: Optional[int]) -> tuple[Any, str]:
        started = time.perf_counter()
        compiled_graph = compiler_manager.load(
            meta['submod'],
            meta['args'],
            graph_index=graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
            count_backend_compilations=False,
        )
        logger.info(
            'Parallel compile timing graph_%s parent_artifact_load_s=%.6f',
            graph_index,
            time.perf_counter() - started,
        )
        return compiled_graph, 'compile'

    def _can_use_subprocess_compile(
            self,
            compiler_manager: CompilerManager,
            compilation_config: CompilationConfig) -> bool:
        # Keep subprocess compile opt-in only. The current subprocess path is
        # experimental and can regress startup latency when it fails.
        if not getattr(compilation_config, 'use_subprocess_compile', False):
            return False
        if not compiler_manager.supports_external_compile_handles():
            return False
        if not getattr(compiler_manager, 'cache_dir', None):
            return False
        if not getattr(compilation_config, 'local_cache_dir', None):
            return False
        return True

    def _prepare_subprocess_compile_assets(
            self,
            source_indices: List[int],
            metadata_by_index: Dict[int, Dict[str, Any]],
            compiler_manager: CompilerManager,
            compilation_config: CompilationConfig,
            vllm_config: VllmConfig,
            runtime_shape: Optional[int]) -> Dict[int, Dict[str, str]]:
        cache_dir = compiler_manager.cache_dir
        prefix = os.path.basename(cache_dir)
        job_dir = os.path.join(cache_dir, 'subprocess_compile_jobs')
        os.makedirs(job_dir, exist_ok=True)

        config_path = os.path.join(job_dir, 'inductor_compile_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(compilation_config.inductor_compile_config, f)

        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if enabled_by_master("VLLM_ENABLE_PARALLEL_COMPILE_METADATA"):
            metadata_cache_dir = str(parallel_compile_metadata_dir(
                compilation_config.cache_dir, rank, dp_rank, prefix))
        else:
            metadata_cache_dir = os.path.join(
                compilation_config.local_cache_dir,
                'submodule_inputs_metadata')
        model_name = vllm_config.model_config.model.split('/')[-1]
        compiler_class = compiler_manager.compiler.__class__.__name__
        runtime_label = _runtime_shape_label(runtime_shape)

        descriptors: Dict[int, Dict[str, str]] = {}
        for source_index in source_indices:
            meta = metadata_by_index[source_index]
            graph_path = os.path.join(job_dir, f'graph_{source_index}')
            result_path = os.path.join(
                job_dir,
                f'result_{runtime_label}_graph_{source_index}.pkl')
            _save_graph_module_for_subprocess(meta['submod'], graph_path)
            if os.path.exists(result_path):
                os.remove(result_path)
            descriptors[source_index] = {
                'graph_path': graph_path,
                'result_path': result_path,
                'config_path': config_path,
                'metadata_cache_dir': metadata_cache_dir,
                'model_name': model_name,
                'submod_name': meta['submod']._get_name()
                if hasattr(meta['submod'], '_get_name') else '',
                'compiler_class': compiler_class,
                'cache_dir': cache_dir,
                'prefix': prefix,
            }

        return descriptors

    def _launch_compile_worker(
            self,
            graph_index: int,
            descriptor: Dict[str, str],
            compiler_manager: CompilerManager,
            runtime_shape: Optional[int]) -> subprocess.Popen:
        command = [
            sys.executable,
            '-m',
            'vllm.compilation.subprocess_compile_worker',
            '--graph-path',
            descriptor['graph_path'],
            '--result-path',
            descriptor['result_path'],
            '--config-path',
            descriptor['config_path'],
            '--metadata-cache-dir',
            descriptor['metadata_cache_dir'],
            '--model-name',
            descriptor['model_name'],
            '--graph-index',
            str(graph_index),
            '--compiler-class',
            descriptor['compiler_class'],
            '--cache-dir',
            descriptor['cache_dir'],
            '--prefix',
            descriptor['prefix'],
        ]
        if compiler_manager.disable_cache:
            command.append('--disable-cache')
        if runtime_shape is not None:
            command.extend(['--runtime-shape', str(runtime_shape)])

        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _compile_sources_in_subprocesses(
            self,
            source_indices: List[int],
            metadata_by_index: Dict[int, Dict[str, Any]],
            compiler_manager: CompilerManager,
            compilation_config: CompilationConfig,
            vllm_config: VllmConfig,
            runtime_shape: Optional[int]) -> tuple[set[int], dict[int, str]]:
        del vllm_config

        from torch._inductor.compile_fx_full_subproc import FullFxCompilePool
        from .compiler_interface import set_inductor_config
        from .inductor_pass import pass_context

        os.environ.setdefault('TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS',
                              str(self.max_workers))
        os.environ.setdefault('TORCHINDUCTOR_FX_COMPILE_INNER_THREADS', '1')
        worker_imports = os.environ.get(
            'TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS', '')
        required_imports = (
            'vllm._custom_ops',
            'vllm.compilation.pass_manager',
        )
        import_names = [name for name in worker_imports.split(',') if name]
        for module_name in required_imports:
            if module_name not in import_names:
                import_names.append(module_name)
        os.environ['TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS'] = ','.join(
            import_names)

        succeeded_indices: set[int] = set()
        failed_graphs: dict[int, str] = {}

        prewarm_wait_started = time.perf_counter()
        prewarm_results = [
            future.result()
            for future in FullFxCompilePool.all_prewarm_futures()
        ]
        logger.info(
            'Parallel compile timing worker_prewarm wait_s=%.6f results=%s',
            time.perf_counter() - prewarm_wait_started,
            [(result.worker_pid, result.prewarm_time_s,
              result.imports_time_s, result.cuda_init_time_s)
             for result in prewarm_results],
        )

        compile_config = dict(compilation_config.inductor_compile_config)
        set_inductor_config(compile_config, runtime_shape)
        worker_contexts = (functools.partial(pass_context, runtime_shape),)
        batch_started = time.perf_counter()
        transport_prepare_total_s = 0.0
        submit_total_s = 0.0
        completion_times: dict[int, float] = {}
        jobs = {}
        for graph_index in source_indices:
            meta = metadata_by_index[graph_index]
            try:
                # Strip runtime-only GraphModule state (weakrefs, hook tables,
                # generated code objects) while preserving the executable FX
                # graph and get_attr payloads. FakeTensor/SymInt inputs still
                # use PyTorch GraphPickler below.
                phase_started = time.perf_counter()
                transport_graph = _deserialize_graph_module_for_subprocess(
                    _serialize_graph_module_for_subprocess(meta['submod']))
                transport_prepare_s = time.perf_counter() - phase_started
                transport_prepare_total_s += transport_prepare_s

                phase_started = time.perf_counter()
                future = FullFxCompilePool.submit(
                    transport_graph,
                    meta['args'],
                    compile_config,
                    unsafe_force_cache=True,
                    worker_context_factories=worker_contexts,
                )
                submit_s = time.perf_counter() - phase_started
                submit_total_s += submit_s
                submit_finished = time.perf_counter()
                future.add_done_callback(
                    lambda _future, index=graph_index: completion_times.__setitem__(
                        index, time.perf_counter()))
                jobs[graph_index] = (future, submit_finished)
                logger.info(
                    'Parallel compile timing graph_%s '
                    'transport_prepare_s=%.6f submit_serialize_send_s=%.6f',
                    graph_index,
                    transport_prepare_s,
                    submit_s,
                )
            except Exception as exc:
                failed_graphs[graph_index] = repr(exc)

        all_submitted_at = time.perf_counter()
        logger.info(
            'Parallel compile timing submit_summary graphs=%s '
            'transport_prepare_total_s=%.6f submit_total_s=%.6f '
            'submit_phase_wall_s=%.6f',
            len(jobs),
            transport_prepare_total_s,
            submit_total_s,
            all_submitted_at - batch_started,
        )

        for graph_index, (future, submit_finished) in jobs.items():
            try:
                result = future.result()
                completed_at = completion_times.get(graph_index,
                                                    time.perf_counter())
                if len(result.artifacts) != 1:
                    raise RuntimeError(
                        'Expected one compiled FX artifact for piecewise '
                        f'graph_{graph_index}, got {len(result.artifacts)}')
                artifact = result.artifacts[0]
                handle = (artifact.cache_key, artifact.artifact_path)
                compiler_manager.record_compiled_handle(
                    graph_index, runtime_shape, handle)
                succeeded_indices.add(graph_index)
                compilation_counter.num_backend_compilations += 1
                if (compiler_manager.compiler.name.startswith('inductor')
                        and hasattr(compilation_counter,
                                    'num_inductor_compiles')):
                    compilation_counter.num_inductor_compiles += 1
                logger.info(
                    'Full-FX worker compiled graph_%s in %.2f s with key %s '
                    'worker_pid=%s serialized_input_nbytes=%s timings=%s '
                    'parent_submit_to_done_s=%.6f '
                    'dispatch_result_ipc_residual_s=%.6f.',
                    graph_index,
                    result.compile_time_s,
                    artifact.cache_key,
                    result.worker_pid,
                    result.serialized_input_nbytes,
                    result.timings,
                    completed_at - submit_finished,
                    max(0.0, completed_at - submit_finished -
                        result.timings.get('worker_total_s', 0.0)),
                )
            except Exception as exc:
                failed_graphs[graph_index] = repr(exc)

        # Start releasing compiler workers now, but defer the synchronization
        # until KV-cache sizing. Parent-side artifact loading and graph
        # assembly can safely overlap with process teardown.
        results_ready_at = time.perf_counter()
        shutdown_started = time.perf_counter()
        keep_compile_pool = os.environ.get(
            "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0").lower() in {
                "1", "true", "yes", "on"
            }
        shutdown_started_async = (
            False if keep_compile_pool
            else FullFxCompilePool.begin_shutdown_pool())
        shutdown_dispatch_s = time.perf_counter() - shutdown_started
        logger.info(
            'Parallel compile timing batch_summary submit_phase_wall_s=%.6f '
            'wait_after_submit_s=%.6f shutdown_started_async=%s '
            'shutdown_dispatch_s=%.6f '
            'subprocess_batch_total_s=%.6f',
            all_submitted_at - batch_started,
            results_ready_at - all_submitted_at,
            shutdown_started_async,
            shutdown_dispatch_s,
            time.perf_counter() - batch_started,
        )
        return succeeded_indices, failed_graphs

    def _reuse_graph(self,
                     graph_index: int,
                     source_index: int,
                     source_future,
                     compiler_manager: CompilerManager,
                     runtime_shape: Optional[int]) -> tuple[Any, str]:
        compiled_graph, _ = source_future.result()
        compiler_manager.alias_graph(graph_index, source_index, runtime_shape)
        return compiled_graph, 'reuse'

    def materialize_all(self,
                        metadata: Dict[str, Dict[str, Any]],
                        submod_names: List[str],
                        compiler_manager: CompilerManager,
                        compilation_config: CompilationConfig,
                        vllm_config: VllmConfig,
                        reuse_index_map: Optional[dict[int, int]] = None,
                        runtime_shape: Optional[int] = None) -> Dict[str, Any]:
        num_graphs = len(submod_names)
        start_time = time.time()
        reuse_index_map = reuse_index_map or {}
        if self._can_use_subprocess_compile(compiler_manager,
                                            compilation_config):
            self._load_subprocess_blacklist(compiler_manager)

        metadata_by_index = {
            meta['index']: meta
            for name, meta in metadata.items() if name in submod_names
        }
        canonical_map = {}
        for index in metadata_by_index:
            source_index = self._resolve_reuse_source(index, reuse_index_map)
            if source_index not in metadata_by_index:
                source_index = index
            canonical_map[index] = source_index

        source_futures = {}
        source_indices = sorted(set(canonical_map.values()))
        cached_source_indices = [
            source_index for source_index in source_indices
            if compiler_manager.has_cached_graph(source_index, runtime_shape)
        ]
        missing_source_indices = [
            source_index for source_index in source_indices
            if source_index not in cached_source_indices
        ]
        pre_blacklisted_indices = [
            source_index for source_index in missing_source_indices
            if source_index in self.subprocess_graph_blacklist
        ]
        subprocess_candidate_indices = [
            source_index for source_index in missing_source_indices
            if source_index not in self.subprocess_graph_blacklist
        ]

        for source_index in cached_source_indices:
            meta = metadata_by_index[source_index]
            source_futures[source_index] = self.thread_executor.submit(
                self._load_cached_source_graph,
                meta,
                compiler_manager,
                source_index,
                num_graphs,
                runtime_shape,
            )

        for source_index in pre_blacklisted_indices:
            meta = metadata_by_index[source_index]
            source_futures[source_index] = self._completed_future(
                self._compile_source_graph_serial(
                    meta,
                    compiler_manager,
                    compilation_config,
                    source_index,
                    num_graphs,
                    runtime_shape,
                ))

        if subprocess_candidate_indices:
            if self._can_use_subprocess_compile(compiler_manager,
                                                compilation_config):
                try:
                    logger.info(
                        'Compiling %s canonical graphs in external Python '
                        'subprocesses.',
                        len(subprocess_candidate_indices))
                    succeeded_indices, failed_graphs = (
                        self._compile_sources_in_subprocesses(
                            subprocess_candidate_indices,
                            metadata_by_index,
                            compiler_manager,
                            compilation_config,
                            vllm_config,
                            runtime_shape,
                        ))
                    for source_index in succeeded_indices:
                        meta = metadata_by_index[source_index]
                        source_futures[source_index] = self.thread_executor.submit(
                            self._load_freshly_compiled_source_graph,
                            meta,
                            compiler_manager,
                            source_index,
                            num_graphs,
                            runtime_shape,
                        )
                    for source_index in sorted(failed_graphs):
                        self.subprocess_graph_blacklist.add(source_index)
                        logger.warning(
                            'External subprocess compile failed for '
                            'graph_%s; falling back to in-process serial '
                            'compile for this graph only: %s',
                            source_index,
                            failed_graphs[source_index],
                        )
                        meta = metadata_by_index[source_index]
                        source_futures[source_index] = self._completed_future(
                            self._compile_source_graph_serial(
                                meta,
                                compiler_manager,
                                compilation_config,
                                source_index,
                                num_graphs,
                                runtime_shape,
                            ))
                    if failed_graphs:
                        self._save_subprocess_blacklist(compiler_manager)
                    if failed_graphs and succeeded_indices:
                        logger.info(
                            'Subprocess compile succeeded for %s canonical '
                            'graphs and fell back to serial compile for %s.',
                            len(succeeded_indices),
                            len(failed_graphs),
                        )
                    elif failed_graphs and not succeeded_indices:
                        logger.warning(
                            'All external subprocess compile jobs failed in '
                            'this run; future attempts will skip the failing '
                            'graph indices in this process.')
                except Exception as exc:
                    compilation_config.use_subprocess_compile = False
                    logger.warning(
                        'External subprocess compile failed and has been '
                        'disabled for the rest of this run; falling back to '
                        'in-process serial compile: %s',
                        exc)
                    for source_index in subprocess_candidate_indices:
                        meta = metadata_by_index[source_index]
                        source_futures[source_index] = self._completed_future(
                            self._compile_source_graph_serial(
                                meta,
                                compiler_manager,
                                compilation_config,
                                source_index,
                                num_graphs,
                                runtime_shape,
                            ))
            else:
                use_thread_parallel_compile = (
                    os.environ.get("VLLM_ENABLE_THREAD_PARALLEL_COMPILE", "").lower()
                    in {"1", "true", "yes", "on"})
                logger.info(
                    'Falling back to in-process %s compile for %s canonical '
                    'graphs.',
                    'thread-parallel' if use_thread_parallel_compile else 'serial',
                    len(subprocess_candidate_indices))
                for source_index in subprocess_candidate_indices:
                    meta = metadata_by_index[source_index]
                    compile_args = (
                        meta,
                        compiler_manager,
                        compilation_config,
                        source_index,
                        num_graphs,
                        runtime_shape,
                    )
                    if use_thread_parallel_compile:
                        source_futures[source_index] = self.thread_executor.submit(
                            self._compile_source_graph_serial, *compile_args)
                    else:
                        source_futures[source_index] = self._completed_future(
                            self._compile_source_graph_serial(*compile_args))
        elif pre_blacklisted_indices:
            logger.info(
                'Falling back to in-process serial compile for %s '
                'canonical graphs that are blacklisted from subprocess '
                'compile in this process.',
                len(pre_blacklisted_indices))

        graph_futures = {}
        for index in sorted(metadata_by_index):
            source_index = canonical_map[index]
            if source_index == index:
                graph_futures[index] = source_futures[source_index]
            else:
                graph_futures[index] = self.thread_executor.submit(
                    self._reuse_graph,
                    index,
                    source_index,
                    source_futures[source_index],
                    compiler_manager,
                    runtime_shape,
                )

        compiled_by_index: dict[int, Any] = {}
        action_counts = {'compile': 0, 'load': 0, 'reuse': 0}
        resolve_started = time.perf_counter()
        try:
            for index, future in graph_futures.items():
                compiled_graph, action = future.result()
                compiled_by_index[index] = compiled_graph
                action_counts[action] += 1
        finally:
            if (os.environ.get(
                    "VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
                        "1", "true", "yes", "on"
                    } and os.environ.get(
                        "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0").lower()
                    not in {"1", "true", "yes", "on"}):
                from torch._inductor.compile_fx_full_subproc import (
                    FullFxCompilePool,
                )
                FullFxCompilePool.begin_shutdown_pool()
        logger.info(
            'Parallel compile timing parent_load_and_alias_resolve_s=%.6f',
            time.perf_counter() - resolve_started,
        )

        elapsed = time.time() - start_time
        compilation_config.compilation_time += elapsed
        logger.info(
            'Parallel materialization finished in %.2f s '
            '(graphs=%s, compiled=%s, loaded=%s, reused=%s)',
            elapsed,
            num_graphs,
            action_counts['compile'],
            action_counts['load'],
            action_counts['reuse'],
        )

        return {
            name: compiled_by_index[metadata[name]['index']]
            for name in submod_names
        }

    def load_all_parallel(self,
                          metadata: Dict[str, Dict[str, Any]],
                          submod_names: List[str],
                          compiler_manager: CompilerManager,
                          compilation_config: CompilationConfig,
                          vllm_config: VllmConfig,
                          reuse_index_map: Optional[dict[int, int]] = None,
                          runtime_shape: Optional[int] = None) -> Dict[str, Any]:
        return self.materialize_all(metadata, submod_names, compiler_manager,
                                    compilation_config, vllm_config,
                                    reuse_index_map, runtime_shape)

    def compile_all_parallel(self,
                             metadata: Dict[str, Dict[str, Any]],
                             submod_names: List[str],
                             compiler_manager: CompilerManager,
                             compilation_config: CompilationConfig,
                             vllm_config: VllmConfig,
                             reuse_index_map: Optional[dict[int, int]] = None,
                             runtime_shape: Optional[int] = None) -> Dict[str, Any]:
        return self.materialize_all(metadata, submod_names, compiler_manager,
                                    compilation_config, vllm_config,
                                    reuse_index_map, runtime_shape)

    def shutdown(self):
        if (os.environ.get(
                "VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
                    "1", "true", "yes", "on"
                } and os.environ.get(
                    "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0").lower()
                not in {"1", "true", "yes", "on"}):
            from torch._inductor.compile_fx_full_subproc import FullFxCompilePool
            FullFxCompilePool.shutdown_pool()
        self.thread_executor.shutdown(wait=True)


class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager
    parallel_manager: ParallelCompileManager

    def __init__(
            self,
            vllm_config: VllmConfig,
            prefix: str = "",
    ):
        logger.info("Call VllmBackend.__init__()...")
        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. launguage_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag
        logger.info("VllmBackend's prefix:%s", self.prefix)
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = current_platform.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        if not hasattr(self.compilation_config,
                       "cuda_piecewise_backends_list"):
            self.compilation_config.cuda_piecewise_backends_list = {}
        # Default back to the stable path: serial compile of canonical graphs
        # plus graph reuse. Subprocess compile remains opt-in because it still
        # relies on PyTorch/Inductor internals that can fail under concurrency.
        self.compilation_config.use_subprocess_compile = (
            os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower()
            in {"1", "true", "yes", "on"})
        if self.compilation_config.use_subprocess_compile:
            logger.warning(
                "Experimental subprocess parallel compile is enabled via "
                "VLLM_ENABLE_SUBPROCESS_COMPILE=1")

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config)

        max_workers = getattr(self.compilation_config, "max_compile_workers",
                              None)
        self.parallel_manager: ParallelCompileManager = ParallelCompileManager(
            max_workers=max_workers)
        self.meta_data = {}
        self.reuse_index_map: dict[int, int] = {}

    def configure_post_pass(self):
        config = self.compilation_config
        self.post_grad_pass_manager.configure(self.vllm_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        inductor_config = config.inductor_compile_config
        PASS_KEY = "post_grad_custom_post_pass"
        if PASS_KEY in inductor_config:
            # Config should automatically wrap all inductor passes
            if isinstance(inductor_config[PASS_KEY], PostGradPassManager):
                assert (inductor_config[PASS_KEY].uuid() ==
                        self.post_grad_pass_manager.uuid())
            else:
                assert isinstance(inductor_config[PASS_KEY], InductorPass)
                self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def dynamo_frontend_runtime_identity(self) -> dict[str, Any]:
        import platform

        cached = getattr(self, "_dynamo_frontend_identity", None)
        if cached is not None:
            return cached
        with normalized_model_paths(self.vllm_config) as hash_config:
            config_hash = hash_config.compute_hash()
            compiler_hash = self.compiler_manager.compute_hash(hash_config)
        identity = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "env_hash": envs.compute_hash(),
            "config_hash": config_hash,
            "compiler_hash": compiler_hash,
            "prefix": self.prefix,
        }
        self._dynamo_frontend_identity = identity
        return identity

    def visualize_graph(self, gm: fx.GraphModule, output_path):
        from torch.fx.passes.graph_drawer import FxGraphDrawer

        drawer = FxGraphDrawer(gm, "forward")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_svg(output_path)

        logger.info(f"Graph visualization saved to {output_path}")

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        backend_started = time.perf_counter()
        if not getattr(self, "_dynamo_frontend_replaying", False):
            _dump_dynamo_fx_artifact_probe(
                graph, example_inputs, self.compilation_config.traced_files,
                self.dynamo_frontend_runtime_identity())
        from .monitor import torch_compile_start_time
        dynamo_frontend_time = time.time() - torch_compile_start_time
        logger.info("Dynamo frontend before backend entry: %.6f s",
                    dynamo_frontend_time)
        cache_analysis_started = time.perf_counter()
        timer.start("analyzing cache_dir and files")
        vllm_config = self.vllm_config
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.

            factors = []
            # 0. factors come from the env, for example, The values of
            # VLLM_PP_LAYER_PARTITION will affects the computation graph.
            env_hash = envs.compute_hash()
            factors.append(env_hash)

            # 1. factors come from the vllm_config (it mainly summarizes how the model is created)
            with normalized_model_paths(vllm_config) as hash_config:
                config_hash = hash_config.compute_hash()
            factors.append(config_hash)

            # 2. factors come from the code files that are traced by Dynamo (
            #    it mainly summarizes how the model is used in forward pass)
            forward_code_files = list(sorted(self.compilation_config.traced_files))
            self.compilation_config.traced_files.clear()
            logger.debug(
                "Traced files (to be considered for compilation cache):\n%s",
                "\n".join(forward_code_files))
            hash_content = []
            for filepath in forward_code_files:
                hash_content.append(filepath)
                if filepath == "<string>":
                    # This means the function was dynamically generated, with
                    # e.g. exec(). We can't actually check these.
                    continue
                with open(filepath) as f:
                    hash_content.append(f.read())
            import hashlib
            code_hash = hashlib.md5("\n".join(hash_content).encode(),
                                    usedforsecurity=False).hexdigest()
            factors.append(code_hash)
            # 3. compiler hash
            with normalized_model_paths(vllm_config) as hash_config:
                compiler_hash = self.compiler_manager.compute_hash(hash_config)
            factors.append(compiler_hash)

            # combine all factors to generate the cache dir
            hash_key = hashlib.md5(str(factors).encode(),
                                   usedforsecurity=False).hexdigest()[:10]

            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT,
                "torch_compile_cache",
                hash_key,
            )
            self.compilation_config.cache_dir = cache_dir

        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir

        # -------------------更新全局缓存搜索表---------------------
        # 改---全局缓存搜索表---待进一步优化
        model_path = vllm_config.model_config.model
        model_name = model_path.split('/')[-1]
        key = f"{model_path}|{model_name}"
        if enabled_by_master("VLLM_ENABLE_COMPILE_CACHE_ALIAS"):
            from .cache_reuse import canonical_model_identity
            key = f"{canonical_model_identity(model_path)}|{model_name}"
        global_cache_file = str(resource_file(
            "compile_cache_alias", "global_model_cache_map.json"))
        os.makedirs(os.path.dirname(global_cache_file), exist_ok=True)
        # 1. 文件不存在 → 创建并写入
        if not os.path.exists(global_cache_file):
            with open(global_cache_file, 'w') as f:
                json.dump({key: cache_dir}, f, indent=1)
        # 2. 文件存在
        else:
            with open(global_cache_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    data = {}
                else:
                    data = json.loads(content)
            # 2.1 键不存在 → 追加
            if key not in data:
                data[key] = cache_dir
                with open(global_cache_file, 'w') as f:
                    json.dump(data, f, indent=1)
            # 2.2 键存在
            else:
                # 缓存路径不存在 → 覆盖
                if not os.path.exists(data[key]):
                    data[key] = cache_dir
                    with open(global_cache_file, 'w') as f:
                        json.dump(data, f, indent=1)
                # 缓存路径存在 → 跳过
                else:
                    pass
        # -------------------更新全局缓存搜索表---------------------

        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir
        disable_cache = envs.VLLM_DISABLE_COMPILE_CACHE
        if disable_cache:
            logger.info("vLLM's torch.compile cache is disabled.")
        else:
            logger.info("Using cache directory: %s for vLLM's torch.compile", local_cache_dir)
        self.compiler_manager.initialize_cache(local_cache_dir, disable_cache, self.prefix)
        timer.end("analyzing cache_dir and files")
        logger.info("Compilation cache analysis time: %.6f s",
                    time.perf_counter() - cache_analysis_started)

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Tracing operational time including cache analysis: %.6f s",
                    dynamo_time)
        timer.end("Tracing")
        # self.compilation_config.compilation_time += dynamo_time
        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        timer.start("Compiling")
        timer.start("split graph")
        split_started = time.perf_counter()
        self.graph = graph
        self.configure_post_pass()
        self.split_gm, self.piecewise_graphs = split_graph(graph, self.compilation_config.splitting_ops)
        from .operator_pipeline_manifest import export_operator_pipeline_manifest
        pipeline_manifest = export_operator_pipeline_manifest(
            self.split_gm,
            self.piecewise_graphs,
            self.vllm_config.model_config.model,
            self.prefix,
            self.compilation_config.splitting_ops,
        )
        if pipeline_manifest is not None:
            logger.info("Operator pipeline manifest saved to %s",
                        pipeline_manifest)
        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]
        timer.end("split graph")
        logger.info("FX graph split time: %.6f s",
                    time.perf_counter() - split_started)
        # logger.info("split graph takes:%.3f", time.perf_counter() - start_time)
        # propagate the split graph to the piecewise backend, compile submodules with symbolic shapes
        logger.info("start compile submodules with symbolic shapes")
        start_time = time.perf_counter()
        if enabled_by_master("VLLM_ENABLE_PARALLEL_COMPILE_METADATA"):
            submodule_inputs_metadata_cache_dir = str(
                parallel_compile_metadata_dir(cache_dir, rank, dp_rank,
                                              self.prefix))
        else:
            submodule_inputs_metadata_cache_dir = os.path.join(
                local_cache_dir,
                f"submodule_inputs_metadata"
            )
        # get the metadata from fake_run or metadata_cache
        metadata_started = time.perf_counter()
        timer.start("get meta_data")
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.vllm_config, self.graph_pool,
                                    self, metadata_cache_dir=submodule_inputs_metadata_cache_dir).run(*example_inputs)
        timer.end("get meta_data")
        logger.info("Piecewise metadata restore time: %.6f s",
                    time.perf_counter() - metadata_started)
        materialization_started = time.perf_counter()
        compiled_graphs = self.parallel_manager.materialize_all(
            self.meta_data,
            submod_names_to_compile,
            self.compiler_manager,
            self.compilation_config,
            self.vllm_config,
            reuse_index_map=self.reuse_index_map,
            runtime_shape=None,
        )
        logger.info("Piecewise cache materialization time: %.6f s",
                    time.perf_counter() - materialization_started)

        assembly_started = time.perf_counter()
        piecewise_backend = resolve_obj_by_qualname(current_platform.get_piecewise_backend_cls())
        '''
        Original self.split_gm.__dict__.keys():dict_keys(['_modules', '_parameters', '_buffers', ...])
        The self.split_gm._modules includes submodules.
        In fact, self.split_gm.__dict__[name] = piecewise_backend() is to add key name to self.split_gm.__dict__.keys()
        So, the self.split_gm include orignal submodules of Fx graph ,also includes the compiled_graph.
        why the last code can correctly access and work the compiled_graph is duo to PyTorch Module's find:
        1.to first find __dict__, then _modules, so the compiled_graph can be found firstly.
        '''
        for i, name in enumerate(submod_names_to_compile):
            meta = self.meta_data[name]
            compiled_graph = compiled_graphs[name]
            # logger.info("original self.split_gm.__dict__[%s]: %s",name, self.split_gm._modules[name])
            self.split_gm.__dict__[name] = piecewise_backend(
                meta['submod'],
                self.vllm_config,
                self.graph_pool,
                i,  # index
                len(submod_names_to_compile),  # num_graphs
                meta['sym_shape_indices'],
                compiled_graph,
                self
            )
            self.vllm_config.compilation_config.cuda_piecewise_backends_list[name] = self.split_gm.__dict__[name]
            # logger.info("new self.split_gm.__dict__[%s]: %s", name, self.split_gm.__dict__[name])
            # 记录被编译的被识别并已处理的分段图数量
            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        logger.info("Piecewise backend assembly time: %.6f s",
                    time.perf_counter() - assembly_started)
        '''
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.vllm_config, self.graph_pool,
                                    self).run(*example_inputs)
        '''
        logger.info("end compile submodules with symbolic shapes, takes:%.3f", time.perf_counter() - start_time)
        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
            # use `print_readable` because it can include submodules
            src = "from __future__ import annotations\nimport torch\n" + \
                  self.split_gm.print_readable(print_output=False)
            src = src.replace("<lambda>", "GraphModule")
            with open(graph_path, "w") as f:
                f.write(src)

            logger.debug("Computation graph saved to %s", graph_path)

        self._called = True

        logger.info("VllmBackend total time: %.6f s",
                    time.perf_counter() - backend_started)
        if not self.compilation_config.use_cudagraph or not self.compilation_config.cudagraph_copy_inputs:
            logger.debug("Not use cudagraph_copy_inputs...")
            timer.end("Compiling")
            from vllm.compilation import gap_profiler
            gap_profiler.start("qwen14")
            timer.start("after_backend_return_until_first_piecewise")
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        # for weights and static buffers, they will have concrete shapes.
        # symbolic shape only happens for input tensors.
        from torch.fx.experimental.symbolic_shapes import is_symbolic
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor) and \
               any(is_symbolic(d) for d in x.size())
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]
        logger.info("Use cudagraph_copy_inputs and Return a function: copy_and_call()...")

        # this is the callable we return to Dynamo to run
        def copy_and_call(*args):
            logger.info("Call copy_and_call()...")
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]

                # copy the tensor to the static buffer
                static_tensor.copy_(runtime_tensor)

                # replace the tensor in the list_args to the static buffer
                list_args[index] = static_tensor
            return self.split_gm(*list_args)

        return copy_and_call
