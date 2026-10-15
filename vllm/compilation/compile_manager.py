# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-subgraph compilation, validated asset loading and result registration."""

import ast
import os
import pprint
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch.fx as fx

import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils import is_torch_equal_or_newer

from . import startup_assets as assets
from .cache_reuse import GlobalSubgraphCache, compilation_context, subgraph_identity
from .compiler_interface import (
    CompilerInterface,
    EagerAdaptor,
    InductorAdaptor,
    InductorStandaloneAdaptor,
    ThreadSafeInductorAdaptor,
)
from .counter import compilation_counter
from .input_contract import graph_key, input_contract
from .startup_resources import enabled_by_master, env_flag

logger = init_logger(__name__)


def structural_reuse_enabled() -> bool:
    if env_flag("VLLM_DISABLE_STRUCTURAL_REUSE", False):
        return False
    return enabled_by_master("VLLM_ENABLE_STRUCTURAL_REUSE")


def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    if compilation_config.use_inductor:
        if envs.VLLM_USE_STANDALONE_COMPILE and is_torch_equal_or_newer("2.8.0.dev"):
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
        self.reuse_context = None
        self.semantic_graphs = {}
        self.global_entries = {}
        self.pending_artifacts = {}

    def _cache_key(
        self, runtime_shape: Optional[int], graph_index: int
    ) -> tuple[Optional[int], int, str]:
        return runtime_shape, graph_index, self.compiler.name

    def _compiled_key(
        self, runtime_shape: Optional[int], graph_index: int
    ) -> tuple[Optional[int], int]:
        return runtime_shape, graph_index

    def _maybe_start_timer(self, graph_index: int, track_metrics: bool) -> None:
        if graph_index == 0 and track_metrics:
            global compilation_start_time
            compilation_start_time = time.time()

    def _maybe_finish_timer(
        self,
        graph_index: int,
        num_graphs: int,
        runtime_shape: Optional[int],
        track_metrics: bool,
        action: str,
    ) -> None:
        if not track_metrics or graph_index != num_graphs - 1:
            return
        elapsed = time.time() - compilation_start_time
        self.compilation_config.compilation_time += elapsed
        if runtime_shape is None:
            logger.info("%s a graph for general shape takes %.2f s", action, elapsed)
        else:
            logger.info(
                "%s a graph for shape %s takes %.2f s", action, runtime_shape, elapsed
            )

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)

    def initialize_cache(
        self,
        cache_dir: str,
        disable_cache: bool = False,
        prefix: str = "",
        identity=None,
    ):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "vllm_compile_cache.py")
        self.cache_metadata_file_path = os.path.join(
            cache_dir, "vllm_compile_reuse_cache.py"
        )
        self.contract_path = Path(cache_dir) / "input_contracts.json"
        self.cache_identity = {"schema": "compile-map-v1", "runtime": identity}
        self.cache_contracts = (
            assets.read_record(self.contract_path, {"schema": "compiled-inputs-v1"})
            or {}
        )
        valid = assets.verify_bundle(Path(cache_dir), self.cache_identity)

        if valid and not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())

        if (
            valid
            and not disable_cache
            and os.path.exists(self.cache_metadata_file_path)
        ):
            with open(self.cache_metadata_file_path) as f:
                self.compile_metadata_for_reuse = ast.literal_eval(f.read())

        self.cache = {
            k: self._disk_handle(v, restore=True) for k, v in self.cache.items()
        }
        self.compile_metadata_for_reuse = {
            k: (index, self._disk_handle(handle, restore=True))
            for k, (index, handle) in self.compile_metadata_for_reuse.items()
        }

        if (
            not disable_cache
            and isinstance(self.compiler, ThreadSafeInductorAdaptor)
            and is_torch_equal_or_newer("2.6")
            and enabled_by_master("VLLM_ENABLE_GLOBAL_SUBGRAPH_CACHE")
        ):
            self.global_cache = GlobalSubgraphCache(cache_dir, prefix)
        else:
            self.global_cache = None

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def initialize_cache_model_meta(self, model_name: str):
        del model_name

    def _disk_handle(self, handle, restore=False):
        from .startup_resources import resource_root

        if restore:
            if len(handle) == 3 and handle[2] == "repository-relative":
                return (handle[0], str(assets.member(resource_root(), handle[1])))
            return tuple(handle)
        try:
            relative = Path(handle[1]).resolve().relative_to(resource_root())
            return (handle[0], relative.as_posix(), "repository-relative")
        except (ValueError, TypeError):
            return handle  # Explicit external cache directories stay external.

    def save_to_file(self):
        if self.disable_cache:
            return
        # Publish both maps and their contracts together under one lock.
        with assets.locked(Path(self.cache_dir) / "maps"):
            self._save_to_file()

    def _save_to_file(self):
        if not self.disable_cache and self.is_cache_updated:
            printer = pprint.PrettyPrinter(indent=4)
            data = {k: self._disk_handle(v) for k, v in self.cache.items()}
            assets.atomic_write(
                Path(self.cache_file_path), printer.pformat(data).encode()
            )

        if not self.disable_cache and self.is_cache_metadata_updated:
            printer = pprint.PrettyPrinter(indent=4)
            data = {
                k: (i, self._disk_handle(h))
                for k, (i, h) in self.compile_metadata_for_reuse.items()
            }
            assets.atomic_write(
                Path(self.cache_metadata_file_path), printer.pformat(data).encode()
            )
        assets.write_json(
            self.contract_path,
            assets.record({"schema": "compiled-inputs-v1"}, self.cache_contracts),
        )
        files = [
            Path(p).name
            for p in (
                self.cache_file_path,
                self.cache_metadata_file_path,
                self.contract_path,
            )
            if Path(p).is_file()
        ]
        if files:
            assets.publish_bundle(Path(self.cache_dir), self.cache_identity, files)
        # Publish after successful parent-side binding, including subprocess
        # results. Descriptors never certify a handle that has only been queued.
        if self.global_cache is not None:
            for identity, handle, index, shape in self.pending_artifacts.values():
                self.global_cache.register(identity, handle, index, shape)
            self.pending_artifacts.clear()

    def configure_reuse(self, vllm_config):
        if not structural_reuse_enabled() and self.global_cache is None:
            return
        try:
            self.reuse_context = compilation_context(
                vllm_config,
                self.compiler,
                self.compilation_config.inductor_compile_config,
            )
        except (ValueError, TypeError, AttributeError) as exc:
            self.reuse_context = None
            logger.info("Semantic subgraph reuse disabled: %s", exc)

    def semantic_identity(self, graph, example_inputs, runtime_shape=None):
        if self.reuse_context is None:
            return None
        try:
            identity = subgraph_identity(
                graph, example_inputs, self.reuse_context, runtime_shape
            )
            assets.digest(identity)  # Reject non-deterministic/non-finite fields.
            return identity
        except (ValueError, TypeError, KeyError, AttributeError, StopIteration):
            return None

    def _remember_semantic(self, graph, inputs, index, shape, compiled, handle):
        identity = self.semantic_identity(graph, inputs, shape)
        if identity is not None and compiled is not None:
            key = assets.digest(identity)
            self.semantic_graphs[self._memory_key(identity, inputs)] = (compiled, index)
            if handle is not None:
                self.pending_artifacts[key] = (identity, handle, index, shape)

    def _memory_key(self, identity, inputs):
        # Direct callable sharing cannot run the disk loader's guard checks.
        # Keep exact current symbolic constraints for this fast path.
        return assets.digest([identity, input_contract(tuple(inputs))])

    def reuse_key(self, graph, inputs, runtime_shape=None):
        """Validated grouping key for the batch coordinator, not an asset hit."""
        identity = self.semantic_identity(graph, inputs, runtime_shape)
        if identity is not None and structural_reuse_enabled():
            return self._memory_key(identity, inputs)
        return None

    def _input_key(self, graph, example_inputs):
        try:
            contract = input_contract(tuple(example_inputs))
            # Inductor evaluates its own stored guards at load time. Unrelated
            # guards accumulated later in Dynamo must not change this key.
            contract.pop("guards")
            return assets.digest([graph_key(graph), contract])
        except (ValueError, TypeError, AttributeError):
            return None

    def has_cached_graph(
        self,
        graph_index: int,
        runtime_shape: Optional[int] = None,
        graph=None,
        example_inputs=(),
    ) -> bool:
        if self._cache_key(runtime_shape, graph_index) in self.cache:
            return True
        return self._global_entry(graph, example_inputs, runtime_shape) is not None

    def _global_entry(self, graph, inputs, shape):
        if (
            self.global_cache is None
            or graph is None
            or not env_flag("VLLM_ENABLE_GLOBAL_SUBGRAPH_DIRECT_LOOKUP", True)
        ):
            return None
        identity = self.semantic_identity(graph, inputs, shape)
        if identity is None:
            return None
        key = assets.digest(identity)
        if key not in self.global_entries:
            self.global_entries[key] = self.global_cache.lookup(identity)
        return self.global_entries[key]

    def get_cached_handle(
        self, graph_index: int, runtime_shape: Optional[int] = None
    ) -> Any:
        return self.cache.get(self._cache_key(runtime_shape, graph_index))

    def get_compiled_graph(
        self, graph_index: int, runtime_shape: Optional[int] = None
    ) -> Any:
        return self.compiled_graph_list.get(
            self._compiled_key(runtime_shape, graph_index)
        )

    def get_reuse_source(
        self, graph_index: int, runtime_shape: Optional[int] = None
    ) -> int:
        reuse_entry = self.compile_metadata_for_reuse.get(
            self._cache_key(runtime_shape, graph_index)
        )
        if reuse_entry is None:
            return graph_index
        return reuse_entry[0]

    def alias_graph(
        self, graph_index: int, source_index: int, runtime_shape: Optional[int] = None
    ) -> None:
        source_handle = self.get_cached_handle(source_index, runtime_shape)
        source_graph = self.get_compiled_graph(source_index, runtime_shape)
        if source_handle is not None:
            self.cache[self._cache_key(runtime_shape, graph_index)] = source_handle
            self.cache_contracts[str(self._cache_key(runtime_shape, graph_index))] = (
                self.cache_contracts.get(
                    str(self._cache_key(runtime_shape, source_index))
                )
            )
            self.compile_metadata_for_reuse[
                self._cache_key(runtime_shape, graph_index)
            ] = (source_index, source_handle)
            self.is_cache_updated = True
            self.is_cache_metadata_updated = True
        if source_graph is not None:
            self.compiled_graph_list[self._compiled_key(runtime_shape, graph_index)] = (
                source_graph
            )

    def _record_compiled_artifact(
        self,
        graph: fx.GraphModule,
        graph_index: int,
        runtime_shape: Optional[int],
        compiled_graph: Optional[Any],
        handle: Any,
        example_inputs=(),
    ) -> None:
        if compiled_graph is not None:
            self.compiled_graph_list[self._compiled_key(runtime_shape, graph_index)] = (
                compiled_graph
            )
            self._remember_semantic(
                graph,
                example_inputs,
                graph_index,
                runtime_shape,
                compiled_graph,
                handle,
            )

        if handle is None:
            return

        cache_key = self._cache_key(runtime_shape, graph_index)
        self.cache_contracts[str(cache_key)] = self._input_key(graph, example_inputs)
        self.cache[cache_key] = handle
        self.is_cache_updated = True

        reuse_source = self.get_reuse_source(graph_index, runtime_shape)
        if reuse_source == graph_index:
            for (
                cached_shape,
                cached_graph_index,
                compiler_name,
            ), cached_handle in self.cache.items():
                if (
                    cached_shape == runtime_shape
                    and compiler_name == self.compiler.name
                    and cached_graph_index != graph_index
                    and cached_handle == handle
                ):
                    reuse_source = self.get_reuse_source(
                        cached_graph_index, runtime_shape
                    )
                    break

        self.compile_metadata_for_reuse[cache_key] = (reuse_source, handle)
        self.is_cache_metadata_updated = True

    def supports_external_compile_handles(self) -> bool:
        return not self.disable_cache and isinstance(
            self.compiler,
            (InductorAdaptor, ThreadSafeInductorAdaptor, InductorStandaloneAdaptor),
        )

    def record_compiled_handle(
        self,
        graph_index: int,
        runtime_shape: Optional[int],
        handle: Any,
        graph=None,
        example_inputs=(),
    ) -> None:
        if handle is None:
            raise RuntimeError(
                "Subprocess compile requires a persistent compiler handle."
            )
        cache_key = self._cache_key(runtime_shape, graph_index)
        self.cache_contracts[str(cache_key)] = (
            self._input_key(graph, example_inputs) if graph is not None else None
        )
        self.cache[cache_key] = handle
        self.compile_metadata_for_reuse[cache_key] = (graph_index, handle)
        self.is_cache_updated = True
        self.is_cache_metadata_updated = True

    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
        track_metrics: bool = True,
        count_backend_compilations: bool = True,
    ) -> Optional[Callable]:
        self._maybe_start_timer(graph_index, track_metrics)

        identity = self.semantic_identity(graph, example_inputs, runtime_shape)
        reusable = (
            self.semantic_graphs.get(self._memory_key(identity, example_inputs))
            if identity and structural_reuse_enabled()
            else None
        )
        if reusable is not None:
            compiled_graph, source = reusable
            self.alias_graph(graph_index, source, runtime_shape)
            return compiled_graph

        handle = self.get_cached_handle(graph_index, runtime_shape)
        current_key = self._input_key(graph, example_inputs)
        if (
            current_key is None
            or self.cache_contracts.get(
                str(self._cache_key(runtime_shape, graph_index))
            )
            != current_key
        ):
            self.cache.pop(self._cache_key(runtime_shape, graph_index), None)
            self.compile_metadata_for_reuse.pop(
                self._cache_key(runtime_shape, graph_index), None
            )
            handle = None
        reuse_source = (
            self.get_reuse_source(graph_index, runtime_shape) if handle else graph_index
        )
        if reuse_source != graph_index:
            reusable_graph = self.get_compiled_graph(reuse_source, runtime_shape)
            if reusable_graph is not None:
                self.alias_graph(graph_index, reuse_source, runtime_shape)
                logger.debug(
                    "Reuse loaded graph %s for graph %s on shape %s.",
                    reuse_source,
                    graph_index,
                    str(runtime_shape),
                )
                self._maybe_finish_timer(
                    graph_index, num_graphs, runtime_shape, track_metrics, "Loading"
                )
                return reusable_graph

        compiled_graph = None
        if handle is not None:
            try:
                compiled_graph = self.compiler.load(
                    handle, graph, example_inputs, graph_index, runtime_shape
                )
            except Exception as exc:
                logger.warning(
                    "Local compiled asset rejected for graph %s: %s", graph_index, exc
                )
                self.cache.pop(self._cache_key(runtime_shape, graph_index), None)
                self.compile_metadata_for_reuse.pop(
                    self._cache_key(runtime_shape, graph_index), None
                )
        if compiled_graph is None:
            entry = self._global_entry(graph, example_inputs, runtime_shape)
            if entry is not None:
                try:
                    handle = (
                        entry["handle"][0],
                        str(assets.member(self.global_cache.root, entry["handle"][1])),
                    )
                    compiled_graph = self.compiler.load(
                        handle,
                        graph,
                        example_inputs,
                        graph_index,
                        runtime_shape,
                        artifact_payloads=self.global_cache.payloads(entry),
                    )
                    if compiled_graph is not None:
                        self._record_compiled_artifact(
                            graph,
                            graph_index,
                            runtime_shape,
                            compiled_graph,
                            handle,
                            example_inputs,
                        )
                        logger.info(
                            "Global subgraph hit graph_%s key=%s",
                            graph_index,
                            assets.digest(identity),
                        )
                except Exception as exc:
                    logger.warning(
                        "Global compiled asset rejected for graph %s: %s",
                        graph_index,
                        exc,
                    )
                    self.global_entries[assets.digest(identity)] = None
        if compiled_graph is None:
            return None
        if count_backend_compilations:
            compilation_counter.num_backend_compilations += 1
        self._remember_semantic(
            graph, example_inputs, graph_index, runtime_shape, compiled_graph, handle
        )
        self.compiled_graph_list[self._compiled_key(runtime_shape, graph_index)] = (
            compiled_graph
        )
        logger.debug(
            "Loaded graph %s for shape %s from %s via handle %s.",
            graph_index,
            str(runtime_shape),
            self.compiler.name,
            handle,
        )
        self._maybe_finish_timer(
            graph_index, num_graphs, runtime_shape, track_metrics, "Loading"
        )
        return compiled_graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        additional_inductor_config,
        compilation_config: CompilationConfig,
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
        track_metrics: bool = True,
    ) -> Any:
        del compilation_config
        self._maybe_start_timer(graph_index, track_metrics)
        compilation_counter.num_backend_compilations += 1

        compiled_graph = self.load(
            graph,
            example_inputs,
            graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
            count_backend_compilations=False,
        )
        if compiled_graph is not None:
            self._maybe_finish_timer(
                graph_index, num_graphs, runtime_shape, track_metrics, "Loading"
            )
            return compiled_graph

        reuse_source = self.get_reuse_source(graph_index, runtime_shape)
        if reuse_source != graph_index:
            reusable_graph = self.get_compiled_graph(reuse_source, runtime_shape)
            if reusable_graph is not None:
                self.alias_graph(graph_index, reuse_source, runtime_shape)
                logger.debug(
                    "Reuse compiled graph %s for graph %s on shape %s.",
                    reuse_source,
                    graph_index,
                    str(runtime_shape),
                )
                self._maybe_finish_timer(
                    graph_index, num_graphs, runtime_shape, track_metrics, "Compiling"
                )
                return reusable_graph

        if isinstance(self.compiler, (InductorAdaptor, ThreadSafeInductorAdaptor)):
            maybe_key = None
        else:
            maybe_key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"

        compiled_graph, handle = self.compiler.compile(
            graph,
            example_inputs,
            additional_inductor_config,
            runtime_shape,
            maybe_key,
            graph_index=graph_index,
        )
        assert compiled_graph is not None, "Failed to compile the graph"

        self._record_compiled_artifact(
            graph, graph_index, runtime_shape, compiled_graph, handle, example_inputs
        )
        if handle is not None:
            if graph_index == 0:
                logger.info(
                    "Cache the graph of shape %s for later use", str(runtime_shape)
                )
            logger.debug(
                "Stored graph %s for shape %s from %s via handle %s.",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )

        self._maybe_finish_timer(
            graph_index, num_graphs, runtime_shape, track_metrics, "Compiling"
        )
        return compiled_graph
