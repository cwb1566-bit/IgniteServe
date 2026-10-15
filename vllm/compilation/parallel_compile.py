# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch coordination: canonical tasks, asynchronous materialization and fallback."""

import functools
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.v1.phase_profiler import timer

from .compile_manager import CompilerManager
from .compiler_interface import FullFxCompileBackend
from .counter import compilation_counter
from .fx_transport import (
    _deserialize_graph_module_for_subprocess,
    _runtime_shape_label,
    _save_graph_module_for_subprocess,
    _serialize_graph_module_for_subprocess,
)
from .startup_resources import enabled_by_master, parallel_compile_metadata_dir

logger = init_logger(__name__)


class ParallelCompileManager:
    """Coordinate a batch; CompilerManager validates individual artifacts.

    Input collection finishes before materialization, as in the original path.
    Cache loads overlap compilation; all process jobs are submitted before wait.
    """

    def compile_subgraphs(
        self,
        collector_factory,
        example_inputs,
        compiler_manager,
        compilation_config,
        vllm_config,
    ):
        """Single engine-facing entry; keep metadata and Future ownership local."""
        started = time.perf_counter()
        timer.start("get meta_data")
        collector = collector_factory()
        collector.run(*example_inputs)
        timer.end("get meta_data")
        logger.info(
            "Piecewise metadata restore time: %.6f s", time.perf_counter() - started
        )
        started = time.perf_counter()
        compiled = self.materialize_all(
            collector.metadata,
            collector.compile_submod_names,
            compiler_manager,
            compilation_config,
            vllm_config,
            reuse_index_map=collector.reuse_index_map,
            runtime_shape=None,
        )
        logger.info(
            "Piecewise cache materialization time: %.6f s",
            time.perf_counter() - started,
        )
        return compiled, collector

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(os.cpu_count() or 1, 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.subprocess_graph_blacklist: set[int] = set()
        self._blacklist_cache_dir: Optional[str] = None
        logger.info(
            "ParallelCompileManager initialized with %s workers "
            "(thread cache load + optional process compile)",
            self.max_workers,
        )
        if os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            FullFxCompileBackend.configure(self.max_workers, prewarm_cuda=True)
            try:
                FullFxCompileBackend.prewarm()
                logger.info(
                    "Started and queued import prewarm for Full-FX compiler "
                    "workers during model init."
                )
            except Exception as exc:
                logger.warning(
                    "Failed to pre-start Full-FX compiler worker pool: %s", exc
                )

    def _subprocess_blacklist_path(self, compiler_manager: CompilerManager) -> str:
        return os.path.join(
            compiler_manager.cache_dir, "subprocess_compile_blacklist.json"
        )

    def _load_subprocess_blacklist(self, compiler_manager: CompilerManager) -> None:
        cache_dir = compiler_manager.cache_dir
        if not cache_dir:
            return
        if self._blacklist_cache_dir == cache_dir:
            return

        blacklist_path = self._subprocess_blacklist_path(compiler_manager)
        blacklist: set[int] = set()
        if os.path.exists(blacklist_path):
            try:
                with open(blacklist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                blacklist = {int(index) for index in data.get("graph_indices", [])}
            except Exception as exc:
                logger.warning(
                    "Failed to load subprocess compile blacklist from %s: %s",
                    blacklist_path,
                    exc,
                )

        self.subprocess_graph_blacklist = blacklist
        self._blacklist_cache_dir = cache_dir

    def _save_subprocess_blacklist(self, compiler_manager: CompilerManager) -> None:
        cache_dir = compiler_manager.cache_dir
        if not cache_dir:
            return
        blacklist_path = self._subprocess_blacklist_path(compiler_manager)
        with open(blacklist_path, "w", encoding="utf-8") as f:
            json.dump(
                {"graph_indices": sorted(self.subprocess_graph_blacklist)},
                f,
                indent=2,
            )

    def _resolve_reuse_source(
        self, graph_index: int, reuse_index_map: dict[int, int]
    ) -> int:
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

    def _load_cached_source_graph(
        self,
        meta: Dict[str, Any],
        compiler_manager: CompilerManager,
        graph_index: int,
        num_graphs: int,
        runtime_shape: Optional[int],
    ) -> tuple[Any, str]:
        compiled_graph = compiler_manager.load(
            meta["submod"],
            meta["args"],
            graph_index=graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
        )
        return compiled_graph, "load"

    def _compile_source_graph_serial(
        self,
        meta: Dict[str, Any],
        compiler_manager: CompilerManager,
        compilation_config: CompilationConfig,
        graph_index: int,
        num_graphs: int,
        runtime_shape: Optional[int],
    ) -> tuple[Any, str]:
        compiled_graph = compiler_manager.compile(
            meta["submod"],
            meta["args"],
            compilation_config.inductor_compile_config,
            compilation_config,
            graph_index=graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
        )
        return compiled_graph, "compile"

    def _load_freshly_compiled_source_graph(
        self,
        meta: Dict[str, Any],
        compiler_manager: CompilerManager,
        graph_index: int,
        num_graphs: int,
        runtime_shape: Optional[int],
    ) -> tuple[Any, str]:
        started = time.perf_counter()
        compiled_graph = compiler_manager.load(
            meta["submod"],
            meta["args"],
            graph_index=graph_index,
            num_graphs=num_graphs,
            runtime_shape=runtime_shape,
            track_metrics=False,
            count_backend_compilations=False,
        )
        logger.info(
            "Parallel compile timing graph_%s parent_artifact_load_s=%.6f",
            graph_index,
            time.perf_counter() - started,
        )
        return compiled_graph, "compile"

    def _can_use_subprocess_compile(
        self, compiler_manager: CompilerManager, compilation_config: CompilationConfig
    ) -> bool:
        # Keep subprocess compile opt-in only. The current subprocess path is
        # experimental and can regress startup latency when it fails.
        if not getattr(compilation_config, "use_subprocess_compile", False):
            return False
        if not compiler_manager.supports_external_compile_handles():
            return False
        if not getattr(compiler_manager, "cache_dir", None):
            return False
        if not getattr(compilation_config, "local_cache_dir", None):
            return False
        return True

    def _prepare_subprocess_compile_assets(
        self,
        source_indices: List[int],
        metadata_by_index: Dict[int, Dict[str, Any]],
        compiler_manager: CompilerManager,
        compilation_config: CompilationConfig,
        vllm_config: VllmConfig,
        runtime_shape: Optional[int],
    ) -> Dict[int, Dict[str, str]]:
        cache_dir = compiler_manager.cache_dir
        prefix = os.path.basename(cache_dir)
        job_dir = os.path.join(cache_dir, "subprocess_compile_jobs")
        os.makedirs(job_dir, exist_ok=True)

        config_path = os.path.join(job_dir, "inductor_compile_config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(compilation_config.inductor_compile_config, f)

        rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if enabled_by_master("VLLM_ENABLE_PARALLEL_COMPILE_METADATA"):
            metadata_cache_dir = str(
                parallel_compile_metadata_dir(
                    compilation_config.cache_dir, rank, dp_rank, prefix, vllm_config
                )
            )
        else:
            metadata_cache_dir = os.path.join(
                compilation_config.local_cache_dir, "submodule_inputs_metadata"
            )
        model_name = vllm_config.model_config.model.split("/")[-1]
        compiler_class = compiler_manager.compiler.__class__.__name__
        runtime_label = _runtime_shape_label(runtime_shape)

        descriptors: Dict[int, Dict[str, str]] = {}
        for source_index in source_indices:
            meta = metadata_by_index[source_index]
            graph_path = os.path.join(job_dir, f"graph_{source_index}")
            result_path = os.path.join(
                job_dir, f"result_{runtime_label}_graph_{source_index}.pkl"
            )
            _save_graph_module_for_subprocess(meta["submod"], graph_path)
            if os.path.exists(result_path):
                os.remove(result_path)
            descriptors[source_index] = {
                "graph_path": graph_path,
                "result_path": result_path,
                "config_path": config_path,
                "metadata_cache_dir": metadata_cache_dir,
                "model_name": model_name,
                "submod_name": meta["submod"]._get_name()
                if hasattr(meta["submod"], "_get_name")
                else "",
                "compiler_class": compiler_class,
                "cache_dir": cache_dir,
                "prefix": prefix,
            }

        return descriptors

    def _launch_compile_worker(
        self,
        graph_index: int,
        descriptor: Dict[str, str],
        compiler_manager: CompilerManager,
        runtime_shape: Optional[int],
    ) -> subprocess.Popen:
        command = [
            sys.executable,
            "-m",
            "vllm.compilation.subprocess_compile_worker",
            "--graph-path",
            descriptor["graph_path"],
            "--result-path",
            descriptor["result_path"],
            "--config-path",
            descriptor["config_path"],
            "--metadata-cache-dir",
            descriptor["metadata_cache_dir"],
            "--model-name",
            descriptor["model_name"],
            "--graph-index",
            str(graph_index),
            "--compiler-class",
            descriptor["compiler_class"],
            "--cache-dir",
            descriptor["cache_dir"],
            "--prefix",
            descriptor["prefix"],
        ]
        if compiler_manager.disable_cache:
            command.append("--disable-cache")
        if runtime_shape is not None:
            command.extend(["--runtime-shape", str(runtime_shape)])

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
        runtime_shape: Optional[int],
    ) -> tuple[set[int], dict[int, str]]:
        del vllm_config

        from .compiler_interface import set_inductor_config
        from .inductor_pass import pass_context

        FullFxCompileBackend.configure(self.max_workers)

        succeeded_indices: set[int] = set()
        failed_graphs: dict[int, str] = {}

        prewarm_wait_started = time.perf_counter()
        prewarm_results = [
            future.result() for future in FullFxCompileBackend.prewarm_futures()
        ]
        logger.info(
            "Parallel compile timing worker_prewarm wait_s=%.6f results=%s",
            time.perf_counter() - prewarm_wait_started,
            [
                (
                    result.worker_pid,
                    result.prewarm_time_s,
                    result.imports_time_s,
                    result.cuda_init_time_s,
                )
                for result in prewarm_results
            ],
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
                    _serialize_graph_module_for_subprocess(meta["submod"])
                )
                transport_prepare_s = time.perf_counter() - phase_started
                transport_prepare_total_s += transport_prepare_s

                phase_started = time.perf_counter()
                future = FullFxCompileBackend.submit(
                    transport_graph,
                    meta["args"],
                    compile_config,
                    worker_contexts,
                )
                submit_s = time.perf_counter() - phase_started
                submit_total_s += submit_s
                submit_finished = time.perf_counter()
                future.add_done_callback(
                    lambda _future, index=graph_index: completion_times.__setitem__(
                        index, time.perf_counter()
                    )
                )
                jobs[graph_index] = (future, submit_finished)
                logger.info(
                    "Parallel compile timing graph_%s "
                    "transport_prepare_s=%.6f submit_serialize_send_s=%.6f",
                    graph_index,
                    transport_prepare_s,
                    submit_s,
                )
            except Exception as exc:
                failed_graphs[graph_index] = repr(exc)

        all_submitted_at = time.perf_counter()
        logger.info(
            "Parallel compile timing submit_summary graphs=%s "
            "transport_prepare_total_s=%.6f submit_total_s=%.6f "
            "submit_phase_wall_s=%.6f",
            len(jobs),
            transport_prepare_total_s,
            submit_total_s,
            all_submitted_at - batch_started,
        )

        for graph_index, (future, submit_finished) in jobs.items():
            try:
                result = future.result()
                completed_at = completion_times.get(graph_index, time.perf_counter())
                if len(result.artifacts) != 1:
                    raise RuntimeError(
                        "Expected one compiled FX artifact for piecewise "
                        f"graph_{graph_index}, got {len(result.artifacts)}"
                    )
                artifact = result.artifacts[0]
                handle = (artifact.cache_key, artifact.artifact_path)
                compiler_manager.record_compiled_handle(
                    graph_index,
                    runtime_shape,
                    handle,
                    metadata_by_index[graph_index]["submod"],
                    metadata_by_index[graph_index]["args"],
                )
                succeeded_indices.add(graph_index)
                compilation_counter.num_backend_compilations += 1
                if compiler_manager.compiler.name.startswith("inductor") and hasattr(
                    compilation_counter, "num_inductor_compiles"
                ):
                    compilation_counter.num_inductor_compiles += 1
                logger.info(
                    "Full-FX worker compiled graph_%s in %.2f s with key %s "
                    "worker_pid=%s serialized_input_nbytes=%s timings=%s "
                    "parent_submit_to_done_s=%.6f "
                    "dispatch_result_ipc_residual_s=%.6f.",
                    graph_index,
                    result.compile_time_s,
                    artifact.cache_key,
                    result.worker_pid,
                    result.serialized_input_nbytes,
                    result.timings,
                    completed_at - submit_finished,
                    max(
                        0.0,
                        completed_at
                        - submit_finished
                        - result.timings.get("worker_total_s", 0.0),
                    ),
                )
            except Exception as exc:
                failed_graphs[graph_index] = repr(exc)

        # Start releasing compiler workers now, but defer the synchronization
        # until KV-cache sizing. Parent-side artifact loading and graph
        # assembly can safely overlap with process teardown.
        results_ready_at = time.perf_counter()
        shutdown_started = time.perf_counter()
        keep_compile_pool = os.environ.get(
            "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0"
        ).lower() in {"1", "true", "yes", "on"}
        shutdown_started_async = (
            False if keep_compile_pool else FullFxCompileBackend.begin_shutdown()
        )
        shutdown_dispatch_s = time.perf_counter() - shutdown_started
        logger.info(
            "Parallel compile timing batch_summary submit_phase_wall_s=%.6f "
            "wait_after_submit_s=%.6f shutdown_started_async=%s "
            "shutdown_dispatch_s=%.6f "
            "subprocess_batch_total_s=%.6f",
            all_submitted_at - batch_started,
            results_ready_at - all_submitted_at,
            shutdown_started_async,
            shutdown_dispatch_s,
            time.perf_counter() - batch_started,
        )
        return succeeded_indices, failed_graphs

    def _reuse_graph(
        self,
        graph_index: int,
        source_index: int,
        source_future,
        compiler_manager: CompilerManager,
        runtime_shape: Optional[int],
    ) -> tuple[Any, str]:
        compiled_graph, _ = source_future.result()
        compiler_manager.alias_graph(graph_index, source_index, runtime_shape)
        return compiled_graph, "reuse"

    def materialize_all(
        self,
        metadata: Dict[str, Dict[str, Any]],
        submod_names: List[str],
        compiler_manager: CompilerManager,
        compilation_config: CompilationConfig,
        vllm_config: VllmConfig,
        reuse_index_map: Optional[dict[int, int]] = None,
        runtime_shape: Optional[int] = None,
    ) -> Dict[str, Any]:
        num_graphs = len(submod_names)
        start_time = time.time()
        reuse_index_map = reuse_index_map or {}
        if self._can_use_subprocess_compile(compiler_manager, compilation_config):
            self._load_subprocess_blacklist(compiler_manager)

        metadata_by_index = {
            meta["index"]: meta
            for name, meta in metadata.items()
            if name in submod_names
        }
        canonical_map = {}
        semantic_sources = {}
        for index in metadata_by_index:
            source_index = self._resolve_reuse_source(index, reuse_index_map)
            if source_index not in metadata_by_index:
                source_index = index
            meta = metadata_by_index[index]
            reuse_key = compiler_manager.reuse_key(
                meta["submod"], meta["args"], runtime_shape
            )
            if reuse_key is not None:
                source_index = semantic_sources.setdefault(reuse_key, index)
            canonical_map[index] = source_index

        source_futures = {}
        source_indices = sorted(set(canonical_map.values()))
        cached_source_indices = [
            source_index
            for source_index in source_indices
            if compiler_manager.has_cached_graph(
                source_index,
                runtime_shape,
                metadata_by_index[source_index]["submod"],
                metadata_by_index[source_index]["args"],
            )
        ]
        missing_source_indices = [
            source_index
            for source_index in source_indices
            if source_index not in cached_source_indices
        ]
        pre_blacklisted_indices = [
            source_index
            for source_index in missing_source_indices
            if source_index in self.subprocess_graph_blacklist
        ]
        subprocess_candidate_indices = [
            source_index
            for source_index in missing_source_indices
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
                )
            )

        if subprocess_candidate_indices:
            if self._can_use_subprocess_compile(compiler_manager, compilation_config):
                try:
                    logger.info(
                        "Compiling %s canonical graphs in external Python "
                        "subprocesses.",
                        len(subprocess_candidate_indices),
                    )
                    succeeded_indices, failed_graphs = (
                        self._compile_sources_in_subprocesses(
                            subprocess_candidate_indices,
                            metadata_by_index,
                            compiler_manager,
                            compilation_config,
                            vllm_config,
                            runtime_shape,
                        )
                    )
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
                            "External subprocess compile failed for "
                            "graph_%s; falling back to in-process serial "
                            "compile for this graph only: %s",
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
                            )
                        )
                    if failed_graphs:
                        self._save_subprocess_blacklist(compiler_manager)
                    if failed_graphs and succeeded_indices:
                        logger.info(
                            "Subprocess compile succeeded for %s canonical "
                            "graphs and fell back to serial compile for %s.",
                            len(succeeded_indices),
                            len(failed_graphs),
                        )
                    elif failed_graphs and not succeeded_indices:
                        logger.warning(
                            "All external subprocess compile jobs failed in "
                            "this run; future attempts will skip the failing "
                            "graph indices in this process."
                        )
                except Exception as exc:
                    compilation_config.use_subprocess_compile = False
                    logger.warning(
                        "External subprocess compile failed and has been "
                        "disabled for the rest of this run; falling back to "
                        "in-process serial compile: %s",
                        exc,
                    )
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
                            )
                        )
            else:
                use_thread_parallel_compile = os.environ.get(
                    "VLLM_ENABLE_THREAD_PARALLEL_COMPILE", ""
                ).lower() in {"1", "true", "yes", "on"}
                logger.info(
                    "Falling back to in-process %s compile for %s canonical graphs.",
                    "thread-parallel" if use_thread_parallel_compile else "serial",
                    len(subprocess_candidate_indices),
                )
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
                            self._compile_source_graph_serial, *compile_args
                        )
                    else:
                        source_futures[source_index] = self._completed_future(
                            self._compile_source_graph_serial(*compile_args)
                        )
        elif pre_blacklisted_indices:
            logger.info(
                "Falling back to in-process serial compile for %s "
                "canonical graphs that are blacklisted from subprocess "
                "compile in this process.",
                len(pre_blacklisted_indices),
            )

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
        action_counts = {"compile": 0, "load": 0, "reuse": 0}
        resolve_started = time.perf_counter()
        try:
            for index, future in graph_futures.items():
                compiled_graph, action = future.result()
                if compiled_graph is None:
                    source = canonical_map[index]
                    compiled_graph = compiled_by_index.get(source)
                    if compiled_graph is None:
                        logger.warning(
                            "Artifact rejected for graph %s; compiling with current inputs",
                            index,
                        )
                        compiled_graph, action = self._compile_source_graph_serial(
                            metadata_by_index[index],
                            compiler_manager,
                            compilation_config,
                            index,
                            num_graphs,
                            runtime_shape,
                        )
                    else:
                        compiler_manager.alias_graph(index, source, runtime_shape)
                        action = "reuse"
                compiled_by_index[index] = compiled_graph
                action_counts[action] += 1
        finally:
            if os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            } and os.environ.get(
                "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0"
            ).lower() not in {"1", "true", "yes", "on"}:
                FullFxCompileBackend.begin_shutdown()
        logger.info(
            "Parallel compile timing parent_load_and_alias_resolve_s=%.6f",
            time.perf_counter() - resolve_started,
        )

        elapsed = time.time() - start_time
        compilation_config.compilation_time += elapsed
        logger.info(
            "Parallel materialization finished in %.2f s "
            "(graphs=%s, compiled=%s, loaded=%s, reused=%s)",
            elapsed,
            num_graphs,
            action_counts["compile"],
            action_counts["load"],
            action_counts["reuse"],
        )

        return {
            name: compiled_by_index[metadata[name]["index"]] for name in submod_names
        }

    def load_all_parallel(
        self,
        metadata: Dict[str, Dict[str, Any]],
        submod_names: List[str],
        compiler_manager: CompilerManager,
        compilation_config: CompilationConfig,
        vllm_config: VllmConfig,
        reuse_index_map: Optional[dict[int, int]] = None,
        runtime_shape: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.materialize_all(
            metadata,
            submod_names,
            compiler_manager,
            compilation_config,
            vllm_config,
            reuse_index_map,
            runtime_shape,
        )

    def compile_all_parallel(
        self,
        metadata: Dict[str, Dict[str, Any]],
        submod_names: List[str],
        compiler_manager: CompilerManager,
        compilation_config: CompilationConfig,
        vllm_config: VllmConfig,
        reuse_index_map: Optional[dict[int, int]] = None,
        runtime_shape: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.materialize_all(
            metadata,
            submod_names,
            compiler_manager,
            compilation_config,
            vllm_config,
            reuse_index_map,
            runtime_shape,
        )

    def shutdown(self):
        if os.environ.get("VLLM_ENABLE_SUBPROCESS_COMPILE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        } and os.environ.get(
            "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0"
        ).lower() not in {"1", "true", "yes", "on"}:
            FullFxCompileBackend.shutdown()
        self.thread_executor.shutdown(wait=True)
