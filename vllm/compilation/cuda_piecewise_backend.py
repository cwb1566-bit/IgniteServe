# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import time
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.compilation.backends import VllmBackend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.operator_pipeline_trace import operator_pipeline_range
from vllm.utils import weak_ref_tensors

logger = init_logger(__name__)
from vllm.v1.phase_profiler import timer

@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in cudagraph_capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None

    # Experimental formal-capture state. The whole shape is published only
    # after every piece has been captured by the background worker.
    concurrent_capture_in_progress: bool = False
    concurrent_capture_gate: Optional[Any] = None
    concurrent_capture_pool: Optional[Any] = None
    concurrent_capture_stream: Optional[torch.cuda.Stream] = None


class CUDAPiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: list[int],
                 compiled_graph_for_general_shape: Callable,
                 vllm_backend: VllmBackend):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: set[int] = set(
            self.compilation_config.compile_sizes)
        self.cudagraph_capture_sizes: set[int] = set(
            self.compilation_config.cudagraph_capture_sizes
        ) if self.compilation_config.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
            )

    def prepare_concurrent_capture(self, runtime_shape: int, graph_pool: Any,
                                   capture_stream: torch.cuda.Stream,
                                   capture_gate: Any) -> bool:
        """Arm one warmed, uncaptured entry for background formal capture."""
        entry = self.concrete_size_entries.get(runtime_shape)
        if (entry is None or not entry.use_cudagraph
                or entry.cudagraph is not None
                or entry.concurrent_capture_in_progress
                or entry.runnable is None
                or (entry.need_to_compile and not entry.compiled)
                or entry.num_finished_warmup <
                self.compilation_config.cudagraph_num_of_warmups):
            return False
        entry.concurrent_capture_in_progress = True
        entry.concurrent_capture_gate = capture_gate
        entry.concurrent_capture_pool = graph_pool
        entry.concurrent_capture_stream = capture_stream
        return True

    def finish_concurrent_capture(self, runtime_shape: int) -> bool:
        """Publish an entry after the shape-wide capture pass succeeds."""
        entry = self.concrete_size_entries.get(runtime_shape)
        if entry is None or entry.cudagraph is None:
            return False
        entry.concurrent_capture_in_progress = False
        entry.concurrent_capture_gate = None
        entry.concurrent_capture_pool = None
        entry.concurrent_capture_stream = None
        return True

    def abort_concurrent_capture(self, runtime_shape: int) -> None:
        """Discard a partially captured entry and restore eager fallback."""
        entry = self.concrete_size_entries.get(runtime_shape)
        if entry is None:
            return
        if entry.concurrent_capture_in_progress and entry.cudagraph is not None:
            try:
                entry.cudagraph.reset()
            except Exception:
                logger.exception(
                    "Failed to reset partial concurrent CUDA graph: "
                    "piece=%d shape=%d", self.piecewise_compile_index,
                    runtime_shape)
            entry.cudagraph = None
            entry.output = None
            entry.input_addresses = None
        entry.concurrent_capture_in_progress = False
        entry.concurrent_capture_gate = None
        entry.concurrent_capture_pool = None
        entry.concurrent_capture_stream = None

    def _run(self, mode: str, runnable: Callable, args: tuple[Any, ...]) -> Any:
        label = f"piece.{self.piecewise_compile_index}.{mode}"
        with operator_pipeline_range(label):
            return runnable(*args)

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            timer.end("compile_profile_run")
            timer.start("compile_cache_save_to_file")
            self.vllm_backend.compiler_manager.save_to_file()
            timer.end("compile_cache_save_to_file")
            timer.start("compile_monitor_end")
            end_monitoring_torch_compile(self.vllm_config)
            timer.end("compile_monitor_end")
            timer.start("after_last_piecewise_until_model_forward_end")

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            if self.is_first_graph:
                if "after_backend_return_until_first_piecewise" in timer.start_times:
                    from vllm.compilation import gap_profiler
                    gap_profiler.stop()
                    timer.end("after_backend_return_until_first_piecewise")
                timer.start("compile_profile_run")
            output = self._run("general", self.compiled_graph_for_general_shape,
                               args)
            self.check_for_ending_compilation()
            return output

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self._run("general", self.compiled_graph_for_general_shape,
                             args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        # Skip CUDA graphs if this entry doesn't use them OR
        # if we're supposed to skip them globally
        forward_context = get_forward_context()
        skip_cuda_graphs = forward_context.skip_cuda_graphs
        if not entry.use_cudagraph or skip_cuda_graphs:
            output = self._run("eager", entry.runnable, args)
            if entry.use_cudagraph and forward_context.cudagraph_runtime_warmup:
                # This target shape just completed a real eager execution.
                # Count it as the required warmup so the idle controller can
                # perform capture without repeating dummy model forwards.
                entry.num_finished_warmup = max(
                    entry.num_finished_warmup,
                    self.compilation_config.cudagraph_num_of_warmups,
                )
            return output

        capture_not_published = (
            entry.concurrent_capture_in_progress
            and (entry.concurrent_capture_gate is None
                 or not entry.concurrent_capture_gate.is_set()))
        if (capture_not_published
                and not forward_context.concurrent_cudagraph_capture):
            # Do not expose a partially captured shape to foreground requests.
            # The request continues eagerly while the background worker builds
            # and instantiates all piece graphs.
            return self._run("eager-during-concurrent-capture",
                             entry.runnable, args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                capture_budget = forward_context.cudagraph_capture_budget
                if capture_budget is not None:
                    capture_budget.defer(self.piecewise_compile_index,
                                         self.total_piecewise_compiles)
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
                        runtime_shape)
                return self._run("warmup", entry.runnable, args)

            capture_budget = forward_context.cudagraph_capture_budget
            if (capture_budget is not None
                    and not capture_budget.try_claim(
                        self.piecewise_compile_index,
                        self.total_piecewise_compiles)):
                if capture_budget.should_stop_before(
                        self.piecewise_compile_index,
                        self.total_piecewise_compiles):
                    from vllm.forward_context import CUDAGraphCaptureStop
                    raise CUDAGraphCaptureStop
                return self._run("capture-deferred", entry.runnable, args)

            if self.is_first_graph:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()
            piece_capture_started = time.perf_counter()

            with operator_pipeline_range(
                    f"piece.{self.piecewise_compile_index}.capture"), ExitStack() as stack:
                if (not self.is_first_graph
                        and not forward_context.concurrent_cudagraph_capture):
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                graph_pool = (entry.concurrent_capture_pool
                              if forward_context.concurrent_cudagraph_capture
                              else self.graph_pool)
                graph_kwargs = {}
                if forward_context.concurrent_cudagraph_capture:
                    graph_kwargs = {
                        "stream": entry.concurrent_capture_stream,
                        "capture_error_mode": "thread_local",
                        "synchronize_before_capture": False,
                        "collect_before_capture": False,
                        "empty_cache_before_capture": False,
                        "manage_default_generator": False,
                    }
                with torch.cuda.graph(cudagraph,
                                      pool=graph_pool,
                                      **graph_kwargs):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            if capture_budget is not None:
                capture_budget.mark_captured(
                    self.piecewise_compile_index,
                    time.perf_counter() - piece_capture_started,
                )

            compilation_counter.num_cudagraph_captured += 1

            if (capture_budget is not None
                    and capture_budget.should_stop_after(
                        self.piecewise_compile_index,
                        self.total_piecewise_compiles)):
                # This piece is fully instantiated and published. Stop before
                # running the uncaptured suffix; the next maintenance pass
                # replays the captured prefix and resumes at the next piece.
                from vllm.forward_context import CUDAGraphCaptureStop
                raise CUDAGraphCaptureStop

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        with operator_pipeline_range(
                f"piece.{self.piecewise_compile_index}.replay"):
            entry.cudagraph.replay()
        return entry.output
