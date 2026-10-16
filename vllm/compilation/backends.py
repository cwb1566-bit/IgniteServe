# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM FX entry, graph partitioning and engine-specific callable assembly."""

import json
import os
import pickle
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import resolve_obj_by_qualname
from vllm.v1.phase_profiler import timer

from . import startup_assets as assets
from .cache_reuse import normalized_model_paths

# Compatibility exports for existing callers and serialized references.
from .compile_manager import CompilerManager, make_compiler, structural_reuse_enabled
from .counter import compilation_counter
from .fx_transport import (
    _deserialize_graph_module_for_subprocess,
    _load_graph_module_for_subprocess,
    _serialize_graph_module_for_subprocess,
)
from .inductor_pass import InductorPass
from .input_metadata import (
    SubmoduleInputMetadata,
    SubmoduleInputSerializer,
    SymIntMetadata,
    TensorMetadata,
)
from .parallel_compile import ParallelCompileManager
from .pass_manager import PostGradPassManager
from .piecewise_inputs import PiecewiseCompileInterpreter
from .startup_resources import (
    enabled_by_master,
    parallel_compile_metadata_dir,
    parallel_compile_metadata_identity,
    resource_dir,
)

# Preserve the original import locations for callers of the monolithic backend.
__all__ = ["VllmBackend", "CompilerManager", "ParallelCompileManager",
           "PiecewiseCompileInterpreter", "SubmoduleInputSerializer",
           "TensorMetadata", "SymIntMetadata", "SubmoduleInputMetadata",
           "make_compiler", "structural_reuse_enabled", "set_model_tag",
           "split_graph", "_serialize_graph_module_for_subprocess",
           "_deserialize_graph_module_for_subprocess", "_load_graph_module_for_subprocess"]

logger = init_logger(__name__)


def _dump_dynamo_fx_artifact_probe(graph: fx.GraphModule,
                                   example_inputs: list[Any],
                                   traced_files: Sequence[str],
                                   runtime_identity: dict[str, Any],
                                   output_dir=None) -> None:
    """Persist the pre-backend FX graph with PyTorch's wire serializer."""
    output_dir = output_dir or (os.getenv("VLLM_DYNAMO_FRONTEND_CACHE_DIR")
                  or os.getenv("VLLM_DYNAMO_ARTIFACT_PROBE_DIR"))
    if not output_dir:
        return

    probe_dir = Path(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    stem = f"dynamo_fx_rank_{rank}_{os.getpid()}"
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

        from .frontend_metadata import dump_metadata
        evidence_bytes = dump_metadata(graph, example_inputs)
        evidence_path = probe_dir / f"{stem}.fake_evidence.graphpickle"
        evidence_path.write_bytes(evidence_bytes)

        metadata.update({
            "fxstruct_bytes": len(graph_bytes),
            "fxstruct_path": str(graph_path),
            "fake_evidence_bytes": len(evidence_bytes),
            "fake_evidence_path": str(evidence_path),
            "structural_seconds": time.perf_counter() - structural_started,
        })
    except Exception as exc:
        metadata.update({
            "structural_error": f"{type(exc).__name__}: {exc}",
            "structural_seconds": time.perf_counter() - structural_started,
        })
    (probe_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


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
        logger.debug("Call VllmBackend.__init__()...")
        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. launguage_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag
        logger.debug("VllmBackend's prefix:%s", self.prefix)
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
            "rank": self.vllm_config.parallel_config.rank,
            "dp_rank": self.vllm_config.parallel_config.data_parallel_rank,
            "source": assets.source_hash("compilation", "model_executor"),
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
                self.dynamo_frontend_runtime_identity(),
                output_dir=getattr(self, "_frontend_cache_dir", None))
        from .monitor import torch_compile_start_time
        dynamo_frontend_time = time.time() - torch_compile_start_time
        logger.debug("Dynamo frontend before backend entry: %.6f s",
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
                str(resource_dir("compile")),
                hash_key,
            )
            self.compilation_config.cache_dir = cache_dir

        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir

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
        self.compiler_manager.initialize_cache(local_cache_dir, disable_cache, self.prefix,
                                               self.dynamo_frontend_runtime_identity())
        timer.end("analyzing cache_dir and files")
        logger.debug("Compilation cache analysis time: %.6f s",
                    time.perf_counter() - cache_analysis_started)

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        dynamo_time = time.time() - torch_compile_start_time
        logger.debug("Tracing operational time including cache analysis: %.6f s",
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
        self.compiler_manager.configure_reuse(vllm_config)
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
        logger.debug("FX graph split time: %.6f s",
                    time.perf_counter() - split_started)
        # propagate the split graph to the piecewise backend, compile submodules with symbolic shapes
        logger.debug("start compile submodules with symbolic shapes")
        start_time = time.perf_counter()
        if enabled_by_master("VLLM_ENABLE_PARALLEL_COMPILE_METADATA"):
            submodule_inputs_metadata_cache_dir = str(
                parallel_compile_metadata_dir(cache_dir, rank, dp_rank,
                                              self.prefix, self.vllm_config))
        else:
            submodule_inputs_metadata_cache_dir = os.path.join(
                local_cache_dir,
                "submodule_inputs_metadata"
            )
        # Translate engine configuration here; the collector only consumes FX
        # evidence and an explicit metadata store, never the engine backend.
        def collector_factory():
            identity = parallel_compile_metadata_identity(
                local_cache_dir, rank, dp_rank, self.prefix, self.vllm_config)
            serializer = SubmoduleInputSerializer(
                submodule_inputs_metadata_cache_dir,
                self.vllm_config.model_config.model.split('/')[-1], identity)
            return PiecewiseCompileInterpreter(
                self.split_gm, submod_names_to_compile, serializer,
                restore_input_metadata=enabled_by_master("VLLM_ENABLE_PARALLEL_COMPILE_METADATA"),
                structural_reuse=structural_reuse_enabled())
        compiled_graphs, collector = self.parallel_manager.compile_subgraphs(
            collector_factory, example_inputs,
            self.compiler_manager,
            self.compilation_config,
            self.vllm_config,
        )
        self.meta_data = collector.metadata
        self.reuse_index_map = collector.reuse_index_map

        assembly_started = time.perf_counter()
        piecewise_backend = resolve_obj_by_qualname(current_platform.get_piecewise_backend_cls())
        # Shadow FX submodules with callable backends without changing the IR.
        for i, name in enumerate(submod_names_to_compile):
            meta = self.meta_data[name]
            compiled_graph = compiled_graphs[name]
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
            # 记录被编译的被识别并已处理的分段图数量
            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        logger.debug("Piecewise backend assembly time: %.6f s",
                    time.perf_counter() - assembly_started)
        logger.debug("end compile submodules with symbolic shapes, takes:%.3f", time.perf_counter() - start_time)
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

        logger.debug("VllmBackend total time: %.6f s",
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
        logger.debug("Use cudagraph_copy_inputs and Return a function: copy_and_call()...")

        # this is the callable we return to Dynamo to run
        def copy_and_call(*args):
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
