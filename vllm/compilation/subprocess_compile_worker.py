# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import pickle
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm.compilation.backends import (SubmoduleInputSerializer,
                                       _load_graph_module_for_subprocess)
from vllm.compilation.compiler_interface import (
    EagerAdaptor,
    InductorAdaptor,
    InductorStandaloneAdaptor,
    ThreadSafeInductorAdaptor,
)


COMPILER_CLASS_MAP = {
    "ThreadSafeInductorAdaptor": ThreadSafeInductorAdaptor,
    "InductorAdaptor": InductorAdaptor,
    "InductorStandaloneAdaptor": InductorStandaloneAdaptor,
    "EagerAdaptor": EagerAdaptor,
}


def _load_graph(graph_path: str) -> Any:
    return _load_graph_module_for_subprocess(graph_path)


def _load_input_metadata(metadata_cache_dir: str,
                         model_name: str,
                         graph_index: int):
    serializer = SubmoduleInputSerializer(metadata_cache_dir, model_name)
    for metadata in serializer.load_all_metadata():
        if metadata.index == graph_index:
            return serializer, metadata
    raise RuntimeError(
        f"Failed to find input metadata for graph_{graph_index} "
        f"in {metadata_cache_dir}")


def _load_compile_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        return pickle.load(f)


def _build_example_inputs(metadata_cache_dir: str,
                          model_name: str,
                          graph_index: int):
    serializer, metadata = _load_input_metadata(metadata_cache_dir,
                                                model_name,
                                                graph_index)
    fake_mode = FakeTensorMode(shape_env=ShapeEnv())
    args, kwargs = serializer.reconstruct_submodule_inputs(metadata, fake_mode)
    if kwargs:
        raise RuntimeError(
            "Subprocess compile worker does not support kwargs inputs yet.")
    return list(args)


def _maybe_key(compiler: Any,
               runtime_shape: Optional[int],
               graph_index: int) -> Optional[str]:
    if isinstance(compiler, (InductorAdaptor, ThreadSafeInductorAdaptor)):
        return None
    return f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--metadata-cache-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--graph-index", type=int, required=True)
    parser.add_argument("--compiler-class", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--runtime-shape", type=int)
    parser.add_argument("--disable-cache", action="store_true")
    args = parser.parse_args()

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        compiler_cls = COMPILER_CLASS_MAP[args.compiler_class]
        compiler = compiler_cls()
        compiler.initialize_cache(args.cache_dir, args.disable_cache,
                                  args.prefix)

        graph = _load_graph(args.graph_path)
        if hasattr(graph, "eval"):
            graph.eval()
        example_inputs = _build_example_inputs(args.metadata_cache_dir,
                                               args.model_name,
                                               args.graph_index)
        compile_config = _load_compile_config(args.config_path)

        with torch.inference_mode():
            _, handle = compiler.compile(
                graph,
                example_inputs,
                compile_config,
                args.runtime_shape,
                _maybe_key(compiler, args.runtime_shape, args.graph_index),
                graph_index=args.graph_index,
            )
        if handle is None:
            raise RuntimeError(
                "Compiler did not return a reusable handle in subprocess "
                "worker mode.")

        payload = {
            "status": "ok",
            "graph_index": args.graph_index,
            "handle": handle,
        }
    except Exception as exc:
        payload = {
            "status": "error",
            "graph_index": args.graph_index,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    with open(result_path, "wb") as f:
        pickle.dump(payload, f)

    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
