# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from abc import abstractmethod
from contextlib import contextmanager
from types import CodeType
from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


def _dump_dynamo_frontend_artifact_probe(frame, old_code: CodeType,
                                         new_code: CodeType) -> None:
    """Dump serialization facts for Dynamo-generated frontend state."""
    output_dir = (os.getenv("VLLM_DYNAMO_FRONTEND_CACHE_DIR")
                  or os.getenv("VLLM_DYNAMO_ARTIFACT_PROBE_DIR"))
    if not output_dir:
        return

    import json
    import marshal
    from pathlib import Path

    probe_dir = Path(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    stem = f"dynamo_frontend_rank_{rank}_{os.getpid()}"

    generated_names = sorted(
        name for name in new_code.co_names
        if name.startswith(("__compiled_fn", "__import_")))
    global_descriptions = []
    for name in generated_names:
        value = frame.f_globals.get(name)
        description = {
            "name": name,
            "present": name in frame.f_globals,
        }
        if value is not None:
            value_type = type(value)
            description.update({
                "type": f"{value_type.__module__}.{value_type.__qualname__}",
                "module": getattr(value, "__module__", None),
                "qualname": getattr(value, "__qualname__", None),
                "object_name": getattr(value, "__name__", None),
                "callable": callable(value),
            })
        global_descriptions.append(description)

    with (probe_dir / f"{stem}_transformed_code.marshal").open("wb") as f:
        marshal.dump(new_code, f)
    metadata = {
        "old_code": {
            "name": old_code.co_name,
            "qualname": getattr(old_code, "co_qualname", old_code.co_name),
            "filename": old_code.co_filename,
            "argcount": old_code.co_argcount,
            "names": list(old_code.co_names),
            "structural_hash": __import__(
                "vllm.compilation.transformed_code_cache",
                fromlist=["_code_hash"])._code_hash(old_code),
        },
        "new_code": {
            "name": new_code.co_name,
            "qualname": getattr(new_code, "co_qualname", new_code.co_name),
            "filename": new_code.co_filename,
            "argcount": new_code.co_argcount,
            "names": list(new_code.co_names),
            "freevars": list(new_code.co_freevars),
            "cellvars": list(new_code.co_cellvars),
        },
        "compiled_globals": global_descriptions,
    }
    with (probe_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    def __init__(self,
                 compiled_callable: Optional[Callable] = None,
                 compilation_level: int = 0):

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        self._dynamo_backend = None
        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method

            backend = vllm_config.compilation_config.init_backend(vllm_config)
            self._dynamo_backend = backend
            options = None
            if isinstance(backend, str) and backend == "inductor":
                options = get_current_vllm_config(
                ).compilation_config.inductor_compile_config

            compiled_callable = torch.compile(
                self.forward,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend,
                options=options)

        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        self.use_custom_dispatcher: bool = \
            compilation_level >= CompilationLevel.DYNAMO_ONCE

    def try_restore_dynamo_frontend(self) -> bool:
        """Restore a cached frontend or safely fall back to Dynamo."""
        try:
            return self._try_restore_dynamo_frontend_impl()
        except Exception:
            logger.exception(
                "Dynamo frontend replay failed; falling back to tracing")
            return False

    def _try_restore_dynamo_frontend_impl(self) -> bool:
        """Restore an explicitly selected, exact-config Dynamo frontend."""
        replay_dir_text = (os.getenv("VLLM_DYNAMO_FRONTEND_CACHE_DIR")
                           or os.getenv("VLLM_DYNAMO_FRONTEND_REPLAY_DIR"))
        backend = self._dynamo_backend
        if not replay_dir_text:
            from vllm.compilation.startup_resources import (env_flag,
                                                             resource_dir)
            if env_flag("VLLM_ENABLE_DYNAMO_FRONTEND_CACHE", False):
                import hashlib
                import json
                from vllm.compilation.transformed_code_cache import _code_hash
                identity = {
                    "runtime": backend.dynamo_frontend_runtime_identity()
                    if backend is not None and not isinstance(backend, str)
                    else None,
                    "code": _code_hash(self.original_code_object),
                    "wrapper": (f"{self.__class__.__module__}."
                                f"{self.__class__.__qualname__}"),
                }
                key = hashlib.sha256(json.dumps(
                    identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
                rank = self.vllm_config.parallel_config.rank
                dp_rank = self.vllm_config.parallel_config.data_parallel_rank
                prefix = getattr(backend, "prefix", "default") or "default"
                replay_dir_text = str(
                    resource_dir("dynamo_frontend") / key /
                    f"rank_{rank}_{dp_rank}" / prefix)
                os.environ["VLLM_DYNAMO_FRONTEND_CACHE_DIR"] = replay_dir_text
        if not replay_dir_text or backend is None or isinstance(backend, str):
            return False

        import importlib
        import json
        import marshal
        import pickle
        import time
        from pathlib import Path

        replay_dir = Path(replay_dir_text)
        bundles = []
        for code_metadata_path in replay_dir.glob(
                "dynamo_frontend_rank_*.json"):
            suffix = code_metadata_path.stem.removeprefix(
                "dynamo_frontend_")
            fx_stem = f"dynamo_fx_{suffix}"
            bundle = {
                "code_metadata": code_metadata_path,
                "code": replay_dir / (
                    f"dynamo_frontend_{suffix}_transformed_code.marshal"),
                "fx_metadata": replay_dir / f"{fx_stem}.json",
                "graph": replay_dir / f"{fx_stem}.fxstruct.pkl",
                "direct_symbols": replay_dir / (
                    f"{fx_stem}.direct_symbols.graphpickle"),
                "example_values": replay_dir / (
                    f"{fx_stem}.example_values.graphpickle"),
            }
            required = (
                bundle["code_metadata"], bundle["code"],
                bundle["fx_metadata"], bundle["graph"])
            if (all(path.is_file() for path in required)
                    and (bundle["direct_symbols"].is_file()
                         or bundle["example_values"].is_file())):
                bundles.append(bundle)
        if not bundles:
            logger.info("Dynamo frontend replay miss: incomplete artifacts in %s",
                        replay_dir)
            return False
        bundle = max(
            bundles,
            key=lambda item: item["code_metadata"].stat().st_mtime_ns)

        started = time.perf_counter()
        fx_metadata = json.loads(bundle["fx_metadata"].read_text(
            encoding="utf-8"))
        code_metadata = json.loads(bundle["code_metadata"].read_text(
            encoding="utf-8"))
        old_metadata = code_metadata.get("old_code", {})
        expected = {
            "name": self.original_code_object.co_name,
            "filename": self.original_code_object.co_filename,
            "argcount": self.original_code_object.co_argcount,
            "names": list(self.original_code_object.co_names),
            "structural_hash": __import__(
                "vllm.compilation.transformed_code_cache",
                fromlist=["_code_hash"])._code_hash(
                    self.original_code_object),
        }
        if any(old_metadata.get(key) != value
               for key, value in expected.items()):
            logger.warning("Dynamo frontend replay rejected: original code "
                           "metadata changed")
            return False

        artifact_identity = fx_metadata.get("runtime_identity")
        current_identity = backend.dynamo_frontend_runtime_identity()
        if artifact_identity != current_identity:
            logger.warning("Dynamo frontend replay rejected: runtime identity "
                           "changed (cached=%s current=%s)", artifact_identity,
                           current_identity)
            return False

        transformed_code = marshal.loads(bundle["code"].read_bytes())
        if (transformed_code.co_argcount !=
                self.original_code_object.co_argcount):
            logger.warning("Dynamo frontend replay rejected: transformed "
                           "bytecode signature changed")
            return False

        traced_files = fx_metadata.get("traced_files", [])
        missing_files = [path for path in traced_files
                         if path != "<string>" and not Path(path).is_file()]
        if missing_files:
            logger.warning("Dynamo frontend replay rejected: traced files "
                           "missing: %s", missing_files)
            return False
        import hashlib
        for path, expected_sha256 in fx_metadata.get(
                "traced_file_sha256", {}).items():
            with open(path, "rb") as source_file:
                actual_sha256 = hashlib.sha256(source_file.read()).hexdigest()
            if actual_sha256 != expected_sha256:
                logger.warning("Dynamo frontend replay rejected: traced source "
                               "changed: %s", path)
                return False
        backend.compilation_config.traced_files.update(traced_files)
        validation_s = time.perf_counter() - started

        graph_restore_started = time.perf_counter()
        from vllm.compilation.backends import (
            _deserialize_graph_module_for_subprocess)
        graph_payload = pickle.loads(bundle["graph"].read_bytes())
        graph = _deserialize_graph_module_for_subprocess(graph_payload)
        graph_restore_s = time.perf_counter() - graph_restore_started
        symbol_restore_started = time.perf_counter()
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        nodes_by_name = {node.name: node for node in graph.graph.nodes}
        if bundle["direct_symbols"].is_file():
            direct_symbols = GraphPickler.loads(
                bundle["direct_symbols"].read_bytes(), fake_mode)
            symbols_by_name = {}
            direct_node_names = set()
            for node_name, symbolic_value in direct_symbols:
                nodes_by_name[node_name].meta["example_value"] = symbolic_value
                direct_node_names.add(node_name)
                symbols_by_name[str(symbolic_value.node.expr)] = (
                    symbolic_value.node.expr)
            for node_name, symbol_names in fx_metadata.get(
                    "symbol_dependencies", {}).items():
                if node_name in direct_node_names:
                    continue
                nodes_by_name[node_name].meta["example_value"] = tuple(
                    symbols_by_name[name] for name in symbol_names)
        else:
            example_values = GraphPickler.loads(
                bundle["example_values"].read_bytes(), fake_mode)
            for node_name, example_value in example_values:
                nodes_by_name[node_name].meta["example_value"] = example_value
        symbol_restore_s = time.perf_counter() - symbol_restore_started
        backend_started = time.perf_counter()
        backend._dynamo_frontend_replaying = True
        try:
            compiled_graph = backend(graph, ())
        finally:
            backend._dynamo_frontend_replaying = False
        backend_s = time.perf_counter() - backend_started
        compiled_fn = torch._dynamo.disable(compiled_graph)

        install_started = time.perf_counter()
        forward_globals = self.__class__.forward.__globals__
        compiled_names = []
        for description in code_metadata.get("compiled_globals", []):
            name = description.get("name", "")
            if name.startswith("__import_"):
                module_name = description.get("object_name")
                if not module_name:
                    raise RuntimeError(
                        f"Missing module identity for generated global {name}")
                forward_globals[name] = importlib.import_module(module_name)
            elif name.startswith("__compiled_fn"):
                compiled_names.append(name)
        if len(compiled_names) != 1:
            raise RuntimeError("Expected exactly one generated compiled_fn, "
                               f"found {compiled_names}")
        forward_globals[compiled_names[0]] = compiled_fn
        self.compiled_codes.append(transformed_code)
        install_s = time.perf_counter() - install_started
        logger.info(
            "Dynamo frontend replay breakdown: validation=%.6f s "
            "graph_restore=%.6f s symbols=%.6f s backend=%.6f s "
            "install=%.6f s",
            validation_s, graph_restore_s, symbol_restore_s, backend_s,
            install_s)
        logger.info("Dynamo frontend replay hit: restored %d-node FX graph "
                    "and transformed bytecode in %.6f s from %s",
                    len(list(graph.graph.nodes)),
                    time.perf_counter() - started, replay_dir)
        return True

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object:
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return

        self.compiled_codes.append(new_code)
        try:
            _dump_dynamo_frontend_artifact_probe(frame, old_code, new_code)
        except Exception:
            logger.exception("Failed to dump Dynamo frontend artifact probe")
        local_cache_dir = self.vllm_config.compilation_config.local_cache_dir
        if isinstance(local_cache_dir, str):
            decompiled_file = os.path.join(local_cache_dir,
                                           "transformed_code.py")
            try:
                from vllm.compilation import transformed_code_cache
                if os.path.exists(decompiled_file):
                    transformed_code_cache.save(
                        old_code,
                        new_code,
                        decompiled_file,
                        self.__class__,
                        local_cache_dir=local_cache_dir)
                else:
                    restored, cache_key, _, _ = transformed_code_cache.lookup(
                        old_code, new_code, decompiled_file, self.__class__)
                    if restored:
                        logger.info(
                            "Reused Dynamo transformed code from semantic "
                            "cache key=%s into %s", cache_key[:12],
                            decompiled_file)
                    else:
                        # Usually the decompilation will succeed for most
                        # models, as we guarantee a full-graph compilation in
                        # Dynamo. There is no 100% guarantee, since
                        # decompilation is not a reversible process.
                        import depyf
                        src = depyf.decompile(new_code)
                        with open(decompiled_file, "w") as f:
                            f.write(src)
                        transformed_code_cache.save(
                            old_code,
                            new_code,
                            decompiled_file,
                            self.__class__,
                            local_cache_dir=local_cache_dir)
                        logger.debug("Dynamo transformed code saved to %s",
                                     decompiled_file)
            except Exception:
                pass

        if self.vllm_config.compilation_config.use_cudagraph and \
            "update" in new_code.co_names:
            import depyf
            src = depyf.decompile(new_code)
            msg = "Assigning / modifying buffers of nn.Module during forward pass is not allowed when using cudagraph inside the compiler because it will cause silent errors. Please use eager mode or fix the code. The following code contains clues about which buffer is being modified (please search for the usage of the function `update`):\n" + src  # noqa
            raise RuntimeError(msg)

    @contextmanager
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """ # noqa
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object
