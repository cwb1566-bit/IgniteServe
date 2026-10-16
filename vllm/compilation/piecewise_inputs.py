# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FX input evidence collection and conservative metadata restoration."""

import time
from typing import Any, Dict, Optional

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher

from vllm.logger import init_logger

from .input_contract import graph_key, input_contract
from .input_metadata import SubmoduleInputMetadata, SubmoduleInputSerializer

logger = init_logger(__name__)


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        serializer: SubmoduleInputSerializer,
        *,
        restore_input_metadata: bool,
        structural_reuse: bool,
        fake_mode=None,
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = fake_mode if fake_mode is not None else detect_fake_mode()
        if self.fake_mode is None:
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            self.fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        self.compile_submod_names = compile_submod_names
        self.metadata: dict[str, Dict[str, Any]] = {}
        self.reuse_index_map: dict[int, int] = {}
        self.structural_reuse = structural_reuse
        self.extra_traceback = False

        self.serializer = serializer
        self.restore_input_metadata = restore_input_metadata
        self.cached_metadata = (
            self._try_load_cached_metadata() if self.restore_input_metadata else None
        )
        if not self.restore_input_metadata and self.serializer.metadata_file.exists():
            logger.info(
                "Parallel compile input metadata exists at %s, but restore is "
                "disabled outside the S4 parallel-compile-metadata path.",
                self.serializer.metadata_file,
            )
        self.fake_run_results_metadata: dict[int, Any] = {}
        self.fake_run_map_list = self._load_or_build_reuse_map()

    def _normalize_reuse_map(self, raw_map: dict[Any, Any]) -> dict[str, int]:
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

    def _submodule_signature(self, submod: torch.fx.GraphModule) -> tuple[Any, ...]:
        try:
            return (graph_key(submod),)
        except (ValueError, TypeError, AttributeError):
            return (id(submod),)  # Opaque graphs never alias another graph.

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
        if not self.structural_reuse:
            return self._normalize_reuse_map(
                {str(index): index for index in range(len(self.compile_submod_names))}
            )
        return self._normalize_reuse_map(self._build_structural_reuse_map())

    def _try_load_cached_metadata(self) -> Optional[Dict[str, SubmoduleInputMetadata]]:
        started = time.perf_counter()
        loaded_count = 0
        try:
            metadata_list = self.serializer.load_all_metadata()
            if metadata_list:
                loaded_count = len(metadata_list)
                by_graph = {
                    meta.graph_signature: meta
                    for meta in metadata_list
                    if meta.graph_signature and meta.contract
                }
                matched = {}
                for name in self.compile_submod_names:
                    try:
                        meta = by_graph.get(graph_key(self.fetch_attr(name)))
                        if meta is not None:
                            matched[name] = meta
                    except ValueError:
                        pass
                return matched or None
        except Exception as exc:
            logger.warning("Failed to load cached metadata: %s", exc)
        finally:
            logger.debug(
                "Offline metadata timing load_s=%.6f loaded=%s path=%s",
                time.perf_counter() - started,
                loaded_count,
                self.serializer.metadata_file,
            )
        return None

    def run(self, *args):
        self.reuse_index_map = {
            int(index): int(mapped_index)
            for index, mapped_index in self.fake_run_map_list.items()
        }
        if self.cached_metadata is not None and not args:
            logger.info(
                "Restoring submodule inputs from cache without "
                "replaying full-graph example inputs..."
            )
            if self._restore_from_cache():
                return None
            raise ValueError("frontend replay lacks validated current subgraph inputs")

        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]

        if self.cached_metadata is not None:
            logger.info("Restoring submodule inputs from cache...")
            restore_started = time.perf_counter()
            self.serializer.seed_symbol_table_from_args(tuple(fake_args))
            restored = self._restore_from_cache()
            logger.debug(
                "Offline metadata timing restore_s=%.6f restored=%s",
                time.perf_counter() - restore_started,
                len(self.metadata),
            )
            if restored:
                return None

        logger.info("Running fake mode inference...")
        fake_started = time.perf_counter()
        with self.fake_mode, enable_python_dispatcher():
            result = super().run(*fake_args)
        logger.debug(
            "Offline metadata timing fake_inference_s=%.6f captured=%s",
            time.perf_counter() - fake_started,
            len(self.metadata),
        )

        save_started = time.perf_counter()
        self._save_metadata()
        self._validate_reuse_inputs()
        logger.debug(
            "Offline metadata timing save_s=%.6f path=%s",
            time.perf_counter() - save_started,
            self.serializer.metadata_file,
        )
        return result

    def _record_submodule_metadata(
        self, target: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> int:
        submod = self.fetch_attr(target)
        sym_shape_indices = [
            i for i, arg in enumerate(args) if isinstance(arg, torch.SymInt)
        ]
        index = self.compile_submod_names.index(target)
        self.metadata[target] = {
            "submod": submod,
            "args": args,
            "kwargs": kwargs,
            "sym_shape_indices": sym_shape_indices,
            "index": index,
        }
        return index

    def _restore_from_cache(self):
        restored = {}
        for submod_name in self.compile_submod_names:
            if submod_name not in self.cached_metadata:
                logger.warning("No cached metadata for %s", submod_name)
                continue

            meta = self.cached_metadata[submod_name]
            try:
                call = next(
                    n
                    for n in self.module.graph.nodes
                    if n.op == "call_module" and n.target == submod_name
                )

                def example(node):
                    if "val" in node.meta:
                        return node.meta["val"]
                    return node.meta["example_value"]

                try:
                    current = fx.node.map_arg((call.args, call.kwargs), example)
                except KeyError:
                    # split_module preserves Dynamo evidence on the submodule
                    # placeholders, but not on newly created parent getitems.
                    if call.kwargs:
                        raise ValueError("unresolved keyword input evidence")
                    placeholders = [
                        n
                        for n in self.fetch_attr(submod_name).graph.nodes
                        if n.op == "placeholder"
                    ]
                    current = (tuple(example(n) for n in placeholders), {})
                if input_contract(current) != meta.contract:
                    raise ValueError("current inputs or guards differ")
                args, kwargs = self.serializer.reconstruct_submodule_inputs(
                    meta, self.fake_mode
                )
                if input_contract((args, kwargs)) != meta.contract:
                    raise ValueError("restored inputs, aliasing or guards differ")
            except (
                ValueError,
                KeyError,
                TypeError,
                StopIteration,
                AttributeError,
            ) as exc:
                logger.info("Input metadata miss for %s: %s", submod_name, exc)
                continue
            submod = self.fetch_attr(submod_name)
            restored[submod_name] = {
                "submod": submod,
                "args": args,
                "kwargs": kwargs,
                "sym_shape_indices": meta.sym_shape_indices,
                "index": self.compile_submod_names.index(submod_name),
            }
        self.metadata.update(restored)
        self._validate_reuse_inputs()
        return len(restored) == len(self.compile_submod_names)

    def _validate_reuse_inputs(self):
        for index, name in enumerate(self.compile_submod_names):
            mapped = self.fake_run_map_list[str(index)]
            if mapped == index:
                continue
            a = self.metadata.get(name)
            b = self.metadata.get(self.compile_submod_names[mapped])
            try:
                if (
                    a is None
                    or b is None
                    or input_contract((a["args"], a["kwargs"]))
                    != input_contract((b["args"], b["kwargs"]))
                ):
                    self.fake_run_map_list[str(index)] = index
            except (ValueError, TypeError, AttributeError):
                self.fake_run_map_list[str(index)] = index
        self.reuse_index_map = {
            int(i): int(j) for i, j in self.fake_run_map_list.items()
        }

    def _save_metadata(self):
        metadata_list = []
        for submod_name in self.compile_submod_names:
            if submod_name not in self.metadata:
                logger.warning("No metadata for %s", submod_name)
                continue
            data = self.metadata[submod_name]
            try:
                meta = self.serializer.extract_submodule_metadata(
                    submod_name=submod_name,
                    args=data["args"],
                    kwargs=data["kwargs"],
                    sym_shape_indices=data["sym_shape_indices"],
                    index=data["index"],
                )
                meta.graph_signature = graph_key(data["submod"])
                meta.contract = input_contract((data["args"], data["kwargs"]))
                metadata_list.append(meta)
            except (ValueError, TypeError, AttributeError) as exc:
                logger.info(
                    "Not persisting unsupported inputs for %s: %s", submod_name, exc
                )

        self.serializer.save_all_metadata(metadata_list)

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        if target not in self.compile_submod_names:
            return super().call_module(target, args, kwargs)

        index = self._record_submodule_metadata(target, args, kwargs)
        mapped_index = int(self.fake_run_map_list.get(str(index), index))
        if mapped_index != index and mapped_index in self.fake_run_results_metadata:
            other = self.metadata[self.compile_submod_names[mapped_index]]
            try:
                equal = input_contract((args, kwargs)) == input_contract(
                    (other["args"], other["kwargs"])
                )
            except (ValueError, TypeError, AttributeError):
                equal = False
            if equal:
                reused_output = self.fake_run_results_metadata[mapped_index]
                self.fake_run_results_metadata[index] = reused_output
                return reused_output
            self.fake_run_map_list[str(index)] = index

        output = super().call_module(target, args, kwargs)
        self.fake_run_results_metadata[index] = output
        if mapped_index not in self.fake_run_results_metadata:
            self.fake_run_map_list[str(index)] = index
        return output
