# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FX graph transport shared by frontend replay and compilation workers."""

import importlib.util
import operator
import os
import pickle
import sys
import time
from typing import Any

import torch
import torch.fx as fx

from vllm.logger import init_logger
from vllm.utils import resolve_obj_by_qualname

logger = init_logger(__name__)

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


def _set_module_attr_by_qualname(
    root: torch.nn.Module, target: str, value: Any
) -> None:
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
            _FX_DICT: [
                [_encode_fx_value(k), _encode_fx_value(v)] for k, v in value.items()
            ]
        }
    return value


def _decode_fx_value(value: Any, env: dict[str, fx.Node]) -> Any:
    if isinstance(value, dict):
        if _FX_NODE_REF in value:
            return env[value[_FX_NODE_REF]]
        if _FX_SIZE in value:
            return torch.Size([_decode_fx_value(v, env) for v in value[_FX_SIZE]])
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
    graph_module: fx.GraphModule,
) -> dict[str, Any]:
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
        payload["nodes"].append(
            {
                "op": node.op,
                "name": node.name,
                "target": _serialize_fx_target(node.target),
                "args": _encode_fx_value(node.args),
                "kwargs": _encode_fx_value(node.kwargs),
            }
        )

    return payload


def _deserialize_graph_module_for_subprocess(payload: dict[str, Any]) -> fx.GraphModule:
    root = torch.nn.Module()
    for target, value in sorted(
        payload.get("attrs", {}).items(), key=lambda item: (item[0].count("."), item[0])
    ):
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


def _load_graph_module_for_subprocess(graph_path: str) -> fx.GraphModule:
    payload_path = os.path.join(graph_path, "graph_module.pkl")
    if os.path.exists(payload_path):
        with open(payload_path, "rb") as f:
            return _deserialize_graph_module_for_subprocess(pickle.load(f))

    init_file = os.path.join(graph_path, "__init__.py")
    if not os.path.exists(init_file):
        raise FileNotFoundError(f"FX graph payload not found at {payload_path}")

    package_name = f"_vllm_fx_subprocess_{abs(hash(graph_path))}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[graph_path],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to create import spec for FX package at {graph_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)

    graph_cls = getattr(module, _FX_SUBPROCESS_MODULE_CLASS)
    return graph_cls()
