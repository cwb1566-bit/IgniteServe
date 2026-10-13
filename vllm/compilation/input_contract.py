# SPDX-License-Identifier: Apache-2.0
"""Conservative, value-free tensor contracts for compilation inputs."""
import torch
from .startup_assets import digest


def input_contract(value):
    storages = {}
    symbols = {}
    guards = {}

    def dim(v):
        if isinstance(v, torch.SymInt):
            expr = str(v.node.expr)
            # Keep expressions and their common ShapeEnv, not just a hint.
            env = v.node.shape_env
            guards.update({str(k): str(r) for k, r in env.var_to_range.items()})
            guards["expressions"] = [str(g.expr) for g in env.guards]
            symbols[expr] = str(v.node.expr)
            return {"symbol": expr}
        return int(v)

    def visit(v):
        if isinstance(v, torch.Tensor):
            if v.layout != torch.strided:
                raise ValueError("unsupported tensor layout")
            storage = v.untyped_storage()._cdata
            group = storages.setdefault(storage, len(storages))
            return {"tensor": {"shape": [dim(d) for d in v.shape],
                               "stride": [dim(d) for d in v.stride()],
                               "dtype": str(v.dtype), "device": str(v.device),
                               "layout": str(v.layout),
                               "offset": dim(v.storage_offset()), "alias": group,
                               "requires_grad": v.requires_grad}}
        if isinstance(v, torch.SymInt):
            return dim(v)
        if isinstance(v, (tuple, list)):
            return {type(v).__name__: [visit(x) for x in v]}
        if isinstance(v, dict) and all(isinstance(k, str) for k in v):
            return {"dict": [[k, visit(x)] for k, x in v.items()]}
        if v is None or type(v) in (int, float, bool, str):
            return v
        raise ValueError(f"unsupported input type: {type(v)}")

    return {"value": visit(value), "guards": guards}


def graph_key(graph):
    """Preserve connections, constants and operand order; never erase digits."""
    nodes = list(graph.graph.nodes)
    indices = {node: i for i, node in enumerate(nodes)}

    def arg(value):
        if isinstance(value, torch.fx.Node):
            return {"node": indices[value]}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [arg(x) for x in value]}
        if isinstance(value, dict):
            return {str(k): arg(v) for k, v in value.items()}
        if isinstance(value, slice):
            return {"slice": [arg(value.start), arg(value.stop), arg(value.step)]}
        if value is Ellipsis:
            return {"ellipsis": True}
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, (torch.dtype, torch.device, torch.layout)):
            return str(value)
        raise ValueError(f"unsupported graph constant: {type(value)}")

    description = []
    for node in nodes:
        if node.op in ("call_module", "get_attr"):
            raise ValueError("opaque module or captured tensor constant")
        target = "input" if node.op == "placeholder" else str(node.target)
        if node.op == "call_function" and callable(node.target):
            function = node.target
            target = f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', str(function))}"
            if hasattr(function, "__code__"):
                if function.__code__.co_freevars:
                    raise ValueError("opaque closure in graph")
                from .transformed_code_cache import _code_hash
                target += ":" + _code_hash(function.__code__)
        description.append([node.op, target, arg(node.args), arg(node.kwargs)])
    return digest(description)
