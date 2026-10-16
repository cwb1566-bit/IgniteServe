# SPDX-License-Identifier: Apache-2.0
"""Value-free FX evidence for exact-config frontend replay."""
import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx._graph_pickler import GraphPickler
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._pytree import tree_leaves

from .input_contract import input_contract

SCHEMA = "frontend-fake-evidence-v1"


def dump_metadata(graph, example_inputs):
    raw_values = {node.name: node.meta["example_value"]
                  for node in graph.graph.nodes if "example_value" in node.meta}
    # Dynamo backend arguments may be real tensors while node evidence belongs
    # to Dynamo's original fake mode, not the ambient compilation context.
    mode = next((v.fake_mode for v in tree_leaves(raw_values)
                 if isinstance(v, FakeTensor)), None)
    mode = mode or detect_fake_mode(example_inputs) or FakeTensorMode(shape_env=ShapeEnv())

    def fake(value):
        if isinstance(value, torch.Tensor):
            if not isinstance(value, FakeTensor):
                value = mode.from_tensor(value, static_shapes=True)
            if value.fake_mode is not mode:
                raise ValueError("foreign fake mode")
            if value.constant is not None:
                raise ValueError(f"data-dependent tensor constant: shape={value.shape}")
            return value
        if isinstance(value, torch.SymInt):
            if value.node.shape_env is not mode.shape_env:
                raise ValueError("foreign symbolic environment")
            return value
        if isinstance(value, (tuple, list)):
            return type(value)(fake(v) for v in value)
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {k: fake(v) for k, v in value.items()}
        if value is None or type(value) in (int, float, bool, str):
            return value
        raise ValueError(f"unsupported frontend evidence: {type(value)}")

    values = {}
    for node in graph.graph.nodes:
        if "example_value" in node.meta:
            try:
                values[node.name] = fake(node.meta["example_value"])
            except ValueError as exc:
                raise ValueError(f"{node.name}: {exc}") from exc
    inputs = fake(tuple(example_inputs))
    placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    if input_contract(tuple(values[n.name] for n in placeholders)) != input_contract(inputs):
        raise ValueError("frontend placeholder/input evidence differs")
    evidence = (values, inputs)
    contract = input_contract(evidence)
    # One pickle keeps storage aliases and all symbols in one common context.
    return GraphPickler.dumps({"schema": SCHEMA, "evidence": evidence,
                               "contract": contract})


def restore_metadata(graph, content):
    mode = FakeTensorMode(shape_env=ShapeEnv())
    payload = GraphPickler.loads(content, mode)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported frontend evidence schema")
    values, inputs = payload["evidence"]
    if input_contract((values, inputs)) != payload["contract"]:
        raise ValueError("restored frontend evidence differs")
    nodes = {n.name: n for n in graph.graph.nodes}
    if not set(values).issubset(nodes):
        raise ValueError("frontend evidence names do not match graph")
    placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    if input_contract(tuple(values[n.name] for n in placeholders)) != input_contract(inputs):
        raise ValueError("restored frontend placeholders differ")
    for name, value in values.items():
        nodes[name].meta["example_value"] = value
    return mode, inputs
