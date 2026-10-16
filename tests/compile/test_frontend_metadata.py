# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch._guards import tracing, TracingContext
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx._graph_pickler import GraphPickler
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm.compilation.frontend_metadata import dump_metadata, restore_metadata
from vllm.compilation.input_contract import input_contract


def graph(values):
    result = torch.fx.Graph()
    for name, value in values.items():
        result.placeholder(name).meta["example_value"] = value
    result.output(None)
    return torch.fx.GraphModule({}, result)


def test_real_parameter_becomes_metadata_not_weights():
    tensor = torch.nn.Parameter(torch.empty(1024, 1024))
    source = graph({"weight": tensor})
    content = dump_metadata(source, [tensor])
    assert len(content) < 100_000 and len(content) < tensor.numel()
    _, inputs = restore_metadata(graph({"weight": None}), content)
    assert isinstance(inputs[0], FakeTensor)
    assert inputs[0].shape == tensor.shape and inputs[0].requires_grad


def test_symbol_stride_alias_and_offset_roundtrip():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    with mode, tracing(TracingContext(mode)):
        symbol = mode.shape_env.create_unbacked_symint()
        base = torch.empty(24)
        view = base.as_strided((2, 3), (1, 4), 2)
        dynamic = torch.empty((symbol, 4))
        source = graph({"symbol": symbol, "base": base, "view": view,
                        "dynamic": dynamic})
        original = (symbol, base, view, dynamic)
        content = dump_metadata(source, original)
    target = graph(dict.fromkeys(("symbol", "base", "view", "dynamic")))
    restored_mode, inputs = restore_metadata(target, content)
    assert input_contract(inputs) == input_contract(original)
    assert inputs[2].stride() == (1, 4) and inputs[2].storage_offset() == 2
    assert inputs[1].untyped_storage()._cdata == inputs[2].untyped_storage()._cdata
    assert inputs[3].shape[0].node.shape_env is restored_mode.shape_env
    assert all(n.meta["example_value"] is not None
               for n in target.graph.nodes if n.op == "placeholder")


def test_missing_or_mismatched_evidence_rejected():
    source = graph({"x": torch.empty(2)})
    with pytest.raises(ValueError, match="differs"):
        dump_metadata(source, [torch.empty(3)])
    content = dump_metadata(source, [torch.empty(2)])
    with pytest.raises(ValueError, match="names"):
        restore_metadata(graph({"other": None}), content)
    payload = GraphPickler.loads(content, FakeTensorMode(shape_env=ShapeEnv()))
    payload["contract"] = {}
    with pytest.raises(ValueError, match="differs"):
        restore_metadata(source, GraphPickler.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        restore_metadata(source, GraphPickler.dumps({"schema": "old"}))


def test_node_mode_selected_for_real_backend_args():
    tensor = torch.empty(3)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    source = graph({"x": mode.from_tensor(tensor)})
    content = dump_metadata(source, [tensor])
    _, inputs = restore_metadata(graph({"x": None}), content)
    assert isinstance(inputs[0], FakeTensor) and inputs[0].shape == tensor.shape


def test_data_dependent_constant_rejected():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    fake = mode.from_tensor(torch.tensor(1.0))
    fake.constant = torch.tensor(1.0)
    with tracing(TracingContext(mode)), pytest.raises(ValueError, match="constant"):
        dump_metadata(graph({"x": fake}), [fake])
