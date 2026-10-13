# SPDX-License-Identifier: Apache-2.0
"""Exercise worker eligibility without constructing a GPU worker."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def eligible():
    source = Path(__file__).parents[2] / "vllm/v1/worker/gpu_worker.py"
    tree = ast.parse(source.read_text())
    worker = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "Worker")
    method = next(node for node in worker.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_weight_compile_overlap_eligibility")
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[method.name]


def config(architectures=None, **overrides):
    values = dict(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures),
            quantization=None),
        speculative_config=None,
        lora_config=None,
        load_config=SimpleNamespace(load_format="safetensors"),
    )
    values.update(overrides)
    return SimpleNamespace(vllm_config=SimpleNamespace(**values))


@pytest.mark.parametrize("architectures", [
    ["LlamaForCausalLM"], ["Qwen2ForCausalLM"], ["Qwen3ForCausalLM"],
    ["PreviouslyUnknownModel"], ["ModelA", "ModelB"], [], None,
])
def test_architecture_names_do_not_limit_overlap(eligible, architectures):
    assert eligible(config(architectures)) == (True, [])


def test_hf_architecture_metadata_is_not_required(eligible):
    worker = config()
    del worker.vllm_config.model_config.hf_config
    assert eligible(worker) == (True, [])


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_supported_load_formats(eligible, load_format):
    assert eligible(config(load_config=SimpleNamespace(
        load_format=load_format))) == (True, [])


@pytest.mark.parametrize("attribute, value, reason", [
    ("model_config", SimpleNamespace(quantization="awq"), "quantization=awq"),
    ("speculative_config", object(), "speculative_config"),
    ("lora_config", object(), "lora_config"),
    ("load_config", SimpleNamespace(load_format="pt"), "load_format=pt"),
])
def test_configuration_safety_checks_remain(eligible, attribute, value, reason):
    assert eligible(config(**{attribute: value})) == (False, [reason])
