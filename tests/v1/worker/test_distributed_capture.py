# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the communication-free capture boundary and buffer contract."""
import importlib.util
import ast
import json
from pathlib import Path
import threading

import pytest
import torch

PATH = Path(__file__).resolve().parents[3] / 'vllm/v1/worker/distributed_capture.py'
spec = importlib.util.spec_from_file_location('distributed_capture_test', PATH)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_recording_is_thread_local():
    token = capture._recording.set(True)
    seen = []
    try:
        thread = threading.Thread(target=lambda: seen.append(capture.recording()))
        thread.start()
        thread.join()
        assert capture.recording()
        assert seen == [False]
    finally:
        capture._recording.reset(token)
    assert not capture.recording()


def test_private_inputs_copy_activations_not_weights():
    activation, weight = torch.ones(3, 4), torch.ones(4, 4)
    static = {weight.untyped_storage().data_ptr()}
    dest = capture.private_inputs((activation, weight, [2, None]), static)
    assert dest[0].data_ptr() != activation.data_ptr()
    assert dest[1] is weight
    capture.copy_inputs(dest, (activation, weight, [2, None]))
    torch.testing.assert_close(dest[0], activation)
    dest[0].add_(2)
    capture.copy_inputs((activation, weight, [2, None]), dest)
    torch.testing.assert_close(activation, torch.full_like(activation, 3))
    activation.fill_(7)
    capture.copy_inputs(dest, (activation, weight, [2, None]))
    torch.testing.assert_close(dest[0], activation)


def test_unsafe_input_aliases_are_rejected():
    x = torch.ones(4, 4)
    for args in ((x, x), (x[1:],), (x.t(),)):
        with pytest.raises(ValueError, match='unaliased contiguous'):
            capture.private_inputs(args, set())


def test_collectives_are_boundaries():
    assert set(capture.COLLECTIVE_OPS) == {
        'vllm.all_reduce', 'vllm.all_gather', 'vllm.reduce_scatter'}


def test_stream_cache_is_thread_local():
    # Execute the actual setter without importing CUDA-dependent engine modules.
    path = PATH.parents[2] / 'utils/__init__.py'
    tree = ast.parse(path.read_text())
    setter = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_patched_set_stream')
    state = threading.local()
    namespace = dict(torch=torch, _current_stream_state=state,
                     prev_set_stream=lambda stream: None)
    exec(compile(ast.Module(body=[setter], type_ignores=[]), str(path), 'exec'), namespace)
    set_stream = namespace['_patched_set_stream']
    set_stream('foreground')
    seen = []

    def background():
        set_stream('capture')
        seen.append(state.stream)

    thread = threading.Thread(target=background)
    thread.start()
    thread.join()
    assert seen == ['capture']
    assert state.stream == 'foreground'


def test_distributed_contract_has_one_rpc_lane(tmp_path):
    root = PATH.parents[3]
    spec = importlib.util.spec_from_file_location(
        'contract_builder_test', root / 'tools/build_startup_contract.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for distributed in (False, True):
        output = tmp_path / 'contract.json'
        builder.build(root, output, distributed)
        payload = json.loads(output.read_text())
        assert payload['compatibility']['distributed'] is distributed
        assert payload['resource_capacities']['worker_rpc'] == (1 if distributed else 2)
        assert len(payload['nodes']) == 26
        assert payload['status'] == 'candidate'
