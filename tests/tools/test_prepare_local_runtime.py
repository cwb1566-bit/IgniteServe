from pathlib import Path
import runpy

import pytest


prepare = runpy.run_path(str(Path(__file__).resolve().parents[2]
                            / 'tools/prepare_local_runtime.py'))['prepare']


def _checkouts(tmp_path, monkeypatch):
    source, target = tmp_path / 'source', tmp_path / 'target'
    for root in [source, target]:
        root.mkdir()
        (root / 'CMakeLists.txt').write_text('same native source')
    files = ['_C.abi3.so', '_moe_C.abi3.so', 'cumem_allocator.abi3.so',
             'vllm_flash_attn/__init__.py', 'vllm_flash_attn/layers/rotary.py',
             'vllm_flash_attn/_vllm_fa2_C.abi3.so']
    for name in files:
        path = source / 'vllm' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture artifact')
    monkeypatch.setattr('subprocess.check_output',
                        lambda *args, **kwargs: b'CMakeLists.txt\0')
    return source, target


def test_prepare_copies_complete_runtime_and_is_idempotent(tmp_path, monkeypatch):
    source, target = _checkouts(tmp_path, monkeypatch)
    first = prepare(source, target)
    assert len(first) == 6
    assert (target / 'vllm/vllm_flash_attn/layers/rotary.py').is_file()
    assert prepare(source, target) == first


def test_prepare_rejects_native_source_mismatch(tmp_path, monkeypatch):
    source, target = _checkouts(tmp_path, monkeypatch)
    (source / 'CMakeLists.txt').write_text('different source')
    with pytest.raises(ValueError, match='Native source mismatch'):
        prepare(source, target)
    assert not (target / 'vllm').exists()


def test_prepare_rejects_incomplete_flash_package(tmp_path, monkeypatch):
    source, target = _checkouts(tmp_path, monkeypatch)
    (source / 'vllm/vllm_flash_attn/layers/rotary.py').unlink()
    with pytest.raises(ValueError, match='Missing FlashAttention artifact'):
        prepare(source, target)
    assert not (target / 'vllm').exists()


def test_prepare_does_not_overwrite_different_artifact(tmp_path, monkeypatch):
    source, target = _checkouts(tmp_path, monkeypatch)
    (target / 'vllm').mkdir()
    artifact = target / 'vllm/_C.abi3.so'
    artifact.write_bytes(b'keep existing artifact')
    with pytest.raises(ValueError, match='Existing artifact differs'):
        prepare(source, target)
    assert artifact.read_bytes() == b'keep existing artifact'


def test_prepare_rejects_dangling_symlink(tmp_path, monkeypatch):
    source, target = _checkouts(tmp_path, monkeypatch)
    (target / 'vllm').mkdir()
    external = tmp_path / 'must-not-create.so'
    (target / 'vllm/_C.abi3.so').symlink_to(external)
    with pytest.raises(ValueError, match='Dangling artifact symlink'):
        prepare(source, target)
    assert not external.exists()
