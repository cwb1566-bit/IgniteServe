#!/usr/bin/env python3
"""Attach matching, already-built vLLM artifacts to a new source checkout.

This is a local development helper, not a replacement for building a release
wheel. Run it with the patched PyTorch environment used to build the artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, target: Path) -> dict[str, str]:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError('Source and destination must be different checkouts')
    native_paths = ['csrc', 'cmake', 'CMakeLists.txt']
    tracked = subprocess.check_output(
        ['git', '-C', str(target), 'ls-files', '-z', '--', *native_paths])
    for name in tracked.decode().split('\0'):
        if not name:
            continue
        src, dst = source / name, target / name
        if not src.is_file() or digest(src) != digest(dst):
            raise ValueError(f'Native source mismatch: {name}; rebuild instead')
    files = [Path('vllm') / name for name in
             ['_C.abi3.so', '_moe_C.abi3.so', 'cumem_allocator.abi3.so']]
    flash = source / 'vllm/vllm_flash_attn'
    files += [p.relative_to(source) for p in sorted(flash.rglob('*'))
              if p.is_file() and p.suffix in {'.py', '.so'}]
    for required in ['__init__.py', 'layers/rotary.py', '_vllm_fa2_C.abi3.so']:
        if not (flash / required).is_file():
            raise ValueError(f'Missing FlashAttention artifact: {required}')
    hashes = {}
    for relative in files:
        src, dst = source / relative, target / relative
        if not src.is_file():
            raise ValueError(f'Missing artifact: {src}')
        hashes[str(relative)] = digest(src)
        if dst.is_symlink() and not dst.exists():
            raise ValueError(f'Dangling artifact symlink: {dst}; refusing overwrite')
        if dst.exists() and digest(dst) != hashes[str(relative)]:
            raise ValueError(f'Existing artifact differs: {dst}; refusing overwrite')
    for relative in files:
        dst = target / relative
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dst)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-checkout', type=Path, required=True)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1]
    if importlib.util.find_spec('setuptools_scm') is None:
        sys.path.insert(0, str(target / '.deps/build-tools'))
    from setuptools_scm import get_version
    hashes = prepare(args.from_checkout, target)
    version = get_version(root=str(target), write_to='vllm/_version.py')
    print(f'Prepared {len(hashes)} artifacts; generated source version {version}')
    print('CUDA/ABI compatibility still requires a real model smoke test.')


if __name__ == '__main__':
    main()
