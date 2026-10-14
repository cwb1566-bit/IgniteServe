# SPDX-License-Identifier: Apache-2.0
"""Inspect or copy startup assets without importing their executable payloads."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from vllm.compilation import startup_assets as assets
from vllm.compilation.startup_resources import resource_root

KINDS = {"model_info", "kv_profile", "best_config", "input_metadata",
         "transformed_code", "dynamo_frontend", "compile", "hotspots",
         "startup_contract", "startup_cost"}
EXECUTABLE = {"dynamo_frontend", "compile", "startup_contract"}


def files(root):
    for kind in sorted(KINDS):
        for path in sorted((root / kind).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"refusing symlink {path}")
            if path.is_file() and not path.name.endswith((".lock", ".tmp")):
                yield path


def verify(root):
    failures = []
    checked = 0
    for path in files(root):
        if path.suffix != ".json":
            continue
        try:
            value = json.loads(path.read_text())
            if value.get("schema") != assets.SCHEMA:
                continue  # Backend-private files are covered by their bundle.
            checked += 1
            identity = value["identity"]
            if assets.read_record(path, identity) is None:
                raise ValueError("record invalid")
            if path.name == "manifest.json" and not assets.verify_bundle(path.parent, identity):
                raise ValueError("bundle invalid")
            if isinstance(identity, dict) and identity.get("schema") == "subgraph-v1":
                from vllm.compilation.cache_reuse import GlobalSubgraphCache
                if (path.stem != assets.digest(identity)
                        or not GlobalSubgraphCache.verify_entry(root, value["data"])):
                    raise ValueError("subgraph payload missing or invalid")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    return {"checked": checked, "failures": failures}


def export(root, archive):
    if archive.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive must be outside the repository")
    with assets.locked(root / ".transfer"), tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary)
        for path in files(root):
            assets.atomic_write(snapshot / path.relative_to(root), path.read_bytes())
        report = verify(snapshot)
        if report["failures"]:
            raise ValueError(report)
        hashes = {}
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
            for path in files(snapshot):
                name = path.relative_to(snapshot).as_posix()
                content = path.read_bytes()
                hashes[name] = hashlib.sha256(content).hexdigest()
                output.writestr(name, content)
            output.writestr("repository_manifest.json", json.dumps(
                assets.record({"schema": "startup-export-v1"}, {"files": hashes})))


def import_archive(root, archive, trusted):
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read("repository_manifest.json"))
        identity = {"schema": "startup-export-v1"}
        if (manifest.get("schema") != assets.SCHEMA or manifest.get("identity") != identity
                or manifest.get("checksum") != assets.digest([identity, manifest.get("data")])):
            raise ValueError("invalid export manifest")
        names = manifest["data"]["files"]
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("duplicate archive members")
        if set(source.namelist()) != set(names) | {"repository_manifest.json"}:
            raise ValueError("unlisted archive members")
        with tempfile.TemporaryDirectory(dir=root, prefix=".import-") as temporary:
            staging = Path(temporary)
            for name, sha in names.items():
                path = assets.member(staging, name)
                kind = Path(name).parts[0]
                if kind not in KINDS or (kind in EXECUTABLE and not trusted):
                    raise ValueError(f"unsupported/untrusted asset: {name}; code requires --trusted")
                content = source.read(name)
                if hashlib.sha256(content).hexdigest() != sha:
                    raise ValueError(f"checksum mismatch: {name}")
                assets.atomic_write(path, content)
            report = verify(staging)
            if report["failures"]:
                raise ValueError(report)
            with assets.locked(root / ".transfer"):
                # Never overwrite an asset potentially in use. Preflight all
                # destinations before publishing any new file.
                for name, sha in names.items():
                    target = assets.member(root, name)
                    if target.exists() and assets.file_hash(target) != sha:
                        raise ValueError(f"destination differs: {target}")
                for name in sorted(names, key=lambda n: Path(n).name == "manifest.json"):
                    target = assets.member(root, name)
                    if not target.exists():
                        assets.atomic_write(target, assets.member(staging, name).read_bytes())
    return {"imported_files": len(names), "root": str(root)}


def publish(root, kind, source):
    if kind == "hotspots":
        from vllm.v1.worker.cudagraph_profile import load_cudagraph_profile_manifest
        identity = load_cudagraph_profile_manifest(source)
        if not identity:
            raise ValueError("hotspot runtime manifest missing; recollect the profile")
        names = [p.name for p in source.glob("*.csv")]
        if "cuda_graph_hits.csv" not in names:
            raise ValueError("hotspot hit table missing")
        workload = source / "workload.json"
        if workload.exists():
            identity = {"runtime_profile": identity,
                        "workload": json.loads(workload.read_text())}
            names.append("workload.json")
    else:
        contract = json.loads((source / "startup_contract.json").read_text())
        identity = dict(contract.get("compatibility", {}))
        identity.pop("cache_state", None)
        names = ["startup_contract.json", "startup_adapter.py"]
        if contract.get("status") != "final" or not assets.verify_bundle(source, identity):
            raise ValueError("only finalized, verified contract bundles may be published")
    target = root / kind / assets.digest(identity)
    if target.exists():
        raise ValueError(f"already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".publish-") as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for name in names:
            shutil.copy2(assets.member(source, name), stage / name)
        assets.publish_bundle(stage, identity, names)
        with assets.locked(target):
            if target.exists():
                raise ValueError(f"already exists: {target}")
            os.rename(stage, target)
    return {"published": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "verify", "export", "import", "publish-hotspots", "publish-contract"])
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--trusted", action="store_true", help="explicitly trust imported executable code")
    args = parser.parse_args()
    root = resource_root()
    if args.command in {"export", "import", "publish-hotspots", "publish-contract"} and args.path is None:
        parser.error("path required")
    if args.command == "inspect":
        result = {"root": str(root), "files": {kind: sum(p.is_file() and not p.name.endswith(".lock")
                  for p in (root / kind).rglob("*")) for kind in sorted(KINDS)}}
    elif args.command == "verify":
        result = verify(root)
    elif args.command == "export":
        export(root, args.path)
        result = {"exported": str(args.path)}
    elif args.command == "import":
        result = import_archive(root, args.path, args.trusted)
    else:
        result = publish(root, "hotspots" if args.command.endswith("hotspots") else "startup_contract", args.path.resolve())
    print(json.dumps(result, indent=2))
    if result.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
