import json
import tempfile
from pathlib import Path

from vllm.v1.worker.cudagraph_profile import (
    compare_cudagraph_profile_manifests,
    ensure_cudagraph_profile_manifest,
    load_cudagraph_profile_manifest,
    validate_cudagraph_profile_manifest,
)


def test_manifest_is_created_and_reused() -> None:
    current = {"schema_version": 1, "model": {"hidden_size": 4096}}
    with tempfile.TemporaryDirectory() as directory:
        compatible, reason = ensure_cudagraph_profile_manifest(
            directory, current)
        assert compatible
        assert reason == "manifest-created"
        assert load_cudagraph_profile_manifest(directory) == current

        compatible, reason = ensure_cudagraph_profile_manifest(
            directory, current)
        assert compatible
        assert reason == "manifest-match"


def test_mismatch_is_not_silently_overwritten() -> None:
    old = {"schema_version": 1, "model": {"hidden_size": 4096}}
    current = {"schema_version": 1, "model": {"hidden_size": 5120}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cuda_graph_profile_manifest.json"
        path.write_text(json.dumps(old), encoding="utf-8")
        compatible, reason = ensure_cudagraph_profile_manifest(
            directory, current)
        assert not compatible
        assert "model.hidden_size" in reason
        assert load_cudagraph_profile_manifest(directory) == old


def test_explicit_overwrite_replaces_incompatible_manifest() -> None:
    old = {"schema_version": 1, "model": {"hidden_size": 4096}}
    current = {"schema_version": 1, "model": {"hidden_size": 5120}}
    with tempfile.TemporaryDirectory() as directory:
        ensure_cudagraph_profile_manifest(directory, old)
        compatible, reason = ensure_cudagraph_profile_manifest(
            directory, current, allow_overwrite=True)
        assert compatible
        assert reason == "manifest-replaced"
        assert load_cudagraph_profile_manifest(directory) == current


def test_manifest_comparison_reports_missing_fields() -> None:
    mismatches = compare_cudagraph_profile_manifests(
        {"schema_version": 1, "device": {"name": "A100"}},
        {"schema_version": 1},
    )
    assert mismatches == ["device.name: expected='A100', found=None"]


def test_legacy_resources_require_explicit_adoption() -> None:
    current = {"schema_version": 1, "model": {"hidden_size": 4096}}
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "cuda_graph_hits.csv").write_text(
            "batch_size,hit_count\n1,10\n", encoding="utf-8")
        compatible, reason = ensure_cudagraph_profile_manifest(
            directory, current)
        assert not compatible
        assert reason.startswith("manifest-missing-for-existing-resources")
        assert load_cudagraph_profile_manifest(directory) is None

        compatible, reason = ensure_cudagraph_profile_manifest(
            directory, current, allow_overwrite=True)
        assert compatible
        assert reason == "manifest-created"
        assert validate_cudagraph_profile_manifest(
            directory, current) == (True, "manifest-match")
