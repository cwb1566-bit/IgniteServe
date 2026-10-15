import csv

import pytest

from vllm.v1.worker.HotspotAnalyzer import BatchSizeStats, HotspotAnalyzer
from vllm.v1.worker.cudagraph_profile import ensure_cudagraph_profile_manifest


def _read_counts(path):
    with open(path, newline="", encoding="utf-8") as file:
        return {
            int(row["batch_size"]): int(row["hit_count"])
            for row in csv.DictReader(file)
        }


def test_export_merges_engine_generations_without_double_counting(tmp_path):
    analyzer = HotspotAnalyzer()
    analyzer.output_dir = tmp_path

    analyzer.exporter_id = "engine-a"
    analyzer.total_requests = 2
    analyzer.stats = {1: BatchSizeStats(size=1, hit_count=2, total_tokens=2)}
    analyzer.export_to_csv()

    analyzer.exporter_id = "engine-b"
    analyzer.total_requests = 3
    analyzer.stats = {2: BatchSizeStats(size=2, hit_count=3, total_tokens=6)}
    analyzer.export_to_csv()
    assert _read_counts(tmp_path / "cuda_graph_hits.csv") == {1: 2, 2: 3}

    # A later cumulative snapshot from the same engine replaces its part.
    analyzer.total_requests = 4
    analyzer.stats[2].hit_count = 4
    analyzer.stats[2].total_tokens = 8
    analyzer.export_to_csv()
    assert _read_counts(tmp_path / "cuda_graph_hits.csv") == {1: 2, 2: 4}


def test_export_preserves_incompatible_profile(tmp_path):
    old = {'schema_version': 1, 'software': {'vllm': 'old'}}
    new = {'schema_version': 1, 'software': {'vllm': 'new'}}
    ensure_cudagraph_profile_manifest(tmp_path, old)
    analyzer = HotspotAnalyzer()
    analyzer.output_dir = tmp_path
    analyzer.profile_manifest = old
    analyzer.exporter_id = 'old-engine'
    analyzer.total_requests = 2
    analyzer.stats = {1: BatchSizeStats(size=1, hit_count=2, total_tokens=2)}
    analyzer.export_to_csv()
    csv_path = tmp_path / 'cuda_graph_hits.csv'
    before = csv_path.read_bytes()
    analyzer.profile_manifest = new
    analyzer.exporter_id = 'incompatible-engine'
    analyzer.stats = {8: BatchSizeStats(size=8, hit_count=100, total_tokens=800)}
    analyzer.export_to_csv()
    assert csv_path.read_bytes() == before
    assert not (tmp_path / '.cuda_graph_hits.part-incompatible-engine.csv').exists()


def test_export_rejects_missing_manifest(tmp_path):
    analyzer = HotspotAnalyzer()
    analyzer.output_dir = tmp_path
    analyzer.profile_manifest = {'schema_version': 1}
    analyzer.total_requests = 2
    analyzer.stats = {1: BatchSizeStats(size=1, hit_count=2, total_tokens=2)}
    analyzer.export_to_csv()
    assert not (tmp_path / 'cuda_graph_hits.csv').exists()
