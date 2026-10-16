import tempfile
from pathlib import Path

from vllm.v1.worker.cudagraph_adaptation import GraphAdapter


def test_native_registered_padding_obeys_ratio() -> None:
    matcher = GraphAdapter(captured_sizes=[4, 8, 16])
    result = matcher.find_best_match(5, phase="decode", default_graph_size=8)
    assert result.strategy == "eager"
    assert result.execution_size == 5


def test_sparse_padding_obeys_ratio_guard() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,39,20.0\n"
            "decode,eager,40,20.0\n"
            "decode,graph,48,10.0\n",
            encoding="utf-8",
        )
        matcher = GraphAdapter(
            captured_sizes=[48], max_pad_ratio=0.20, cost_table_path=str(path)
        )
        accepted = matcher.find_best_match(
            40, phase="decode", default_graph_size=40
        )
        rejected = matcher.find_best_match(
            39, phase="decode", default_graph_size=40
        )
        assert accepted.strategy == "pad_up"
        assert accepted.execution_size == 48
        assert rejected.strategy == "eager"


def test_dynamic_decode_graph_is_not_used_for_mixed_batch() -> None:
    matcher = GraphAdapter(captured_sizes=[48])
    decode = matcher.find_best_match(
        48,
        phase="decode",
        default_graph_size=48,
        dynamic_sizes={48},
    )
    mixed = matcher.find_best_match(
        48,
        phase="mixed",
        default_graph_size=48,
        dynamic_sizes={48},
    )
    assert decode.strategy == "exact"
    assert mixed.strategy == "eager"


def test_piecewise_dynamic_graph_can_be_reused_across_phase() -> None:
    matcher = GraphAdapter(captured_sizes=[48])
    mixed = matcher.find_best_match(
        48,
        phase="mixed",
        default_graph_size=48,
        dynamic_sizes={48},
        allow_dynamic_cross_phase=True,
    )
    assert mixed.strategy == "exact"
    assert mixed.execution_size == 48


def test_cost_table_can_reject_padding() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,40,1.0\n"
            "decode,graph,48,1.1\n",
            encoding="utf-8",
        )
        matcher = GraphAdapter(captured_sizes=[48], cost_table_path=str(path))
        result = matcher.find_best_match(
            40, phase="decode", default_graph_size=40
        )
        assert result.strategy == "eager"
        assert "cost-reject" in result.reason


def test_sparse_padding_requires_cost_evidence() -> None:
    matcher = GraphAdapter(captured_sizes=[48], max_pad_ratio=0.20)
    result = matcher.find_best_match(40, phase="decode", default_graph_size=40)
    assert result.strategy == "eager"
    assert result.graph_size == 48
    assert result.reason == "cost-unavailable-for-sparse-pad"


def test_native_padding_requires_cost_table() -> None:
    matcher = GraphAdapter(captured_sizes=[48], max_pad_ratio=0.20)
    result = matcher.find_best_match(40, phase="decode", default_graph_size=48)
    assert result.strategy == "eager"
    assert result.execution_size == 40


def test_duplicate_cost_samples_are_averaged_by_shape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,40,8.0\n"
            "decode,eager,40,12.0\n"
            "decode,graph,48,9.0\n",
            encoding="utf-8",
        )
        matcher = GraphAdapter(captured_sizes=[48], cost_table_path=str(path))
        result = matcher.find_best_match(
            40, phase="decode", default_graph_size=40
        )
        assert result.strategy == "pad_up"
        assert result.estimated_eager_ms == 10.0
        assert result.estimated_graph_ms == 9.0


def test_cost_table_does_not_extrapolate_unmeasured_shapes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "prefill,eager,8,4.0\n"
            "prefill,eager,16,6.0\n",
            encoding="utf-8",
        )
        matcher = GraphAdapter(captured_sizes=[], cost_table_path=str(path))
        assert matcher.cost_table.estimate("prefill", "eager", 12) == 5.0
        assert matcher.cost_table.estimate("prefill", "eager", 4) is None
        assert matcher.cost_table.estimate("prefill", "eager", 32) is None
