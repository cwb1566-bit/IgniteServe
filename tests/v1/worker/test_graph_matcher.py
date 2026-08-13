import tempfile
from pathlib import Path

from vllm.v1.worker.GraphMatcher import GraphMatcher


def test_native_registered_padding_is_preserved() -> None:
    matcher = GraphMatcher([4, 8, 16])
    result = matcher.find_best_match(
        5, phase="decode", default_graph_size=8)
    assert result.strategy == "native_pad"
    assert result.execution_size == 8


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
        matcher = GraphMatcher(
            [48], max_pad_ratio=0.20, cost_table_path=str(path))
        accepted = matcher.find_best_match(
            40, phase="decode", default_graph_size=40)
        rejected = matcher.find_best_match(
            39, phase="decode", default_graph_size=40)
        assert accepted.strategy == "pad_up"
        assert accepted.execution_size == 48
        assert rejected.strategy == "eager"


def test_dynamic_decode_graph_is_not_used_for_mixed_batch() -> None:
    matcher = GraphMatcher([48])
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
    matcher = GraphMatcher([48])
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
        matcher = GraphMatcher([48], cost_table_path=str(path))
        result = matcher.find_best_match(
            40, phase="decode", default_graph_size=40)
        assert result.strategy == "eager"
        assert "cost-reject" in result.reason


def test_sparse_padding_requires_cost_evidence() -> None:
    matcher = GraphMatcher([48], max_pad_ratio=0.20)
    result = matcher.find_best_match(
        40, phase="decode", default_graph_size=40)
    assert result.strategy == "eager"
    assert result.graph_size == 48
    assert result.reason == "cost-unavailable-for-sparse-pad"


def test_native_padding_does_not_require_cost_table() -> None:
    matcher = GraphMatcher([48], max_pad_ratio=0.20)
    result = matcher.find_best_match(
        40, phase="decode", default_graph_size=48)
    assert result.strategy == "native_pad"
    assert result.execution_size == 48


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
        matcher = GraphMatcher([48], cost_table_path=str(path))
        result = matcher.find_best_match(
            40, phase="decode", default_graph_size=40)
        assert result.strategy == "pad_up"
        assert result.estimated_eager_ms == 10.0
        assert result.estimated_graph_ms == 9.0


def test_split_keeps_request_and_graph_sizes_distinct() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,40,10.0\n"
            "decode,graph,24,4.0\n",
            encoding="utf-8",
        )
        matcher = GraphMatcher([24], cost_table_path=str(path))
        result = matcher.plan_split("decode", 40)
        assert result.strategy == "split_plan_only"
        assert result.split_request_sizes == (24, 16)
        assert result.split_sizes == (24, 24)
        assert result.execution_size == 48


def test_realistic_sequential_split_is_rejected_but_parallel_bound_is_kept(
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,40,15.28\n"
            "decode,graph,24,14.36\n",
            encoding="utf-8",
        )
        matcher = GraphMatcher([24], cost_table_path=str(path))
        sequential = matcher.plan_split("decode", 40)
        safe_parallel = matcher.plan_split(
            "decode", 40, execution_mode="ideal_parallel")
        unconstrained_parallel = matcher.plan_split(
            "decode", 40, gamma=1.0, execution_mode="ideal_parallel")
        assert sequential.strategy == "eager"
        assert sequential.reason == "split-sequential-not-profitable"
        assert safe_parallel.strategy == "eager"
        assert unconstrained_parallel.strategy == "parallel_split_plan_only"
        assert unconstrained_parallel.split_request_sizes == (24, 16)
        assert unconstrained_parallel.split_sizes == (24, 24)
        assert unconstrained_parallel.estimated_graph_ms == 14.36


def test_calibrated_parallel_rejects_ideal_only_candidate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,40,15.28\n"
            "decode,graph,8,12.69\n"
            "decode,graph,16,13.71\n",
            encoding="utf-8",
        )
        matcher = GraphMatcher([8, 16], cost_table_path=str(path))
        ideal = matcher.plan_split(
            "decode", 40, execution_mode="ideal_parallel")
        calibrated = matcher.plan_split(
            "decode",
            40,
            execution_mode="calibrated_parallel",
            parallel_overlap_efficiency=0.20,
        )
        assert ideal.strategy == "parallel_split_plan_only"
        assert ideal.split_sizes == (16, 16, 8)
        assert calibrated.strategy == "eager"
        assert calibrated.reason == (
            "split-calibrated_parallel-not-profitable")


def test_split_rejects_excessive_aggregate_padding() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "decode,eager,33,20.0\n"
            "decode,graph,32,1.0\n",
            encoding="utf-8",
        )
        matcher = GraphMatcher(
            [32], max_pad_ratio=0.20, cost_table_path=str(path))
        result = matcher.plan_split("decode", 33)
        assert result.strategy == "eager"
        assert result.reason == "no-feasible-split"


def test_mixed_phase_separation_is_costed_in_both_modes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "mixed,eager,9,20.0\n"
            "prefill,eager,5,8.0\n"
            "decode,eager,4,6.0\n"
            "decode,graph,4,4.0\n",
            encoding="utf-8",
        )
        matcher = GraphMatcher([4], cost_table_path=str(path))
        sequential = matcher.plan_phase_separation(5, 4)
        parallel = matcher.plan_phase_separation(
            5, 4, execution_mode="ideal_parallel")
        assert sequential.strategy == (
            "phase_separation_sequential_plan_only")
        assert sequential.decode_graph_size == 4
        assert sequential.estimated_total_ms == 12.0
        assert parallel.strategy == (
            "phase_separation_ideal_parallel_plan_only")
        assert parallel.estimated_total_ms == 8.0


def test_cost_table_does_not_extrapolate_unmeasured_shapes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "costs.csv"
        path.write_text(
            "phase,mode,execution_shape,mean_gpu_ms\n"
            "prefill,eager,8,4.0\n"
            "prefill,eager,16,6.0\n",
            encoding="utf-8",
        )
        matcher = GraphMatcher([], cost_table_path=str(path))
        assert matcher.cost_table.estimate("prefill", "eager", 12) == 5.0
        assert matcher.cost_table.estimate("prefill", "eager", 4) is None
        assert matcher.cost_table.estimate("prefill", "eager", 32) is None


if __name__ == "__main__":
    test_native_registered_padding_is_preserved()
    test_sparse_padding_obeys_ratio_guard()
    test_dynamic_decode_graph_is_not_used_for_mixed_batch()
    test_piecewise_dynamic_graph_can_be_reused_across_phase()
    test_cost_table_can_reject_padding()
    test_sparse_padding_requires_cost_evidence()
    test_native_padding_does_not_require_cost_table()
    test_split_keeps_request_and_graph_sizes_distinct()
    test_realistic_sequential_split_is_rejected_but_parallel_bound_is_kept()
    test_calibrated_parallel_rejects_ideal_only_candidate()
    test_split_rejects_excessive_aggregate_padding()
    test_mixed_phase_separation_is_costed_in_both_modes()
    test_cost_table_does_not_extrapolate_unmeasured_shapes()
    print("graph matcher tests passed")
