from vllm.v1.worker.adaptive_cudagraph import (
    AdaptiveCUDAGraphController,
    GraphState,
)


def test_decode_shape_transitions_through_runtime_warmup() -> None:
    controller = AdaptiveCUDAGraphController(enabled=True, min_hits=2)

    first = controller.plan_runtime("decode", 13, 16, True, False)
    assert first.execution_shape == 13
    assert first.skip_cudagraph
    assert not first.mark_runtime_warmup

    threshold = controller.plan_runtime("decode", 13, 16, True, False)
    assert threshold.execution_shape == 16
    assert threshold.skip_cudagraph
    assert threshold.mark_runtime_warmup

    controller.finish_runtime_warmup("decode", 16)
    assert controller.take_queued_shapes() == [16]
    controller.mark_capture_ready("decode", 16, 0.1)

    ready = controller.plan_runtime("decode", 13, 16, True, True)
    assert ready.execution_shape == 16
    assert not ready.skip_cudagraph
    record = controller.snapshot()[0]
    assert record.state == GraphState.READY
    assert record.capture_finished_at > 0


def test_mixed_and_prefill_are_observed_but_not_captured() -> None:
    controller = AdaptiveCUDAGraphController(enabled=True, min_hits=1)

    for phase in ("mixed", "prefill"):
        decision = controller.plan_runtime(phase, 13, 16, True, False)
        assert decision.execution_shape == 13
        assert decision.skip_cudagraph
        assert not decision.mark_runtime_warmup

    assert controller.take_queued_shapes() == []


def test_startup_graph_preserves_existing_replay() -> None:
    controller = AdaptiveCUDAGraphController(enabled=True, min_hits=10)
    decision = controller.plan_runtime("mixed", 13, 16, True, True)
    assert decision.execution_shape == 16
    assert not decision.skip_cudagraph


def test_cost_policy_can_force_ready_graph_to_eager() -> None:
    controller = AdaptiveCUDAGraphController(enabled=True, min_hits=10)
    controller.register_ready_shapes([16])
    decision = controller.plan_runtime(
        "decode", 13, 16, True, False, allow_replay=False)
    assert decision.execution_shape == 13
    assert decision.skip_cudagraph
    assert decision.reason == "cost-policy-eager"


def test_idle_capture_selection_is_bounded_and_frequency_ordered() -> None:
    controller = AdaptiveCUDAGraphController(enabled=True, min_hits=1)
    for shape in (8, 16):
        decision = controller.plan_runtime("decode", shape, shape, True, False)
        assert decision.mark_runtime_warmup
        controller.finish_runtime_warmup("decode", shape)
    controller.plan_runtime("decode", 16, 16, True, False)

    assert controller.take_queued_shapes(limit=1) == [16]
    assert controller.take_queued_shapes(limit=1) == [8]


def test_split_shadow_claims_each_shape_once() -> None:
    controller = AdaptiveCUDAGraphController(enabled=True, min_hits=1)
    assert controller.claim_split_shadow_shape("decode", 40)
    assert not controller.claim_split_shadow_shape("decode", 40)
    assert controller.claim_split_shadow_shape("decode", 41)


if __name__ == "__main__":
    test_decode_shape_transitions_through_runtime_warmup()
    test_mixed_and_prefill_are_observed_but_not_captured()
    test_startup_graph_preserves_existing_replay()
    test_cost_policy_can_force_ready_graph_to_eager()
    test_idle_capture_selection_is_bounded_and_frequency_ordered()
    test_split_shadow_claims_each_shape_once()
    print("adaptive controller tests passed")
