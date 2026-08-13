from vllm.v1.worker.cudagraph_batch_partition import (
    partition_mixed_batch,
    partition_pure_decode,
)


def test_mixed_partition_preserves_request_token_and_logits_maps() -> None:
    prefill, decode = partition_mixed_batch(
        query_lens=[3, 1, 2, 1],
        is_prefill=[True, False, True, False],
    )
    assert prefill.phase == "prefill"
    assert prefill.request_indices == (0, 2)
    assert prefill.token_indices == (0, 1, 2, 4, 5)
    assert prefill.logits_indices == (2, 5)
    assert prefill.query_lens == (3, 2)
    assert decode.phase == "decode"
    assert decode.request_indices == (1, 3)
    assert decode.token_indices == (3, 6)
    assert decode.logits_indices == (3, 6)


def test_decode_partition_distinguishes_payload_from_graph_padding() -> None:
    parts = partition_pure_decode(
        num_requests=40,
        request_sizes=(24, 16),
        graph_sizes=(24, 24),
    )
    assert parts[0].request_indices == tuple(range(24))
    assert parts[0].padding_size == 0
    assert parts[1].request_indices == tuple(range(24, 40))
    assert parts[1].padding_size == 8


def test_invalid_decode_partition_is_rejected() -> None:
    try:
        partition_pure_decode(40, (24, 15), (24, 24))
    except ValueError as error:
        assert "does not cover" in str(error)
    else:
        raise AssertionError("invalid partition should fail")


if __name__ == "__main__":
    test_mixed_partition_preserves_request_token_and_logits_maps()
    test_decode_partition_distinguishes_payload_from_graph_padding()
    test_invalid_decode_partition_is_rejected()
    print("CUDA graph batch partition tests passed")
