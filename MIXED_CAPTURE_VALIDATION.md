# Mixed/prefill concurrent piecewise capture

Experimental branch: `igniteServe/mixed-piecewise-capture`, based on
`ba0c340a4fa001b6329f3f199ba89baa03c867d1` plus the top-99 pre-capture change.
Changes are in the working tree, not a released/committed version.

## Implementation

- The controller defaults to decode-only for the old idle path. Only a runner
  passing the concurrent manager capability checks enables mixed and prefill.
- Real requests execute the target padded token shape eagerly. Each compiled
  piece records its actual input signature and completed warmup.
- Capture requests preserve their originating phase through enqueue, start,
  publication and failure. Shape-level deduplication shares one capture across
  phases, resolving other pending records on completion or failure.
- The idle drain does not race the concurrent worker for the same queued graph.
- Capture checks every piece against the real warmup signature: tensor shape,
  stride, dtype, device, layout, storage offset, alias groups, container structure
  and scalar values. Unsupported input objects fail closed.
- Dynamic replay checks that signature and flat tensor input addresses. A failed
  check runs the piece and its entire remaining suffix eagerly; it does not feed
  new eager outputs into downstream graphs bound to old addresses.
- Attention stays outside the piecewise graph. In the tested FlashAttention path,
  background capture supplies no attention metadata, returning before reading or
  writing the live KV cache. Graph kernels are recorded, not executed by capture.
- Existing patched PyTorch stream/allocator/RNG facilities are used unchanged.
  This change does not claim compatibility with unmodified PyTorch.

## Scope

Single GPU, TP=PP=DP=1, text-only piecewise mode. Full attention graphs,
multimodal, hybrid/stateful models, LoRA runtime and speculative decoding remain
rejected by the concurrent manager. Runtime shapes must be registered capture
sizes. Validation below uses FP16 Llama2-7B, not every supported architecture.

## Validation and limitations

Four new CPU/unit tests cover phase lifecycle, deduplicated completion/failure,
signature shape/layout/alias/constant distinctions, and suffix fallback/abort.
The pre-existing controller script also passes. Pytest is not installed in the
runtime environment; these tests were executed through their script entrypoints.

Real tests use GPU3 and 128 ShareGPT requests from the existing round1/RPS1
workload, online threshold 1. Artifacts are under
`A100_0606/async_capture_stability_20260920/`:

- `mixed_validation_20260922_142123`: concurrent versus separately launched
  uncompiled eager. Both complete 128 requests. Concurrent records 28 mixed and
  3 prefill captures with no capture failures. 24 request token sequences differ
  between launches. This is NOT accepted as exact output-equivalence evidence;
  disabling compilation and changing scheduling confound that comparison.
- `mixed_validation_20260922_142906`: same-step shadow diagnostic. The initial
  profile was inadvertently overwritten by the preceding diagnostic because its
  directory was not private; this run starts with four shapes, not six. Retained
  as diagnostic evidence only. The overwritten profile is archived there.
- `mixed_validation_20260922_143240`: final fixed-profile shadow validation with
  a private profile copy and an immutable `hotspots_before` snapshot. Selection
  must be `[1,2,4,8,16,24]`. See its `validation_summary.json` for audited counts.

Final fixed-profile result: 128/128 requests, zero request/capture failures;
28 mixed-triggered and 3 prefill-triggered captures (33 pieces per shape).
New shapes replayed 34 times in mixed and 4 times in prefill. All 38 same-step
shadow comparisons passed with observed maximum logit difference 0 and exact
greedy-token agreement. There were 363 distinct piece/shape first-replay contract
checks, zero warmup signature failures and zero replay contract fallbacks.
GPU3 memory returned to 0 MiB after cleanup.

The original frozen profile was restored from the untouched offline output,
whose full CSV SHA256 is
`a49a11db8c5299be8d0a0ea1a4854f4181a39434de9813e92bed24b8be634460`.
Previous five formal runs used private copies and are not modified.

Shadow mode (`VLLM_MIXED_CAPTURE_SHADOW_VALIDATE=1`) is diagnostic-only and limited
to single-GPU Llama. Each dynamic replay is compared with compiled-eager on the
same scheduler step, tokens, positions, attention metadata and KV slots, before
sampling. Hidden outputs are cloned before the reference forward. It asserts
logit closeness (rtol=atol=1e-3), exact greedy-token agreement, and logs observed
maximum absolute logit error. KV writes repeat the same token/slot values in this
stateless-attention test; this is not a generic shadow facility for other models.

These are correctness/compatibility probes, not a new 2000-request threshold
performance matrix. Diagnostic extra forwards and signature checks have costs;
latency impact still needs separate measurement. Cross-run token mismatch has
not been attributed conclusively, even if same-step graph/eager checks pass.
