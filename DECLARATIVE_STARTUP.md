# Declarative startup

`VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1` selects the packaged
`vllm/v1/engine/startup_contract.json` and `vllm_startup_adapter.py`.
`VLLM_STARTUP_DAG_MODE=off` keeps the sequential path. An explicit
`VLLM_STARTUP_CONTRACT_PATH` / `VLLM_STARTUP_ADAPTER_PATH` selects a trusted
alternative pair. Python adapters are executable code, not sandboxed data.
Old v2 contracts remain available through `execute-generated`.

## Separation

* `igniteserve_startup/runtime.py`: stdlib-only state exchange, readiness,
  resource accounting, critical-path priority, affinity and completion events.
* `igniteserve_startup/contract.py`: source/adapter integrity, compatibility,
  variant, state producer, resource and cycle validation before binding.
* `vllm/v1/engine/vllm_startup_adapter.py`: engine actions and asset probes;
  no node ordering or DAG construction.
* `tools/build_startup_contract.py`: offline reviewed recipe; never called
  by serving startup. Normal deferred startup enters through a small `core.py`
  hook. Legacy shadow plans and persistent-shell rebind keep their existing
  compatibility builders; they do not generate the new cold-start DAG online.

The scope is EngineCore construction through engine readiness, not HTTP server
binding or autoscaler deployment. Compilation and CUDA capture remain composite
manager actions. Their internal optimizations are retained; this change does
not claim the outer DAG independently schedules every compiler subgraph.
The packaged adapter is currently qualified for single-GPU CUDA. Other device
types and TP/PP/DP configurations retain the existing sequential/distributed
path, rather than silently treating single-GPU validation as distributed proof.

## State and safety

`dependencies` are hard completion barriers. `required_state` can release a
consumer as soon as its producer publishes valid state, even while unrelated
producer work continues. Each state has one reviewed producer. For mutable
objects, reviewers must retain control/resource constraints preventing races.
A returned Future holds the action's resources until real completion. CUDA
adapters must establish stream/event completion before publishing GPU-ready
state; merely submitting a kernel is not readiness.

`main_thread` means the calling thread; other affinity names use persistent
single-thread executors. Resource capacities are admission-control budgets,
not claims to predict GPU occupancy or physical allocator usage.
State probes update pending-node costs and recompute priority. Cache existence
never authorizes skipping an action: the existing per-asset validators decide
whether to load or rebuild. The prewarm variants describe control-flow choices,
not user-supplied cold/hot labels.

`prepare_startup_assets` runs after model-runner construction, before model
initialization and weight loading. It keeps bounded JSON/CSV contents in the
consuming worker: an exact KV-profile candidate, hotspot inputs and speculative
input-metadata/best-config records. The limit is 32 files / 16 MiB, not a recursive
read of the asset repository. A prepared read is not a cache hit. Consumers
still check identity, checksum, graph/input contracts and current memory as
before; file replacement or modification invalidates the prepared copy. JSON
consumers receive separate copies. `ready` clears the worker's preparation store.
Compiled binaries and Dynamo bytecode retain their existing late binding and
validation; they are not deserialized speculatively before their keys exist.
Cached Inductor graph import/binding is serialized across loader threads to
protect shared Triton launcher-module initialization. Compilation in separate
processes remains parallel.

Optional node `routes` select an action, cost, resource demand and affinity using
boolean `when` predicates on declared input state. All alternatives retain the
same dependencies and outputs. The loader rejects missing actions, overlapping
predicates and invalid resources before startup. Pending routes and downstream
priorities are reconsidered on state/completion events; dispatch freezes the
chosen route and its resource reservation. Running work is never restarted.
Running-node cost uses estimated remaining time when an event wakes scheduling.
This is a scheduling heuristic, not an optimal wall-time guarantee.

After the fused weight/compile action joins, `compile_model` selects the
zero-resource `already_compiled` marker. With overlap disabled it runs the usual
compile action. KV readiness selects `try_cached_profile` only after loaded
weights have been checked; the KV action revalidates free memory at consumption
and can still fall back to profiling, so it retains the GPU resource reservation.

Malformed/incompatible contracts fall back before any action. After an action
starts, failures clean up the engine and propagate; initialization is not
silently run twice. Adapter import and registry construction must be pure.

## Offline review and acceptance

The existing `tools/startup_offline_analysis.py` retains root tracing,
AST/use-def extraction, candidate adapters and opaque review. The portable
`python -m igniteserve_startup.offline analyze` additionally exposes conservative
source regions without importing an engine. Neither claims to infer arbitrary
C++/allocator/alias semantics. This vLLM recipe uses reviewed deferred executor
boundaries, not an assertion that all boundaries were automatically inferred.

To regenerate after source changes, run from the repository root:

```sh
PYTHONPATH=. python tools/build_startup_contract.py --output candidate.json
# Review every action, state producer, control edge and affinity against source.
PYTHONPATH=. python tools/validate_startup_contract.py \
  --contract candidate.json --model /path/to/model --gpu 0 --output validation.json
PYTHONPATH=. python -m igniteserve_startup.offline finalize \
  --candidate candidate.json --adapter vllm/v1/engine/vllm_startup_adapter.py \
  --review review.json --validation validation.json --root . \
  --output vllm/v1/engine/startup_contract.json
```

Review JSON binds `identity` to the candidate and includes `reviewer`,
`actions` (each node explicitly `approved`), `constraints: approved`, and
`rationale`. Finalization requires matching serial and scheduled evidence for
all variants. Acceptance runs isolated processes, compares greedy token IDs,
checks exactly-once node coverage and real overlap, and removes raw temporary
traces/cache data automatically. This is empirical validation, not a proof for
all model/configuration combinations.

## Hierarchical compilation reuse

With `VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1`, the existing global-subgraph switch
defaults on. Explicit `VLLM_ENABLE_GLOBAL_SUBGRAPH_CACHE=0` disables persistent
global reuse; `VLLM_ENABLE_GLOBAL_SUBGRAPH_DIRECT_LOOKUP=0` disables its reads.
No new switch is required. Local backend caches remain in their original layout:

```text
~/igniteserve_assets/compile/
  subgraphs/<semantic-key>.json
  <launch-key>/rank_<rank>_<dp_rank>/<prefix>/...local maps...
  <launch-key>/rank_<rank>_<dp_rank>/inductor_cache/...
  <launch-key>/rank_<rank>_<dp_rank>/triton_cache/...
```

`VLLM_STARTUP_RESOURCE_ROOT` changes the repository root. A SubArtifact descriptor
contains the canonical graph/input-output identity, backend/options/native-code
identity, distributed context, relative handle and payload checksums. Its source
index is provenance, not a lookup condition. Unknown code dependencies or missing
output evidence conservatively disable semantic reuse for that graph. Freezing
weights into code is not supported by this value-free semantic identity.

The manager checks instance memory, the launch-local map, then the global index
before scheduling compilation. Same-key graphs with matching current symbolic
constraints share a source Future. Persistent loading still evaluates Inductor
guards with the current ShapeEnv and binds current constants; a JSON hit alone
does not certify reuse. New subprocess handles are published only after successful
parent loading. Invalid/missing payloads fall back to compilation.

Global loading passes verified FX-cache bytes to the backend without changing
process-wide cache directories. It does not copy the entire producer cache;
Inductor may rematerialize runtime wrappers/kernel files in the consumer cache.
Supported persistent loading is the current ThreadSafeInductorAdaptor FX format
(Torch 2.6+); other adaptors retain their local paths. Explicit cache directories
outside the repository are not exported to the global index.

Export/import retains the existing trusted-code requirement and verifies descriptor
references. Transfer the whole repository, not just `subgraphs/`. Producer launch
directories may remain referenced and must not be independently deleted as unused
cache. Relocation does not waive backend/ABI/guard checks; unavailable backend
dependencies are misses. Checksums are corruption checks, not authentication.

The optional `VLLM_TEST_REUSE_GPU=1` test in
`tests/compile/test_hierarchical_reuse.py` exercises separate producer/consumer
processes, different launch paths and indices, and different parameter values.
Startup acceptance also requires actual global hits from a new compilation
directory while preserving greedy outputs and generated-DAG coverage.

## Cost-Gated Graph Adaptation

The adaptive matcher applies the padding ratio guard to native and sparse
targets alike (default 0.2). A padded replay requires compatible measured costs.
Exact matches retain the normal replay path. Device free memory is checked
at most once per second; pressure reduces the ratio to 0.1 or zero with hysteresis.
Unused graph-private pool memory is not treated as reclaimable. Allocation
failure before any forward safely declines splitting; after KV writes there is
no whole-batch retry.

`VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1` also enables cost-gated graph partitioning;
`VLLM_ENABLE_CUDAGRAPH_SPLIT=0` disables only splitting. No qualified cost profile
means no split. The initial implementation supports single-GPU piecewise graphs
with ordinary FlashAttention metadata, including decode, prefill, and mixed
batches. Cascade/local attention, full graphs, TP/PP/DP, LoRA, speculation, hybrid
state, KV connectors, encoder-decoder, and multimodal paths decline splitting
before any KV writes. These are capability checks, not model-name allowlists.

Partitions contain whole contiguous requests. Each part preserves original KV
slots, stages tokens into stable input buffers, replays on the foreground stream,
and copies hidden states before another replay can overwrite them. One original
logits/sampling path handles the assembled output, including prompt logprobs and
partial-prefill handling. An execution error is propagated, never retried as an
entire batch after partial KV writes. Contract fallback completes the affected
part eagerly and disables future splitting for that runner; logs distinguish
actual piece replays from mere split admission. Background capture remains
independent; a split is not counted as warmup for a missing larger graph.
Original-shape hits still authorize capture: the threshold-triggering request
performs whole-shape eager warmup, overriding a split candidate. Requests below
threshold or waiting for publication may reuse smaller graphs. Planning and
execution check every piece and its publication gate; a lost/missing small graph
declines before KV writes. Unknown costs or missing parts fall back to whole
eager, not unpriced hybrid execution.
The padding bound applies to the total tokens across all parts, not separately
to a small tail: 7 tokens may use 4+4 (14.3% total padding) instead of 4+2+1.
The padded tail still requires its own measured workload cost. Partial plans
already exceeding the best cost or eager safety bound are pruned.

Costs use exact query-length and context-bucket histograms, not token count alone.
Qualified CSV rows include `workload`, `scope=forward`, `samples>=3`, and `parts`.
The planner compares graph P90 plus measured preparation/merge overhead with
0.9 times eager P50. Logits/sampling are unchanged on both paths and excluded
from both sides. It uses bounded search (at most three parts, two above 40 layers)
and rejects missing evidence. Search is limited to 1 ms or 5% of estimated eager
time, whichever is smaller, and its measured CPU time is charged to the plan.
Existing unqualified GPU CSVs remain readable for
shadow analysis but cannot authorize runtime padding/splitting.

Offline calibration, on an otherwise free GPU, can be run with:

```bash
PYTHONPATH="$PWD" python tools/profile_cudagraph_adaptation.py \
  --model /data/wenbiao/models/Qwen3-4B --gpu 0
```

Use `--prompts` with a representative JSON string list and set `--max-model-len`,
`--max-num-batched-tokens`, `--max-num-seqs`, and `--capture-sizes` to match the
deployment. Defaults are a small correctness/calibration workload, not ShareGPT
performance evidence. Separate processes measure eager, exercised splits, and
padding, then test the unforced policy and compare greedy outputs. A separate
equal-partition eager control isolates graph replay from batch-shape arithmetic.
The FP16 calibration requires identical greedy tokens, selected-token logprob
absolute error below `--max-logprob-error` (default 0.01) across shapes, and below
1e-5 between split eager and split replay. Cross-shape FP16 arithmetic can exceed
the strict default even for ordinary padding. Investigate the eager/padding
controls before explicitly selecting another tolerance; the tool never relaxes
it automatically. This is bounded numerical validation, not bitwise equivalence to
the original batch or a proof for arbitrary models and inputs. Exercised
plans use synthetic admission costs only to collect measurements; those costs
are never exported. Only measured costs and their compatibility/checksum
manifest survive under the hotspot asset directory; temporary engines/logs are
cleaned up. Rejection of all measured split candidates is a valid outcome.
The profile also retains directly measured `split_forward` elapsed time for
diagnosis. CUDA-event elapsed time includes GPU idle gaps during CPU dispatch;
it is not a measurement of SM occupancy.

### Single-Launch Multi-Stream Composites

With a qualified `composite` cost row, partitioning prefers a private full-forward
fork/join graph. A single `replay()` submits independent sub-batch streams and
their output copies, followed by the unchanged unified logits/sampling path.
The cost is measured for the entire composite, including staging and merging;
it is never approximated by the maximum or sum of isolated small-graph costs.
The row includes each part's size and workload, and requires the same runtime
manifest as other graph costs. Unknown costs retain eager/sequential fallback.

This is additional formal recording of compiled sub-batch forwards (including
attention), not free concatenation of existing graph handles. Only already
compiled/published partition shapes qualify. The first request snapshots private
inputs/metadata and queues recording on the existing capture worker, then runs
whole eager. Recording does not replay or warm by repeating live KV writes.
Other requests remain eager until atomic publication. Both below-threshold and
whole-shape-capture-pending requests can replay a ready composite; the original
shape's threshold-triggering eager warmup always takes precedence.

Each composite owns input/output buffers, streams and a graph pool. Full
attention metadata scalar/tensor contracts are checked, with bounded context
length buckets and model/KV binding validation. At most four plans reside per
runner; capacity exhaustion, capture failure or incompatibility declines before
KV writes. No whole-batch retry is allowed after replay submission. Composite
and ordinary background captures share one serial queue; neither invokes a
device-wide synchronization. Background CPU recording still competes with eager
submission: first-use latency can regress and is reported separately from warm
replay performance. Steady-state cost qualification is not an amortization proof.

`tools/verify_online_composite.py --model MODEL --gpu ID --batch 7 --publish`
calibrates and checks distinct same-length requests, then publishes measured
costs. `--batch 23` adds another partition workload to the compatible profile.
`--policy-only --costs PATH` checks a fresh process without synthetic prices;
`--profile-window` brackets that policy for Nsight. The tool reports cold-build,
warm-forward and generation timings separately. `tools/verify_composite_phases.py`
is forced-admission prefill/mixed correctness coverage, not a performance test.

## Porting

Copy/install only `igniteserve_startup` for another engine; it imports neither
vLLM nor Torch. Supply a reviewed contract, a pure
`build_action_registry(host)` adapter, actual initial inputs, compatibility
facts and a source root to `prepare()`, then call the returned DAG through
`ResourceAwareDAGExecutor.run()`. Keep engine cache validators and cleanup in
the adapter/entry. Constructors which hide all work require deferred hooks;
an external wrapper cannot safely expose their internal parallelism by itself.
The synthetic engine acceptance tests exercise this independent path.
