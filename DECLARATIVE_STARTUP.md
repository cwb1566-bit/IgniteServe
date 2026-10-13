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

Malformed/incompatible contracts fall back before any action. After an action
starts, failures clean up the engine and propagate; initialization is not
silently run twice. Adapter import and registry construction must be pure.

## Offline review and acceptance

The existing `tools/startup_offline_analysis.py` retains root tracing,
AST/use-def extraction, candidate adapters and opaque review. The portable
`python -m igniteserve_startup.offline analyze` additionally exposes conservative
source regions without importing an engine. Neither claims to infer arbitrary
C++/allocator/alias semantics. This vLLM recipe uses reviewed deferred executor
boundaries, not an assertion that all 25 boundaries were automatically inferred.

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

## Porting

Copy/install only `igniteserve_startup` for another engine; it imports neither
vLLM nor Torch. Supply a reviewed contract, a pure
`build_action_registry(host)` adapter, actual initial inputs, compatibility
facts and a source root to `prepare()`, then call the returned DAG through
`ResourceAwareDAGExecutor.run()`. Keep engine cache validators and cleanup in
the adapter/entry. Constructors which hide all work require deferred hooks;
an external wrapper cannot safely expose their internal parallelism by itself.
The synthetic engine acceptance tests exercise this independent path.
