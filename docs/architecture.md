# Architecture and Source Guide

[Overview](../README.md) | [Deployment and startup](deployment.md)

Read the three paths below in order. Links point to files; function names provide
stable landmarks without a second, quickly stale inventory of every log line.

## Startup Orchestration

```text
EngineCore._initialize_engine_core()
  -> declarative_startup.run()
  -> contract.prepare(declaration, adapter, compatibility facts)
  -> ResourceAwareDAGExecutor.run(dag, context)
  -> adapter actions -> existing executor/worker APIs
```

| Responsibility | Source and reading entry |
|---|---|
| Engine integration / cleanup | [core.py](../vllm/v1/engine/core.py), [declarative_startup.py](../vllm/v1/engine/declarative_startup.py): `run` |
| Online declaration validation | [contract.py](../igniteserve_startup/contract.py): `prepare`, `variant_nodes` |
| State, readiness, budgets, scheduling | [runtime.py](../igniteserve_startup/runtime.py): `StartupContext`, `ResourceAwareDAGExecutor` |
| Reviewed node declarations | [startup_contract.json](../vllm/v1/engine/startup_contract.json) |
| Engine actions, no graph construction | [vllm_startup_adapter.py](../vllm/v1/engine/vllm_startup_adapter.py): `build_action_registry`, `build_runtime_actions` |

`execute` loads the packaged 26-node v3 contract for qualified single-GPU CUDA
startup. It covers bootstrap, model/weights/compilation, KV allocation, precapture
and serving-manager readiness, not API-server binding. The `prewarm` and `standard`
variants express initialization ordering, not user-asserted cache temperature.
`off` uses sequential startup; `shadow` retains legacy planning without execution;
`discover` gathers sequential-path evidence; `execute-generated` retains the older
generated-contract loader. These are not interchangeable contract schemas.

The loader checks finalization evidence, source and adapter hashes, compatibility,
unique state producers, resource demands, action bindings and acyclicity before
execution. An invalid plan falls back before actions. Once an action has started,
errors clean up and propagate; the engine is not initialized a second time.
Adapters are trusted Python code: import and registry construction must be pure.

`dependencies` require completion; `required_state` requires published values.
`put/require` exchange values/readiness, not GPU or disk resources. Resources are
admission budgets, not measured occupancy. A returned Future holds reservations
until completion. Adapters must establish CUDA completion with the appropriate
event/stream boundary before publishing GPU-ready state. Mutable aliases require
reviewed ordering even when a value has already been published.

State/completion events update pending costs and downstream critical-path priority.
Boolean `routes` can change pending actions, costs and resources while preserving
dependencies/outputs. Dispatched work is not restarted. This is a heuristic, not
a guarantee of optimal scheduling. `prepare_startup_assets` stages bounded
JSON/CSV data in the consuming worker (32 files / 16 MiB); consumers revalidate
identity and file changes. It does not eagerly deserialize compiler binaries or
Dynamo code. `ready` releases prepared data; existence is never permission to
skip validation or GPU memory checks.

With weight overlap enabled, `load_weights_and_trigger_compile()` joins loading
and compilation internally; the outer `compile_model` route is an
`already_compiled` readiness marker. It is not independent outer-DAG concurrency
between two full actions. Without overlap, the compile action does real work.
See [gpu_worker.py](../vllm/v1/worker/gpu_worker.py) and
[weight_compile_overlap.py](../vllm/compilation/weight_compile_overlap.py).
No model-name whitelist is used, but compilation mode, loader format, quantization,
LoRA and speculative-state checks still apply. Autotuning/real execution wait for
finalized weights. Custom loaders that replace parameter storage need separate
qualification, not merely the overlap switch.

## Offline Contract Workflow

[build_startup_contract.py](../tools/build_startup_contract.py) contains the
reviewed deferred-API recipe; serving does not invoke it to synthesize a graph.
Source changes require a new candidate and acceptance, not edited hashes:

```bash
PYTHONPATH=. python tools/build_startup_contract.py --output candidate.json
PYTHONPATH=. python tools/validate_startup_contract.py \
  --contract candidate.json --model /path/to/model --gpu 0 --output validation.json
PYTHONPATH=. python -m igniteserve_startup.offline finalize \
  --candidate candidate.json --adapter vllm/v1/engine/vllm_startup_adapter.py \
  --review review.json --validation validation.json --root . \
  --output vllm/v1/engine/startup_contract.json
```

Review every action, state producer, control/alias constraint and affinity against
source. `review.json` must bind `identity` to the candidate, approve every node
in `actions`, set `constraints: approved`, and give a `rationale` (record the
reviewer as provenance). The validator compares greedy token IDs to sequential
startup, checks exactly-once coverage and real scheduled overlap, tests both
variants, cold/warm assets, global reuse and the non-overlap path. The finalizer
requires matching serial/scheduled evidence for all variants. Raw acceptance
logs/caches are temporary; retain the finalized contract and its evidence.

For a new engine without reviewed boundaries,
[startup_offline_analysis.py](../tools/startup_offline_analysis.py) retains
root discovery -> AST/use-def extraction -> node-level evidence -> adapter
review -> serial equivalence -> edge experiments -> finalization. Profile tools
run separately; traces do not prove alias, allocator or C++ lifetime safety.
Opaque actions, state aliases, opaque constraints, inconclusive edge tests and
coarsening proposals require review. Source-instrumentation coverage cannot
substitute for executing the generated adapter. Its older v2 artifacts are not
the packaged v3 recipe. See the tool's `--help` rather than historical experiment
commands with fixed server paths.

## Compilation

```text
VllmBackend: split FX and assemble callable
  -> ParallelCompileManager: coordinate a batch of subgraphs
     -> PiecewiseCompileInterpreter: collect/restore validated inputs
     -> CompilerManager: lookup/load/compile/register each artifact
        -> CompilerInterface: backend API
     -> FullFxCompileBackend: modified Torch process pool
```

| Responsibility | Source |
|---|---|
| Engine FX entry and graph splitting | [backends.py](../vllm/compilation/backends.py) |
| Batch classification, futures and fallback | [parallel_compile.py](../vllm/compilation/parallel_compile.py) |
| Input evidence and restoration | [piecewise_inputs.py](../vllm/compilation/piecewise_inputs.py), [input_metadata.py](../vllm/compilation/input_metadata.py), [input_contract.py](../vllm/compilation/input_contract.py) |
| Artifact identity and reuse | [compile_manager.py](../vllm/compilation/compile_manager.py), [cache_reuse.py](../vllm/compilation/cache_reuse.py) |
| Backend and process transport | [compiler_interface.py](../vllm/compilation/compiler_interface.py), [fx_transport.py](../vllm/compilation/fx_transport.py) |
| Dynamo frontend restoration | [wrapper.py](../vllm/compilation/wrapper.py), [frontend_metadata.py](../vllm/compilation/frontend_metadata.py) |

Stage-separated startup compiles before KV profiling. Validated input metadata
reconstructs FakeTensor inputs in the current symbolic context, preserving
supported shape, stride, layout, alias/offset and constant contracts. Unknown
symbols or unsupported relationships fall back to data-flow reconstruction.
This removes compilation-preparation dependencies, not inference dependencies.
The batch coordinator submits process jobs before awaiting results; cache loading
can overlap compilation. Parent artifact binding remains locked to avoid shared
Triton-module races. Worker prewarm and deferred/fast shutdown stay in the worker
lifecycle, not a second scheduling layer.

Reuse queries instance memory, launch-local cache, then a global semantic index.
Keys include canonical operator IR/constants/code dependencies, guard-relevant
input/output signatures, compiler/GPU context and distributed semantics. Source
indices are provenance, not cross-launch equivalence. Missing output evidence,
unknown dependencies or unsupported weight freezing decline semantic reuse.
Backend guards and current constants are checked even after a descriptor match.
Only successfully loaded artifacts are published; bad handles recompile.

Frontend bundles include `fake_evidence.graphpickle` alongside graph/bytecode
artifacts. Restored node examples and backend inputs share a FakeTensorMode and
ShapeEnv; their contracts are checked before backend use. Unsupported constants,
old/incomplete bundles or incompatibility return to tracing. This is an exact-
configuration cache, not a universal Dynamo guard cache or proof that arbitrary
compatible-looking models can share a frontend.

## CUDA Graphs

Each runner owns a [CUDAGraphManager](../vllm/v1/worker/CudaGraphManager.py).
The runner supplies real forwards, attention/KV metadata, logits and sampling;
the manager coordinates these components:

| Component | Responsibility and source |
|---|---|
| `HotspotAnalyzer` | Shape observations, threshold checks and precapture selection: [HotspotAnalyzer.py](../vllm/v1/worker/HotspotAnalyzer.py) |
| `GraphAdapter` | Exact matching, cost-gated padding and bounded splitting: [cudagraph_adaptation.py](../vllm/v1/worker/cudagraph_adaptation.py) |
| `PiecewiseGraphRegistry` | Complete published shapes and compiler-owned handles: [piecewise_graph_registry.py](../vllm/v1/worker/piecewise_graph_registry.py) |
| `CaptureExecutor` | Precapture, safe-point capture and the shared serial background queue: [concurrent_piecewise_cudagraph.py](../vllm/v1/worker/concurrent_piecewise_cudagraph.py) |
| `CompositeGraphStore` | Private multi-stream plans, buffers, publication and replay: [composite_cudagraph.py](../vllm/v1/worker/composite_cudagraph.py) |
| Piecewise backend | Actual piece warmup/capture/replay: [cuda_piecewise_backend.py](../vllm/compilation/cuda_piecewise_backend.py) |

The runtime path is `manager.select()` -> runner forward ->
`manager.after_warmup()` when requested. A threshold-triggering request must
perform whole-shape eager warmup. Below-threshold and capture-pending requests
may use qualified adaptation. Formal capture uses private state and a dedicated
stream; the registry exposes only completely published graphs. It is not a
mirror that records each foreground kernel as it executes. Without concurrent
runtime support, coordinated safe-point maintenance remains a different path.
Decode, prefill and mixed triggers require capability checks; single-GPU,
text-only piecewise support does not imply full-graph, LoRA or distributed support.

Exact matches use normal replay. Padding requires a compatible measured cost
advantage and a default maximum ratio of 0.2, reduced under memory pressure.
Splitting keeps whole contiguous requests and original KV slots, bounds the
number of parts, and combines outputs before the unchanged unified logits/
sampling path. Unsupported attention/state/parallel configurations decline
before KV writes. Missing parts or unqualified costs return to eager; an error
after KV writes must not retry the whole batch.

Qualified cost rows include workload/query-context information, forward scope,
sample count and partition layout. Sequential plans charge preparation/merge
and bounded planning time against a conservative eager budget. A ready
multi-stream composite uses a measured cost for the entire composite, not the
maximum of independent small-graph timings. Its single replay submits a fork/
join graph with private input/output buffers, streams and a graph pool.
Construction is additional recording of compiled forwards, not free composition
of existing handles. It shares the background capture queue; requests stay eager
until publication. Cold construction can still increase latency, so warm replay
gains do not establish first-use or amortized gains. No suitable cost profile
means no evidence-based splitting. Shutdown/rebind drains capture and invalidates
model-bound graphs and plans.

## Asset Repository

[startup_resources.py](../vllm/compilation/startup_resources.py) resolves roots
and switches; [startup_assets.py](../vllm/compilation/startup_assets.py) provides
identity/checksum records, atomic writes, locking and bounded preparation.
No database or additional resident manager is required.

| Directory under the selected root | Producer/consumer and validation |
|---|---|
| `model_info/<key>.json` | Model registry; class/source compatibility. |
| `kv_profile/<key>.json` | [kv_profile_cache.py](../vllm/v1/worker/kv_profile_cache.py); model/config/hardware plus current weights/free-memory budget. |
| `best_config/<key>.json` | [best_config_cache.py](../vllm/v1/worker/best_config_cache.py); kernel, inputs, candidates, device and launcher match. |
| `input_metadata/<key>.json` | Input serializer/collector; current graph/input contracts and restored evidence. |
| `transformed_code/<key>/` | [transformed_code_cache.py](../vllm/compilation/transformed_code_cache.py); source/code/file hashes; not by itself a tracing bypass. |
| `dynamo_frontend/<key>/` | Frontend wrapper; complete bundle and current input/runtime contracts. |
| `compile/<launch-key>/` | Backend rank/prefix, Inductor and Triton artifacts; guards and backend/ABI compatibility. |
| `compile/subgraphs/<semantic-key>.json` | Global descriptors with relative handles and payload checksums; referenced launch artifacts must remain present. |
| `hotspots/<key>/` | Frozen hits/costs; workload selection, runtime identity and file integrity. |
| `startup_contract/<key>/` | Explicitly published trusted contracts, never automatic approval or selection. Default v3 files remain packaged in the source tree. |
| `startup_cost/<key>.json` | [startup_orchestrator.py](../vllm/v1/engine/startup_orchestrator.py); state/source-keyed estimates, not correctness evidence. |
| `runs/<time_ns>_<pid>/` | Runtime observations separate from frozen reusable assets. |

Directories appear on demand. The common flow is current identity -> lookup ->
schema/checksum/identity checks -> stage-specific checks -> load or rebuild/save.
Preparation does not bypass consumption-time validation. Global loading does not
rewrite process-wide cache directories; a backend may rematerialize wrappers or
kernels in the consumer cache. Migration preserves validation, not universal
binary portability. Storage configuration and transfer commands are in
[deployment](deployment.md#assets-and-hotspots).

## Porting and Optional Paths

`igniteserve_startup` is stdlib-only and imports neither vLLM nor Torch. Another
engine supplies a reviewed contract, a pure `build_action_registry(host)`, actual
initial state, compatibility facts and a source root to `prepare()`, then executes
the returned graph. Deferred hooks are necessary when a constructor hides work;
an external wrapper cannot safely infer internal parallelism. Synthetic tests
cover this portable boundary. Compilation and CUDA components remain Torch/FX
and engine-buffer dependent and need their own adapters and validation.

Legacy shadow/rebind builders and TP/PP paths remain for compatibility, not as
proof that the packaged cold-start contract supports them. Optional
[multi_model_server.py](../multi_model_server.py) implements on-demand models and
persistent-shell routing; consult `--help`. A general rebind drains requests and
capture, keeps fixed-topology process/runtime infrastructure, and reconstructs
model-owned modules, KV state, graphs and request managers. It must not reuse old
model buffers, and changing topology requires a new shell. The single-model
master switch does not automatically enable persistent-shell service.
