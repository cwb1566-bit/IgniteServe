# Persistent Engine Shell and Fine-Grained I/O DAG

Date: 2026-08-22

## Verified Baseline

The current runtime does not yet provide cross-model EngineCore reuse.

- The PP=2 same-executor probe measured the first distributed bootstrap at
  1.8326 seconds and an idempotent re-entry of already initialized executor
  stages at 0.000651 seconds.
- This proves that the worker processes, RPC queues, CUDA contexts, and
  compatible distributed process groups can be reused inside one executor.
- It does not prove that a newly created EngineCore can adopt those resources.
  `EngineCore.shutdown()` still terminates the executor workers.
- Full-FX compiler-process reuse was validated separately. A two-worker pool
  needed 9.294932 seconds for first creation and prewarm, while a second
  `prewarm_workers()` call in the same owner process took 9 microseconds.
- The main startup path still closes the Full-FX pool after compilation. The
  persistent executor probe and persistent compiler-pool probe have not yet
  been combined into one production lifecycle.

## Resource Tiers

The implementation must distinguish four ownership tiers.

| Tier | Retained resources | Compatibility requirement |
|---|---|---|
| Service | compiler daemon, model/config manifests, page cache | compiler/runtime ABI |
| Topology shell | worker processes, RPC queues, CUDA contexts, NCCL/model-parallel groups | identical device mapping and TP/PP topology |
| Model skeleton | module objects, parameter storage, compiled graph and optionally CUDA graphs | identical structural signature and tensor addresses |
| Request runtime | KV contents, scheduler queues, requests, sampling state, LoRA state | never retained across model rebinding |

The structural signature must cover model class, architecture/config fields,
parameter names/shapes/dtypes, quantization, attention/KV layout, parallel
partition, compiler settings, and CUDA graph input-buffer layout.

## Single-GPU Rebinding

The single-GPU implementation should keep one EngineCore and its
`UniProcExecutor` alive. A service-side broker drives this state machine:

```text
ACTIVE -> QUIESCING -> SHELL -> MODEL_LOADING -> RUNTIME_BUILD -> ACTIVE
```

Two rebind modes are required.

1. Structure-compatible weight swap:
   - drain requests and synchronize CUDA;
   - clear scheduler, KV contents, prefix cache, LoRA and sampling state;
   - retain the GPUModelRunner, module objects, parameter storage, compiled
     graph, and CUDA graphs;
   - copy the new checkpoint into existing parameter storage;
   - rebuild tokenizer/frontend state and validate the structural signature;
   - perform a correctness probe before admission.
2. General model rebind:
   - retain the process, CUDA context, custom-op imports, and compiler daemon;
   - release model, KV allocations, CUDA graphs, piecewise backend,
     `static_forward_context`, and allocator tags;
   - bind the new VllmConfig and construct a new GPUModelRunner;
   - execute the normal model/runtime DAG from `init_model` onward.

The first mode is applicable to checkpoints/fine-tunes with exactly the same
architecture. Models with different layer counts, hidden sizes, vocabulary
partitions, RoPE settings, or quantization must use general rebind.

## TP/PP Rebinding

The multi-GPU implementation uses the same state machine, but the persistent
EngineCore broker must own the existing `MultiprocExecutor`. A new EngineCore
cannot safely adopt the old message queues with the current API.

The transition protocol is collective:

1. stop request admission and drain every pipeline batch/future;
2. broadcast a monotonically increasing rebind epoch;
3. synchronize all ranks and export final telemetry;
4. clear rank-local request/model runtime state;
5. validate a topology-shell signature containing world size, TP, PP, rank to
   device mapping, NCCL settings, dtype capabilities, and worker class;
6. atomically broadcast the new model configuration;
7. perform structure-compatible weight swap or rank-local model-runner rebuild;
8. load topology-sharded weights and rebuild KV/graph/runtime state;
9. run a distributed correctness probe, barrier, and reopen admission.

Changing TP, PP, worker count, device mapping, or incompatible communicator
settings invalidates the topology shell and requires a full worker restart.

## Current I/O DAG

The executable startup path currently exposes I/O at coarse granularity.

1. `AsyncEngineArgs.from_cli_args()` starts a lightweight external sequential
   page-cache prewarm before EngineCore construction.
2. Environment variables transfer the prewarm PID/model identity across the
   frontend and separately spawned EngineCore.
3. `stop_weight_prewarm` terminates only unfinished prewarm before in-process
   weight loading to avoid duplicate disk traffic.
4. `load_weights` is one DAG node with `disk_io` and `worker_rpc` resources.
5. The topology-aware sharded loader lets each TP rank or PP stage read only
   its own checkpoint files.
6. The optional rank-local path overlaps weight loading with compile-only work.

The `static_resource_probe` reads startup-resource metadata. It is not a model
weight prefetch stage. The current DAG does not yet schedule individual files,
tensors, layers, pinned-memory buffers, or H2D copies.

## Proposed Fine-Grained I/O DAG

Replace the monolithic load node with a bounded streaming pipeline:

```text
checkpoint_manifest
  -> topology_shard_plan
  -> extent_prefetch[i]
  -> file_open_or_mmap[i]
  -> tensor_decode[i]
  -> pinned_stage[i]
  -> async_h2d[i]
  -> post_load_transform[i]
  -> layer_ready[i]
  -> weights_finalized
```

Implementation requirements:

- use a bounded two- or three-slot pinned-memory ring rather than pinning the
  whole checkpoint;
- issue H2D on a dedicated copy stream and publish CUDA events consumed by the
  default stream;
- model disk, network, CPU decode, PCIe copy, and GPU-memory allocation as
  separate DAG resources;
- use rank/stage-specific file lists and avoid all TP ranks reading the full
  checkpoint;
- for PP, prioritize the predicted straggler stage and balance bytes per stage;
- bind reader and pinned buffers to the CPU NUMA node nearest each GPU/NVMe;
- cancel only overlapping prefetch extents when online loading catches up,
  rather than terminating all future prefetch work;
- permit compilation of a region only after every weight read by that region is
  ready. The existing whole-model compile path must retain its finalized-weight
  dependency.

## Expected Effects and Measurement

- Topology-sharded loading is already verified to reduce weight-loading time by
  48.4-74.4% and Total Init by 21.2-38.3% in TP=2/PP=2 tests, depending on page
  cache state.
- Weight/compile overlap reduces the measured critical path by 6.1-25.8%.
- Streaming I/O can change each shard's serialized `read + copy` cost toward
  `max(read, copy)` when storage and PCIe have independent headroom. Its gain
  must be measured; it must not be inferred from nominal bandwidth.
- Persistent topology shells have about 1.7-1.8 seconds of verified distributed
  bootstrap headroom. Compiler-daemon persistence can additionally remove only
  the compiler bootstrap that remains on the critical path, not the full
  9.294932-second first-pool wall time when that work is already overlapped.
- Structure-compatible weight swap can save model construction, tracing, and
  CUDA graph recapture, but only after pointer/guard correctness is validated.

Required counters include per-file read bytes/time, major page faults, pinned
staging wait, H2D stream occupancy, NVMe/network throughput, per-rank load
completion, PP straggler time, retained GPU memory, and post-rebind output
equivalence.

## Implementation Order

1. Add shell/structural compatibility signatures and a no-request quiescence
   checker.
2. Implement single-GPU general rebind while retaining only process/CUDA/
   compiler resources.
3. Add structure-compatible in-place weight swap and validate compiled/CUDA
   graph reuse.
4. Move ownership into an EngineCore broker and implement collective TP/PP
   rebind epochs for an unchanged topology.
5. Split weight loading into the bounded file/tensor/H2D pipeline and feed its
   observed costs back into the startup DAG scheduler.

## Implemented Experimental Branch (2026-08-23)

Branch: `codex/persistent-engine-shell`.

### 1. General persistent shell rebind

The executor, CUDA context, distributed process groups, message queues, and
Full-FX compiler pool can remain alive while model-owned state is released.
`LLMEngine` and `AsyncLLM` expose controlled rebind calls; multiprocessing
control messages serialize `VllmConfig` explicitly with cloudpickle because
the normal msgpack utility channel cannot encode values such as `torch.dtype`.

General rebind intentionally destroys the old model, KV tensors, captured CUDA
graphs, request state, and scheduler. It then rebuilds all model-owned state.
Topology changes are rejected. The feature is enabled with:

```bash
VLLM_ENABLE_PERSISTENT_ENGINE_SHELL=1
```

Single-GPU Llama-8B to Qwen-14B rebind was verified in frontend/EngineCore
multiprocessing mode. A measured shell reset retained about 0.53 GiB of CUDA
runtime state and took 0.27 seconds. A fresh Qwen-14B comparison showed that
about 1.66 seconds of execution-environment bootstrap is removable. Fresh
model configuration construction must be included separately in end-to-end
admission timing; the rebind API receives an already constructed config.

### 2. Structure-compatible in-place weights

The in-place path is enabled in addition to the general shell with:

```bash
VLLM_PERSISTENT_SHELL_INPLACE_WEIGHTS=1
```

Every rank hashes path-independent, shape- and runtime-defining model fields,
including the normalized Hugging Face config, dtype, quantization, TP/PP
topology, KV geometry, scheduler limits, and compilation/CUDA-graph settings.
All ranks must agree that the new model is compatible. Quantized, LoRA, and
speculative configurations currently fall back to general rebind.

The loader writes the new checkpoint into the existing module. It records the
address, shape, and dtype of every parameter and buffer before loading and
requires the complete state dictionary to have identical storage afterwards.
KV tensors and CUDA graph objects are retained. EngineCore transfers the old
KV block geometry to the new process-local config and reconstructs only
frontend/scheduler request state. Completed PP worker cache entries are removed
with `InputBatch.remove_request()` and `condense()` without reallocating its
persistent buffers.

Verified results for a Llama-3-8B checkpoint path alias are below. The alias
test validates path rebinding and storage/graph correctness; a second semantic
checkpoint with exactly the same runtime structure was not available on this
server, so different-weight equivalence remains a stated test gap.

| Mode | EngineCore rebind | Wall rebind | Retained tensors | Retained KV tensors | Retained CUDA graph shapes | Result |
|---|---:|---:|---:|---:|---:|---|
| GPU=1, eager | 5.946 s | 6.447 s | 195 | 32 | 0 | correct |
| GPU=1, compiled | 5.887 s | 6.370 s | 195 | 32 | 3 | correct replay |
| TP=2, eager | 9.940 s | 10.448 s | 195/rank | 32/rank | 0 | correct |
| PP=2, eager | 9.643 s | 10.130 s | 97/98 | 16/rank | 0 | correct |

For the compiled single-GPU case, rebind reported zero model initialization,
compilation, KV allocation/profile, and CUDA graph capture time. The 5.42-second
weight load dominated the critical path.

TP=2 and PP=2 were also tested with Llama-8B to Qwen-14B. Both configurations
rejected in-place reuse on every rank, fell back to general shell rebind, and
produced correct Qwen output. Wall times were 25.117 seconds for TP=2 and
21.137 seconds for PP=2 in these eager runs.

### 3. Distributed DAG integration

The validated optimized distributed startup uses:

```bash
VLLM_STARTUP_DAG_MODE=execute
VLLM_ENABLE_DISTRIBUTED_STARTUP_DAG=1
```

These switches make `MultiprocExecutor` expose deferred worker construction,
preventing its upstream `init_device()+load_model()` lifecycle from running
before the split EngineCore stages. Without these switches, the experimental
split distributed path can load weights a second time; strict native baseline
tests use `VLLM_DISTRIBUTED_NATIVE_BASELINE=1` instead.

Integrated DAG + in-place rebind + shard-pipeline smoke results:

| Topology | DAG bootstrap | DAG runtime | Total Init | In-place EngineCore rebind | Result |
|---|---:|---:|---:|---:|---|
| TP=2 | 1.695 s | 15.771 s | 17.502 s | 9.612 s | correct |
| PP=2 | 1.949 s | 13.205 s | 15.189 s | 12.040 s | correct |

### 4. File-shard I/O pipeline

The default safetensors iterator now optionally prefetches shard `i+1` on one
bounded background thread while shard `i` is decoded and copied into model
storage. Only host rank 0 prefetches by default; TP/PP ranks share the same OS
page cache. If the real loader reaches the next shard before prefetch finishes,
the thread is cancelled and joined at an 8 MiB chunk boundary to avoid I/O
competition.

```bash
VLLM_ENABLE_WEIGHT_SHARD_PIPELINE=1
# Optional; defaults shown.
VLLM_WEIGHT_SHARD_PREFETCH_LEADER_ONLY=1
VLLM_WEIGHT_SHARD_PREFETCH_CHUNK_MB=8
```

Three alternating cold-page runs used `posix_fadvise(DONTNEED)` only on the
Llama-3-8B safetensors files. This avoids globally dropping server page cache.

| Metric | Pipeline off median | Pipeline on median | Saving |
|---|---:|---:|---:|
| Weight loading | 12.59 s | 11.57 s | 1.02 s (8.1%) |
| EngineCore Total Init | 15.273 s | 14.193 s | 1.080 s (7.1%) |

All three pipeline runs completed one future shard and cancelled two partial
prefetches; 4.48-4.60 GB were read ahead per run. TP=2 and PP=2 correctness
tests showed that only rank 0 issued prefetch logs while every rank completed
collective loading and inference.

This is a file/page-cache pipeline, not yet the complete tensor pipeline from
the preceding proposal. A bounded pinned-memory tensor ring, asynchronous H2D
copy stream, CUDA-event dependency publication, NUMA binding, and layer-ready
compile dependencies remain future work because the current model-specific
`weight_loader` callbacks perform their own copies and do not expose a stable
asynchronous completion contract.

### Source changes and test artifacts

Core source changes:

- `vllm/executor/executor_base.py`: executor config rebinding and collective
  general/in-place rebind APIs.
- `vllm/worker/worker_base.py`: wrapper-side config/model rebind dispatch.
- `vllm/v1/worker/gpu_worker.py`: topology/structure validation, worker-shell
  reset, in-place cleanup, and collective rank results.
- `vllm/v1/worker/gpu_model_runner.py`: in-place checkpoint loading and tensor
  address verification.
- `vllm/v1/engine/core.py`: quiescence checks, KV/scheduler ownership transfer,
  general and in-place EngineCore rebuild paths.
- `vllm/v1/engine/core_client.py`, `llm_engine.py`, `async_llm.py`: synchronous
  and asynchronous frontend control APIs.
- `vllm/compilation/backends.py`: retain the Full-FX pool when the persistent
  shell is enabled.
- `vllm/model_executor/model_loader/weight_utils.py`: next-shard prefetch.

Tests and complete logs are under `A100_0606/persistent_shell/`, including
single-GPU eager/compiled, TP=2, PP=2, cross-architecture fallback,
distributed-DAG integration, and three cold-page A/B runs.
