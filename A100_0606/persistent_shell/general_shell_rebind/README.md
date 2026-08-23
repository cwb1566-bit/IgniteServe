# General Persistent Engine Shell Rebind

## 1. Target semantics

This implementation is a reusable inference-engine shell, not a warm model
skeleton. A model switch retains only topology-level infrastructure:

- EngineCore, executor, worker processes, RPC/IPC channels and shared memory.
- CUDA contexts and fixed-topology TP/PP process groups/NCCL communicators.
- Persistent Full-FX compiler worker processes and generic startup-DAG logic.

Every switch reconstructs model-owned state:

- `VllmConfig`, model modules and weights.
- In-memory Dynamo/Inductor callables and piecewise CUDA graphs.
- KV-cache profiling/allocation and attention bindings.
- Startup CUDA graphs, static graph buffers, scheduler and request state.

The shell topology is immutable. A TP=2 shell can switch models using TP=2,
but changing TP/PP size requires a new shell.

## 2. Rebind path

1. Drain all active requests and block new model switches.
2. Start asynchronous page-cache prewarm for the target model.
3. Tear down old scheduler/request/runtime state.
4. Execute a collective worker reset on every rank:
   - export old CUDA-graph hotspot data;
   - unfreeze Python GC state left by CUDA graph capture;
   - reset and release piecewise CUDA graphs and compiled callables;
   - clear model-specific Dynamo/Inductor caches in the main worker and
     persistent Full-FX compiler workers;
   - break vLLM `Parameter.weight_loader` bound-method cycles;
   - detach old parameters and KV bindings without deleting globally cached
     rotary-embedding buffers;
   - return unused CUDA allocations and construct a fresh GPUModelRunner.
5. Run the startup DAG for the target model:
   - initialize model;
   - stop external prewarm immediately before the real loader;
   - load weights with next-shard read-ahead;
   - overlap value-independent structural compile with verified in-place
     dense loaders, then join before autotune or real execution;
   - profile or restore KV capacity, allocate a new KV cache;
   - capture target CUDA graphs;
   - recreate scheduler and batch-queue state.
6. Load the target tokenizer concurrently with EngineCore rebind, then publish
   the switched engine.

For unquantized Llama/Qwen safetensors loaders, model construction fixes the
parameter shape, dtype, device and object identity before loading. The loader
copies values into that storage, while S4 compile-only suppresses normal kernel
launches. Tracing, lowering and artifact materialization can therefore overlap
weight I/O. Online autotune waits at the `weights_ready` boundary, and KV
profiling, CUDA graph capture and inference remain strictly after it. Loaders
that quantize, repack, replace parameters, use LoRA/speculative state, or have
an unsupported architecture fall back to the sequential dependency.

## 3. Main implementation files

- `vllm/v1/engine/core.py`: general rebind orchestration and startup-DAG handoff.
- `vllm/v1/worker/gpu_worker.py`: per-rank teardown, memory release and runner
  reconstruction.
- `vllm/v1/engine/core_client.py`, `vllm/executor/executor_base.py`,
  `vllm/worker/worker_base.py`: serialized control message and collective RPC.
- `vllm/v1/engine/llm_engine.py`, `vllm/v1/engine/async_llm.py`: tokenizer/core
  overlap and frontend publication/rollback.
- `vllm/v1/engine/weight_pagecache.py`: validated runtime environment handoff
  for the external prewarm process and hotspot directory.
- `vllm/model_executor/model_loader/weight_utils.py`: topology-aware next-shard
  prefetch (`shared_leader`, rank-local shards, or single process).
- `vllm/v1/worker/LazyFreezeManager.py`: `gc.unfreeze()` at model teardown.
- `multi_model_server.py`: one persistent shell, request draining and model
  switching instead of engine destruction/recreation.
- `torch/_inductor/compile_fx_full_subproc.py` in IgniteServe-Pytorch: reset
  model-specific caches on every persistent compiler worker without changing
  worker PIDs.

## 4. Important correctness findings

Two failures discovered during real GPU validation were fixed:

1. vLLM stores bound `weight_loader` methods on Parameters. This creates
   `Module -> Parameter -> bound method -> Module` cycles and retained about
   26.34 GiB of Llama3-8B model/KV memory after the top model object died.
   Explicit cycle breaking and parameter detachment reduce post-reset active
   CUDA allocation to about 0.03-0.07 GiB.
2. Clearing every module buffer corrupted globally cached
   `RotaryEmbedding.cos_sin_cache` and broke A->B->A. Teardown now preserves
   reusable buffers and only removes model-owned weights/KV state.

## 5. GPU validation

All tests used A100 GPUs, FP16, max model length 512 and CUDA graph sizes
`[1, 2, 4]`. Each switch performed generation after reconstruction.

### Single GPU, Llama3-8B configuration switch

| Cache state | A -> B | B -> A | Result |
|---|---:|---:|---|
| warm disk compile cache | 16.948 s | 16.750 s | pass |
| cold weights + cold compile cache | 22.403 s | 23.046 s | pass |

The worker PID stayed `1087697` in the warm run and `1091629` in the cold
run. Runner/model IDs, KV pointers and CUDA graph objects changed on every
switch.

### I/O ablation for cold switches

| Path | A -> B load | B -> A load | A -> B total | B -> A total |
|---|---:|---:|---:|---:|
| no page-cache prewarm/shard pipeline | 10.648 s | 10.715 s | 24.113 s | 24.858 s |
| optimized I/O | 8.286 s | 8.540 s | 22.403 s | 23.046 s |

The I/O path saves 2.362 s (22.2%) and 2.175 s (20.3%) in weight loading,
and about 1.7-1.8 s (7.1-7.3%) on the complete switch path in this run.

### Cross-architecture switch

Llama3-8B -> Qwen2.5-14B -> Llama3-8B passed in one worker process
(`PID 1100815`). The runtime changed from 32 to 48 KV tensors and from 33 to
49 piecewise backends. The Qwen switch was a cold compile/KV-profile case and
took 45.683 s; the cached return to Llama took 17.193 s.

### Fixed distributed topologies

| Topology | A -> B | B -> A | Stable worker PIDs | Result |
|---|---:|---:|---|---|
| TP=2 | 32.861 s | 22.264 s | `1103289`, `1103290` | pass |
| PP=2 | 23.134 s | 17.095 s | `1107168`, `1107169` | pass |

The first switch in these fresh cache roots was not a fully seeded warm-cache
measurement; the table is a correctness and lifecycle result, not a formal
TP/PP performance comparison.

### Weight-load/structural-compile dependency validation

The former unconditional `load_weights -> compile` edge was too conservative.
With a verified dense in-place loader, the edge now joins only before autotune
or real execution:

| Switch | Weight load | Structural compile | Overlapped wall | Result |
|---|---:|---:|---:|---|
| Llama3-8B -> Llama3-8B variant | 8.851 s | 9.922 s | 10.258 s | pass |
| Llama3-8B -> Qwen2.5-14B | 16.601 s | 15.960 s | 16.602 s | pass |

The first row saves 8.515 s versus the sum of the two measured tasks. A forced
Qwen best-config miss reached online autotune before cold weight loading had
finished; the first tuner waited 7.648 s at `weights_ready`, then six tuners
benchmarked and generation succeeded. This confirms that tracing/lowering and
materialization are weight-value independent for these loaders, while autotune
and every later real-execution phase retain the finalized-weight dependency.

## 6. Tests and logs

Validation entrypoints:

- `A100_0606/persistent_shell/benchmark_general_shell_switch.py`
- `A100_0606/persistent_shell/test_frontend_rebind_overlap.py`
- `A100_0606/persistent_shell/test_persistent_shell_routing.py`

Key result directories:

- `general_shell_rebind/gpu_smoke_v15/`: single-GPU warm A->B->A.
- `general_shell_rebind/gpu_smoke_v16/`: cold weights/compile cache.
- `general_shell_rebind/gpu_io_baseline_v17/`: I/O optimization disabled.
- `general_shell_rebind/cross_arch_v18/`: Llama3-8B/Qwen2.5-14B.
- `general_shell_rebind/tp2_v19/`: TP=2.
- `general_shell_rebind/pp2_v20/`: PP=2.
- `weight_compile_overlap_probe_v21/`: same-architecture overlap validation.
- `weight_compile_overlap_crossarch_v22/`: cross-architecture overlap.
- `weight_compile_wait_probe_v24/`: forced autotune wait-boundary validation.
- `weight_compile_overlap_capability_v25/`: capability-gated regression.

Enable the service path with:

```bash
export VLLM_ENABLE_PERSISTENT_ENGINE_SHELL=1
export VLLM_PERSISTENT_SHELL_REBIND_MODE=general
export VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1
export VLLM_ENABLE_WEIGHT_SHARD_PIPELINE=1
export VLLM_PERSISTENT_SHELL_CLEAR_INMEM_COMPILE=1
export VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1
export VLLM_PERSISTENT_SHELL_ALLOW_WEIGHT_COMPILE_OVERLAP=1
```

The existing S4/startup-DAG/parallel-compile variables remain applicable.

## 7. Current limits

- Requests must drain before a model switch; concurrent multi-model residency
  is not implemented by this one-shell path.
- TP/PP topology, world size and device mapping cannot change in place.
- General rebind intentionally does not reuse old in-memory model/KV/CUDA
  graph objects. Disk artifacts and semantic startup resources can be reused.
- The direct EngineCore path has real GPU A->B->A coverage. The
  `multi_model_server.py` routing/drain logic has CPU unit coverage; a long
  HTTP concurrency stress test remains useful before production deployment.
- A failure after worker reset is not yet a fully transactional backend
  rollback; the frontend tokenizer path does perform rollback when possible.
