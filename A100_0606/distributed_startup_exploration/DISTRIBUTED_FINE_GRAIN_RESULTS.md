# TP=2 / PP=2 Fine-Grained Cold-Start Exploration

Date: 2026-08-22

## Scope

- Model: Meta-Llama-3-8B-Instruct, FP16, max model length 2048.
- Hardware: two A100 PCIe 40 GB GPUs (no NVLink), local NVMe checkpoint.
- Base path: distributed startup DAG + S4 compile path + adaptive CUDA graph
  pre-capture.
- Every formal point was run three times in alternating order; tables report
  medians.

## Accepted Optimizations

### 1. Fast Full-FX compiler process-group shutdown

The PyTorch compile supervisor and its children are placed in one process
group. After every submitted future is durable, the optional fast path
terminates the idle group rather than waiting for Python interpreter teardown.

- Switch: `TORCHINDUCTOR_FULL_FX_FAST_SHUTDOWN=1`
- Eliminates the 0.2-1.4 s compiler shutdown wait, with the largest gain in
  compile-cache-warm startup.

### 2. Asynchronous compiler-identity prewarm

`CacheBase.get_system()` and the Torch compiler key are computed in a daemon
thread after distributed CUDA initialization and overlap model initialization
and weight loading. The exact hash inputs and cache semantics are unchanged.

- Switch: `VLLM_ENABLE_ASYNC_COMPILER_HASH_PREWARM=1`
- D6 -> D8 `Total Init` reduction: 1.9-4.8% on TP=2 and 1.8-8.8% on PP=2,
  depending on cache state.

### 3. Topology-aware sharded checkpoints for V1

The sharded-state filename formatter now accepts `global_rank`, `tp_rank`, and
`pp_rank`; legacy `rank` remains a TP-rank alias. This prevents PP stages from
overwriting each other and lets every TP rank or PP stage read only its own
state dict. The V1 loader is enabled experimentally by
`VLLM_ENABLE_V1_SHARDED_STATE=1`.

Checkpoint pattern used by the experiment:

```text
model-global-rank-{global_rank}-part-{part}.safetensors
```

| Topology | Page cache | Standard load | Sharded load | Load reduction | Total Init reduction |
|---|---:|---:|---:|---:|---:|
| TP=2 | cold | 11.5409 s | 5.7610 s | 50.1% | 5.3699 s (21.2%) |
| TP=2 | warm | 8.9673 s | 2.4149 s | 73.1% | 6.4529 s (27.8%) |
| PP=2 | cold | 11.5903 s | 5.9820 s | 48.4% | 5.1661 s (25.9%) |
| PP=2 | warm | 9.4670 s | 2.4197 s | 74.4% | 6.7677 s (38.3%) |

### 4. Rank-local weight-loading/compile-only overlap

One collective RPC is sent to each rank. Inside that worker, one thread loads
final weight values while another performs S4 Dynamo tracing and compilation
against the already allocated parameter objects. This avoids concurrent
collective RPC streams. The compiler thread and loader thread explicitly select
the worker's CUDA device. If an offline best-config entry is missing, online
autotuning waits for finalized weights before benchmarking.

- Switch: `VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1`
- Applies only to compile-only modes; autotune-first/native modes fall back to
  sequential execution.

| Topology | State | Sequential Total Init | Overlapped Total Init | Reduction |
|---|---|---:|---:|---:|
| TP=2 | cold model / cold compile | 25.2146 s | 20.9458 s | 4.2688 s (16.9%) |
| TP=2 | cold model / warm compile | 19.8913 s | 14.7602 s | 5.1311 s (25.8%) |
| TP=2 | warm model / warm compile | 16.4419 s | 14.8314 s | 1.6105 s (9.8%) |
| PP=2 | cold model / cold compile | 19.8931 s | 18.6869 s | 1.2062 s (6.1%) |
| PP=2 | cold model / warm compile | 14.3901 s | 10.6788 s | 3.7113 s (25.8%) |
| PP=2 | warm model / warm compile | 11.0382 s | 8.8662 s | 2.1720 s (19.7%) |

The individual stages become slower under contention, but the end-to-end
critical path is shorter. PP=2 cold compilation benefits least because weight
I/O delays completion of the asynchronous Full-FX worker import/CUDA prewarm.

### 5. Two-phase Full-FX worker prewarm

The remaining Full-FX bootstrap was split into two dependency-aware tasks.
Compiler/framework imports are submitted immediately after the rank worker is
constructed, before CUDA and NCCL initialization. After the rank selects its
device, CUDA context prewarm is queued on the same compiler subprocesses.
Materialization waits for both sets of futures before submitting compile jobs.

- Switch: `VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM=1`
- PyTorch API: `FullFxCompilePool.prewarm_cuda_workers(device_index)` and
  `FullFxCompilePool.all_prewarm_futures()`.
- Default behavior is unchanged when the switch is absent.

Three-run medians for cold model pages and cold compile caches:

| Topology | Path | Full-FX wait | Weight/compile overlap | Total Init |
|---|---|---:|---:|---:|
| PP=2 | previous ordering | 4.3164 s | 15.1618 s | 19.2213 s |
| PP=2 | CUDA-before-NCCL only | 4.0890 s | 14.9654 s | 18.6712 s |
| PP=2 | two-phase prewarm | 3.4408 s | 13.9334 s | 17.7271 s |
| TP=2 | previous ordering | 0.2057 s | 16.8944 s | 20.7231 s |
| TP=2 | two-phase prewarm | 0.0000 s | 16.5020 s | 20.6014 s |

The two-phase path reduces PP=2 `Total Init` by 1.4942 s (7.77%) and
TP=2 by 0.1217 s (0.59%). PP benefits more because compiler bootstrap remains
on its cold critical path; TP already overlaps almost all of that work.

## Cumulative Result Relative to the Previous D3 Path

| Topology | State | D3 Total Init | Current Total Init | Reduction |
|---|---|---:|---:|---:|
| TP=2 | cold model / cold compile | 33.5839 s | 20.6014 s | 38.7% |
| TP=2 | cold model / warm compile | 28.0953 s | 14.7602 s | 47.5% |
| TP=2 | warm model / warm compile | 24.8311 s | 14.8314 s | 40.3% |
| PP=2 | cold model / cold compile | 29.4559 s | 17.7271 s | 39.8% |
| PP=2 | cold model / warm compile | 22.8439 s | 10.6788 s | 53.2% |
| PP=2 | warm model / warm compile | 21.1953 s | 8.8662 s | 58.2% |

## Rejected or Deferred Paths

- Cross-rank shared Inductor/Triton backend root: TP improved below 1%, while PP
  regressed about 1.6-1.7% from locks and rank-specific FX keys.
- CPU-only compiler worker CUDA prewarm: no gain; PP cold compilation worsened.
- Persistent source-code hash cache: source hashing was about 1 ms. The actual
  first-call cost came from compiler/system identity and is handled by async
  prewarm.
- Materialization-after-weight gate: no gain. PP cold median became slightly
  worse (18.99 s versus 18.69 s for direct overlap).
- Transformed-code artifacts do not skip Dynamo tracing. They avoid expensive
  depyf decompilation/artifact regeneration; the startup DAG description was
  corrected accordingly.
- Naive regional compilation of each `LlamaDecoderLayer` is not safe or
  beneficial. A 32-layer synthetic PyTorch probe needed only two backend calls,
  but the real vLLM model recompiles layers because every attention module has
  a static `Attention.layer_name` used to select its KV cache and metadata.
  Sharing one `VllmBackend` violates its one-graph contract; independent
  backends either collide in graph-index metadata or compile every layer.
  Namespaced trials spent 14-19 s lowering each region and did not complete a
  correct startup. All regional source changes were removed.

## Remaining-Space Experiments

### Regional decoder compilation

Warm-cache Dynamo tracing remains approximately 7.3 s on TP=2 and 3.5 s per
PP=2 stage, and scales almost linearly with the number of decoder layers seen by
that rank. A correct regional design must split a decoder layer into:

1. an attention pre-region with explicitly lifted weights;
2. a layer-specific opaque attention call carrying a stable layer handle; and
3. a shared attention post/MLP region with explicitly lifted weights.

The cache/guard signature must include placeholder schemas, tensor metadata,
constants, and the layer-handle binding. Merely wrapping the existing layer in
`Module.compile()` cannot remove the static attention identity guard.

### Service-owned Full-FX compiler daemon

A live-pool probe measured the first two-worker pool prewarm at 9.294932 s,
including spawn/bootstrap; the child import/CUDA tasks themselves took about
3.51 s. A second `prewarm_workers()` call in the same parent took 9 microseconds,
demonstrating that the runtime is fully reusable. The two resident helper
processes consumed 418 MiB each (836 MiB for one two-worker pool).

The current pool is process-local: every rank worker owns its pipes and destroys
them at engine shutdown. Cross-engine reuse therefore requires a service-owned
daemon with a Unix-socket/RPC protocol for serialized FX payloads, per-request
compiler/cache environments, device identity, concurrent futures, and artifact
responses. Keeping the current parent-local object alive cannot transfer it to
a new EngineCore.

### Persistent distributed worker shell

Current bootstrap medians are about 1.75 s for TP=2 and 1.69 s for PP=2.
An experimental same-executor replay measured PP=2 initial bootstrap at
1.8326 s and idempotent shell reentry at 0.000651 s. Thus the low-level CUDA,
NCCL, worker, and RPC state is reusable and has roughly 1.7-1.8 s of headroom.

This is not equivalent to `sleep(level=2)`, which retains the ModelRunner and
compiled runtime. A correct cross-model shell must keep the worker processes,
CUDA contexts, compatible NCCL groups, and compiler daemon while explicitly
releasing/rebinding model, KV-cache, CUDA-graph, scheduler, allocator-tag,
`static_forward_context`, and piecewise-backend state. It also needs a
service-owned EngineCore broker because the current `EngineCore.shutdown()`
terminates the workers and a new EngineCore cannot adopt the old message queues.
The reentry instrumentation was removed after measurement; no incomplete shell
handoff code remains in the runtime.

## Remaining Opportunities

1. Implement the service-owned Full-FX daemon protocol. The reuse benefit is
   validated, while the remaining work is ownership, IPC, and isolation.
2. Implement attention-aware regional frames rather than per-layer wrappers.
   This is the only path that can remove the remaining linear Dynamo trace.
3. Move EngineCore ownership into a persistent service broker and add an
   explicit model-state reset/rebind protocol for compatible TP/PP shells.
4. Stage-specific CUDA graph policies have limited remaining startup headroom:
   adaptive pre-capture is already about 0.6-0.9 s for three shapes. Every TP
   rank and PP stage still needs its local graph; graphs cannot be copied across
   GPUs.

## Artifacts

- Topology checkpoint comparison:
  - `A100_0606/distributed_startup_finegrain_sharded_v1_tp2/`
  - `A100_0606/distributed_startup_finegrain_sharded_v1_pp2/`
- Weight/compile overlap formal results:
  - `A100_0606/distributed_startup_finegrain_weight_compile_overlap_tp2/`
  - `A100_0606/distributed_startup_finegrain_weight_compile_overlap_pp2/`
- PP materialization-gate rejection:
  - `A100_0606/distributed_startup_finegrain_weight_compile_gate_pp2_cold/`
- Two-phase Full-FX formal results:
  - `A100_0606/distributed_startup_two_phase_fullfx_pp2_formal/`
  - `A100_0606/distributed_startup_two_phase_fullfx_tp2_formal/`
- Regional compilation failure probes:
  - `A100_0606/distributed_startup_regional_tp2_smoke_v2/`
  - `A100_0606/distributed_startup_regional_tp2_smoke_v4/`
  - `A100_0606/distributed_startup_regional_tp2_smoke_v5/`
- Persistent runtime probes:
  - `A100_0606/distributed_startup_exploration/fullfx_pool_reuse_probe.py`
  - `A100_0606/distributed_startup_exploration/persistent_shell_reentry_pp2.log`
- Generated topology checkpoints:
  - `/data/wenbiao/igniteserve_probe/llama8b_tp2/`
  - `/data/wenbiao/igniteserve_probe/llama8b_pp2/`
