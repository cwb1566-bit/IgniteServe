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

## Cumulative Result Relative to the Previous D3 Path

| Topology | State | D3 Total Init | Current Total Init | Reduction |
|---|---|---:|---:|---:|
| TP=2 | cold model / cold compile | 33.5839 s | 20.9458 s | 37.6% |
| TP=2 | cold model / warm compile | 28.0953 s | 14.7602 s | 47.5% |
| TP=2 | warm model / warm compile | 24.8311 s | 14.8314 s | 40.3% |
| PP=2 | cold model / cold compile | 29.4559 s | 18.6869 s | 36.6% |
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

## Remaining Opportunities

1. Persist or preload the Full-FX worker runtime across engine instances. Its
   imports and CUDA initialization take about 4 s per worker and account for
   the remaining PP cold overlap loss. This requires a service-owned compiler
   daemon or transferable IPC protocol; the current pool is parent-process
   local.
2. Regional compilation of repeated decoder layers. Warm-cache Dynamo tracing
   remains about 7.3 s on TP=2 and 3.5 s on each PP=2 stage. This requires a
   guarded regional-frame design that preserves vLLM attention graph breaks and
   parameter binding; filesystem caches cannot eliminate it.
3. Persistent distributed worker shells. Reusing CUDA contexts, NCCL
   communicators, P2P probes, and compiler daemons across model swaps could
   remove roughly 2 s of bootstrap, but changes the multi-model service
   lifecycle and isolation boundary.
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
- Generated topology checkpoints:
  - `/data/wenbiao/igniteserve_probe/llama8b_tp2/`
  - `/data/wenbiao/igniteserve_probe/llama8b_pp2/`
