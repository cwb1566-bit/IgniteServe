# Configuration Reference

## Core Startup Controls

| Variable | Default | Meaning |
|---|---:|---|
| `VLLM_USE_V1` | vLLM default | Selects the V1 engine. IgniteServe startup work targets V1. |
| `VLLM_ENABLE_STARTUP_OPTIMIZATIONS` | `0` | Master switch for the S4 resource-reuse features. |
| `VLLM_STARTUP_COMPILE_MODE` | `s4` with master switch, otherwise `s0` | Selects the stage-separated compile policy. Recommended values are `s0` and `s4`. |
| `VLLM_STARTUP_RESOURCE_ROOT` | `<repo>/vllm_startup_resources` | Root for all typed startup resources. |
| `VLLM_CACHE_ROOT` | vLLM user cache | Native vLLM/TorchInductor compilation cache root. This is distinct from the semantic startup resource root. |
| `VLLM_STARTUP_DAG_MODE` | `off` | `off`, `shadow`, or `execute`. `execute` is validated for single-process `TP=PP=DP=1`. |
| `VLLM_STARTUP_DAG_BOOTSTRAP_MAX_WORKERS` | `2` | Bootstrap DAG scheduler width. |
| `VLLM_STARTUP_DAG_MAX_WORKERS` | `2` | Runtime DAG scheduler width. More workers did not improve the validated plan. |
| `VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM` | follows master switch | Starts asynchronous weight-file prefetch and stops it before `load_weights()`. |

## S4 Resource Controls

Each switch follows `VLLM_ENABLE_STARTUP_OPTIMIZATIONS` unless explicitly set.

| Variable | Resource | Function |
|---|---|---|
| `VLLM_ENABLE_STRUCTURAL_REUSE` | `parallel_compile_metadata/` | Compiles canonical piecewise graphs and reuses structurally equivalent pieces. |
| `VLLM_DISABLE_STRUCTURAL_REUSE` | none | Forces identity mapping for an S0-style comparison. |
| `VLLM_ENABLE_PARALLEL_COMPILE_METADATA` | `parallel_compile_metadata/` | Persists/restores subgraph input and fake-run/data-flow metadata. |
| `VLLM_ENABLE_BEST_CONFIG_CACHE` | `best_config/` | Reuses Triton selections by semantic kernel/config identity. |
| `VLLM_ENABLE_TRANSFORMED_CODE_CACHE` | `transformed_code/` | Reuses transformed TorchDynamo code artifacts. |
| `VLLM_ENABLE_KV_PROFILE_CACHE` | `kv_profile/` | Reuses available KV-cache bytes for an equivalent memory-affecting configuration. |
| `VLLM_ENABLE_COMPILE_CACHE_ALIAS` | global subgraph map | Removes irrelevant model deployment paths from compatible compile-cache lookup. |
| `VLLM_CACHED_AVAILABLE_KV_CACHE_BYTES` | explicit value | Overrides profiling with a caller-provided byte count. Prefer the keyed cache for normal use. |
| `VLLM_OFFLINE_BEST_CONFIG_ROOT` | optional path | Adds an external `.best_config` source during S4 resolution. |

## Adaptive CUDA Graph Controls

| Variable | Default | Meaning |
|---|---:|---|
| `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH` | `0` | Enables hotspot pre-capture, adaptive state tracking, and admitted online capture. |
| `VLLM_CUDA_GRAPH_HOTSPOT_ROOT` | resource root subdirectory | Multi-model parent directory selected by the server and client hotspot key. |
| `VLLM_CUDA_GRAPH_HOTSPOT_DIR` | resource root subdirectory | Exact workload directory used by a single engine for both load and export. |
| `VLLM_CUDA_GRAPH_HOTSPOT_TOP_K` | unset | Optional Top-K override for controlled experiments. |
| `VLLM_CUDA_GRAPH_PRECAPTURE_MIN_HITS` | `10` | Minimum offline hit count for startup pre-capture. |
| `VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS` | `10` | Runtime observations required before a missing decode shape is queued. |
| `VLLM_ADAPTIVE_CUDAGRAPH_MANDATORY_SHAPES` | implementation default | Shapes always included in the sparse startup set; experiments commonly use `1,2`. |
| `VLLM_ADAPTIVE_CUDAGRAPH_IDLE_GRACE_MS` | `10` | Continuous idle window before maintenance capture. |
| `VLLM_ADAPTIVE_CUDAGRAPH_MAX_CAPTURES_PER_IDLE` | `1` | Maximum graphs admitted in one idle opportunity. |
| `VLLM_CUDAGRAPH_MAX_PAD_RATIO` | `0.20` | Maximum upward-padding ratio for sparse graph matching. |
| `VLLM_CUDAGRAPH_COST_TABLE` | auto-discovered | Optional explicit GPU cost table. The manifest must match. |
| `VLLM_CUDAGRAPH_MIN_ESTIMATED_SAVING_RATIO` | `0.0` | Required estimated graph saving before padded replay. |
| `VLLM_ADAPTIVE_CUDAGRAPH_ENABLE_SLO_CAPTURE` | `1` when adaptive is enabled | Allows capture under conservative request-slack admission. |
| `VLLM_ADAPTIVE_CUDAGRAPH_TPOT_SLO_MS` | `100` | TPOT SLO used for admission estimates. Tune for the workload. |
| `VLLM_ADAPTIVE_CUDAGRAPH_TTFT_SLO_MS` | `2000` | TTFT SLO used for admission estimates. Tune for the workload. |
| `VLLM_ADAPTIVE_CUDAGRAPH_SLO_GUARD_MS` | `5` | Safety margin subtracted from available slack. |
| `VLLM_ADAPTIVE_CUDAGRAPH_SLO_CAPTURE_INTERVAL_STEPS` | `8` | Minimum model steps between SLO capture attempts. |
| `VLLM_ADAPTIVE_CUDAGRAPH_SLO_PIECES_PER_STEP` | `0` | `0` selects adaptive piece budgeting; positive values force a fixed experimental budget. |
| `VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE` | `0` | Disables the legacy dynamic-capture path. Materialization runs use it without enabling adaptive capture. |

## Experimental Compilation Controls

| Variable | Default | Status |
|---|---:|---|
| `VLLM_ENABLE_SUBPROCESS_COMPILE` | `0` | Enables the process-isolated compilation integration. Requires a compatible external PyTorch runtime not shipped in this repository. |
| `VLLM_ENABLE_THREAD_PARALLEL_COMPILE` | `0` | Research-only. In-process compiler state is not sufficiently isolated for the supported path. |

## Recommended Profiles

### Portable stage-separated baseline

```bash
VLLM_STARTUP_COMPILE_MODE=s0
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=0
VLLM_STARTUP_DAG_MODE=off
```

### Optimized single-GPU startup

```bash
VLLM_USE_V1=1
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1
VLLM_STARTUP_COMPILE_MODE=s4
VLLM_STARTUP_DAG_MODE=execute
VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1
VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1
```

Do not enable every experimental flag simply because it exists. The master S4 path and adaptive CUDA Graph controller are the intended research configuration; subprocess compilation is a separate compatibility choice.
