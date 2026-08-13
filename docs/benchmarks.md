# Benchmark Methodology and Results

This document reports validated representative results from the IgniteServe development system. They are not universal claims: model architecture, GPU, driver, compiler versions, sequence limits, and workload distribution all affect startup and serving behavior.

## Metrics

- **Model Compilation:** EngineCore time spent in the compilation phase.
- **KV Profile Probe:** time spent loading or measuring the peak-memory-derived KV-cache budget.
- **compile_probe_total:** native `profile_run()` for the coupled baseline, or `Model Compilation + KV Profile Probe` for a separated path.
- **Total Init:** the `[TIMER] Total Init` interval inside `EngineCore.__init__()`.
- **TTFT:** request arrival to first generated token.
- **TPOT:** per-output-token latency after the first token.

## Compilation and Profiling Ablation

Configuration: Qwen3-14B, FP16, TP=1, NVIDIA A100, cold native compile cache unless marked warm.

| ID | Configuration | compile_probe_total (s) | Total Init (s) | LLM initialization (s) |
|---|---|---:|---:|---:|
| N0 | Native cold `profile_run()` | 97.1154 | 111.2342 | 118.3785 |
| N1 | Native warm compile cache | 29.5892 | 43.2147 | 50.2903 |
| S0 | Stage separation only | 95.6655 | 109.0327 | 116.1077 |
| S1 | S0 + canonical subgraph reuse | 40.8182 | 53.9012 | 60.9400 |
| S2 | S1 + semantic best-config reuse | 39.7443 | 52.8934 | 59.9514 |
| S3 | S2 + transformed-code reuse | 28.7565 | 41.9118 | 49.0182 |
| S4 | S3 + KV-profile reuse | 26.1260 | 39.8928 | 47.0618 |

Key observations:

- Stage separation itself is nearly neutral: S0 is 1.45 s faster than N0 in this run, so the large gains come from reusable work, not relabeling phases.
- Canonical subgraph reuse provides the largest cold-cache reduction in this matrix.
- On this A100 software stack, transformed-code reuse saves about 11.0 s from S2 to S3, while semantic best-config reuse saves about 1.1 s. Earlier RTX 4090 experiments showed a different balance, demonstrating that autotuning/code-transformation costs are environment-dependent.
- S4 reduces `compile_probe_total` by 70.99 s (73.1%) relative to N0 and is 3.46 s faster than native warm-cache N1 under this measurement.

## Sparse and Adaptive CUDA Graph Capture

Configuration: Llama-3-8B-Instruct, FP16, TP=1, A100 PCIe 40 GB, `max_model_len=2048`, ShareGPT prompts. Each reported value is the median of three paired runs unless noted.

| Metric | Full 67 graphs | Adaptive 14 graphs | Change |
|---|---:|---:|---:|
| CUDA Graph pre-capture | 3.5481 s | 1.0421 s | -70.63% |
| EngineCore Total Init | 16.3070 s | 13.6799 s | -16.11% |
| Model-instance initialization | 17.3600 s | 14.7400 s | -15.09% |
| Burst mean TTFT | 4.4924 s | 3.8434 s | -14.45% |
| Burst P99 TTFT | 17.7651 s | 15.1256 s | -14.86% |
| Steady mean TPOT | 19.819 ms | 19.901 ms | +0.42% |
| Steady P99 TPOT | 20.804 ms | 20.902 ms | +0.47% |
| Steady throughput | 1993.31 tok/s | 1995.53 tok/s | +0.11% |

All 1440/1440 requests succeeded. Under this workload, sparse pre-capture lowered cold-start latency while the TPOT delta remained below 0.5%.

### Cost-aware six-graph startup

The next ablation compared adaptive 14 against Top-4 plus mandatory shapes `{1,2}`, using a measured GPU-cost table for padded replay.

| Metric | Adaptive 14 | Cost-aware 6 | Change |
|---|---:|---:|---:|
| CUDA Graph pre-capture | 1.0194 s | 0.7563 s | -25.82% |
| EngineCore Total Init | 13.5148 s | 13.3688 s | -1.08% |
| Burst P99 TTFT | 14.9366 s | 14.8162 s | -0.81% |
| Post-capture mean TPOT | 18.6000 ms | 18.6500 ms | +0.27% |
| Post-capture P99 TPOT | 19.8000 ms | 19.9000 ms | +0.51% |
| Post-capture throughput | 1801.89 tok/s | 1803.70 tok/s | +0.10% |

All 2720/2720 requests succeeded. Missing shapes `4, 8, 24, 40` completed the `warmup -> queued -> ready` transition in every run.

### SLO-admitted online capture

Across three continuous-load runs at RPS 10 (600/600 successful requests), SLO-admitted capture compared with idle-only capture changed mean TPOT by +0.199 ms (+0.69%), P99 TPOT by +0.206 ms (+0.65%), and throughput by +0.51%. All six online capture operations stayed within the admitted slack estimate.

In a cold-instance paired test, pre-capturing only shape 1 instead of all 67 graphs reduced pre-capture from 3.513 s to 0.576 s (-83.6%), EngineCore Total Init from 20.877 s to 18.207 s (-12.8%), and cold-request TTFT from 21.948 s to 19.279 s (-12.2%).

## Startup DAG Correctness Result

The final single-GPU 25-node DAG measured:

| Metric | Result |
|---|---:|
| Bootstrap DAG wall time | 1.2028 s |
| Runtime DAG wall time | 16.6070 s |
| Full DAG wall time | 17.8146 s |
| Sum of node work | 18.2942 s |
| Effective average parallelism | 1.027 |
| EngineCore Total Init | 17.8255 s |

The output hash matched the eager and fixed-path reference across the final correctness run. Increasing scheduler width from two to four workers did not produce a reproducible improvement, so both DAG stages default to two workers.

## Reproducing the Workloads

The repository contains a validated 2,000-request ShareGPT subset and two clients:

- `benchmarks_custom/vllm_benchmark_burst.py`
- `benchmarks_custom/vllm_benchmark_rps.py`

### Manual burst run

```bash
python benchmarks_custom/vllm_benchmark_burst.py \
  --base-url http://127.0.0.1:8089 \
  --model /models/Qwen3-4B \
  --dataset benchmarks_custom/ShareGPT_validated_2000.json \
  --num-prompts 100 \
  --max-workers 10 \
  --hotspot-key Qwen3-4B__burst__cold__req100__workers10 \
  --output-csv benchmark_results/qwen3-4b-burst.csv
```

### Manual Poisson-arrival run

```bash
python benchmarks_custom/vllm_benchmark_rps.py \
  --base-url http://127.0.0.1:8089 \
  --model /models/Qwen3-4B \
  --dataset benchmarks_custom/ShareGPT_validated_2000.json \
  --num-prompts 2000 \
  --max-workers 50 \
  --request-rate 10 \
  --seed 20260810 \
  --hotspot-key Qwen3-4B__rps__hot__req2000__rps10 \
  --output-csv benchmark_results/qwen3-4b-rps10.csv
```

The server adds `max_model_len` and `gpu_memory_utilization` to the key, so client-only and server-only configuration fields remain synchronized without hard-coding them into both commands.

### Two-stage automation

```bash
MODEL_ROOT=/models GPU=0 tools/run_igniteserve_e2e.sh --stage material
MODEL_ROOT=/models GPU=0 tools/run_igniteserve_e2e.sh --stage formal
```

- `material` disables online dynamic capture, executes each workload, and exports the ground-truth hotspot distribution.
- `formal` copies the matching materialized profile into an isolated run directory, enables the optimized serving path, and evaluates it without overwriting the ground truth.

Outputs are written under `benchmark_results/IgniteServe_e2e` by default. Raw local development logs and large ShareGPT datasets are intentionally not versioned.
