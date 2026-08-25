# IgniteServe

IgniteServe is a research inference framework for reducing large-language-model cold-start latency while preserving steady-state serving performance.

It is developed by the **CGCL, Huazhong University of Science and Technology**. This repository is a private research codebase derived from [vLLM](https://github.com/vllm-project/vllm) 0.9.2.

## What IgniteServe Adds

- **Stage-separated model compilation.** Compilation is separated from the peak-memory profiling forward pass, enabling each step to be scheduled and cached independently.
- **S4 startup resource reuse.** Structural subgraph metadata, Triton best configurations, transformed TorchDynamo code, and KV-profile results are persisted under one resource root.
- **Metadata-aware graph reuse.** Repeated piecewise subgraphs reuse canonical compiled artifacts and saved data-flow metadata.
- **Startup DAG orchestration.** A 25-node dependency graph covers EngineCore construction, model initialization, weight loading, compilation, KV-cache setup, CUDA Graph pre-capture, and runtime-manager setup.
- **Asynchronous weight-page-cache prewarm.** Model weight files are prefetched while the engine performs earlier initialization work, then stopped before in-process weight loading to avoid I/O contention.
- **Adaptive CUDA Graph capture.** Offline hotspot profiles select a small pre-capture set; missing decode shapes are warmed by real requests and captured under idle or SLO-slack admission.
- **Cost-aware graph matching.** Runtime execution selects exact replay, bounded upward padding, or eager fallback using profile compatibility and measured GPU costs.

## Validated Environment

The current research branch has been validated primarily with:

| Component | Version |
|---|---|
| GPU | NVIDIA A100 PCIe 40 GB |
| Python | 3.10 |
| PyTorch | 2.7.0+cu126 |
| CUDA runtime | 12.6 |
| Triton | 3.3.0 |
| vLLM base | 0.9.2 |

The stable deployment target is Linux, NVIDIA CUDA, vLLM V1, FP16/BF16, and a single-process single-GPU engine. Distributed configurations retain the normal startup path unless explicitly validated.

## Installation

```bash
git clone git@github.com:cwb1566-bit/IgniteServe.git
cd IgniteServe

conda create -n igniteserve python=3.10 -y
conda activate igniteserve

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
pip install -e . --no-build-isolation
```

IgniteServe does **not** bundle the separately modified PyTorch source tree used by some parallel-compilation experiments. The standard S4 startup path works without enabling that experimental feature.

## Quick Start

Set a model path and start the OpenAI-compatible server:

```bash
export IGNITESERVE_ROOT=$PWD
export MODEL_PATH=/path/to/Qwen3-4B
export RESOURCE_ROOT=$IGNITESERVE_ROOT/vllm_startup_resources

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$IGNITESERVE_ROOT \
VLLM_USE_V1=1 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
VLLM_STARTUP_COMPILE_MODE=s4 \
VLLM_STARTUP_DAG_MODE=execute \
VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1 \
VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1 \
VLLM_STARTUP_RESOURCE_ROOT=$RESOURCE_ROOT \
VLLM_CUDA_GRAPH_HOTSPOT_DIR=$RESOURCE_ROOT/cuda_graph_hotspots/qwen3-4b-default \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-4B \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

The first compatible launch populates missing resources by executing the required native work. Later launches reuse validated entries. Never copy a resource directory between incompatible model, compiler, GPU, dtype, or serving configurations without its manifest checks.

Test the endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "Explain cold starts briefly."}],
    "max_tokens": 64
  }'
```

## Documentation

- [Chinese branch, deployment, and experiment guide](BRANCHES_AND_EXPERIMENTS_CN.md)
- [Architecture](docs/architecture.md)
- [Deployment guide](docs/deployment.md)
- [Configuration reference](docs/configuration.md)
- [Benchmark methodology and results](docs/benchmarks.md)

## Research Status

IgniteServe is a systems research prototype, not a drop-in production replacement for every vLLM configuration. The following constraints are intentional:

- Startup DAG `execute` mode is validated for `TP=PP=DP=1`; unsupported executors/configurations fall back to shadow or conventional startup.
- Adaptive online capture currently authorizes decode-triggered shapes. Full CUDA Graphs remain phase-bound; piecewise graphs may reuse safe cross-phase pieces.
- Downward batch splitting and full-forward multi-stream replay are not enabled in the serving path because the measured coordination and resource-contention costs did not establish a stable benefit.
- Subprocess parallel compilation is experimental and requires a compatible external PyTorch runtime that is not included here.

## License and Attribution

IgniteServe is distributed under the [Apache License 2.0](LICENSE), preserving the license and notices of the upstream vLLM project. See [NOTICE](NOTICE) for project attribution.
