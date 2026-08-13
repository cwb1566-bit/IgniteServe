# Deployment Guide

## Prerequisites

- Linux x86-64
- NVIDIA driver compatible with CUDA 12.6
- Python 3.10
- Local or mounted Hugging Face model weights
- Enough host memory for the selected model's page cache
- `vmtouch` is optional; IgniteServe has a Python file-read fallback

The project does not include model weights, generated startup resources, a modified PyTorch source tree, or private benchmark logs.

## Build From Source

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

Verify the active source tree:

```bash
python -c 'import vllm, pathlib; print(vllm.__version__); print(pathlib.Path(vllm.__file__).resolve())'
```

The printed package path should point into the IgniteServe checkout. Setting `PYTHONPATH=$PWD` is recommended for research runs because it removes ambiguity when another vLLM installation exists in the same environment.

## Single-Model API Server

```bash
export IGNITESERVE_ROOT=/opt/IgniteServe
export MODEL_PATH=/models/Qwen3-4B
export RESOURCE_ROOT=/var/lib/igniteserve/startup_resources

mkdir -p "$RESOURCE_ROOT/cuda_graph_hotspots/qwen3-4b-default"

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
  --host 0.0.0.0 \
  --port 8000 \
  --model "$MODEL_PATH" \
  --served-model-name Qwen3-4B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

For a first correctness check, omit `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH` until an offline hotspot profile has been generated. The engine safely falls back on resource misses, but the first run is expected to perform compilation, profiling, and profile generation work.

## On-Demand Multi-Model Server

`multi_model_server.py` starts an engine when a request names a model that is not resident, keeps it alive for a configurable interval, and evicts the least recently used model when the resident limit is reached.

```bash
export IGNITESERVE_ROOT=/opt/IgniteServe
export RESOURCE_ROOT=/var/lib/igniteserve/startup_resources

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$IGNITESERVE_ROOT \
VLLM_USE_V1=1 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
VLLM_STARTUP_COMPILE_MODE=s4 \
VLLM_STARTUP_DAG_MODE=execute \
VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1 \
VLLM_STARTUP_RESOURCE_ROOT=$RESOURCE_ROOT \
VLLM_CUDA_GRAPH_HOTSPOT_ROOT=$RESOURCE_ROOT/cuda_graph_hotspots \
python multi_model_server.py \
  --host 0.0.0.0 \
  --port 8089 \
  --keepalive 30 \
  --max-concurrent-models 1 \
  --model /models/Qwen3-4B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

The `--model` value initializes the shared vLLM argument template. Requests may name another local model path. Do not expose this research endpoint to untrusted clients: model names are interpreted as loadable model identifiers.

## Workload-Specific Hotspot Profiles

Benchmark clients send a logical key in `X-VLLM-HOTSPOT-KEY`. The server appends settings that only it knows:

```text
<client-key>__len<max-model-len>__gpuutil<gpu-memory-utilization>
```

For example, the client key:

```text
Qwen3-4B__rps__hot__req2000__rps10
```

becomes:

```text
$VLLM_CUDA_GRAPH_HOTSPOT_ROOT/
  Qwen3-4B__rps__hot__req2000__rps10__len2048__gpuutil0.9/
    cuda_graph_hits.csv
    cuda_graph_hits_top99.csv
    cuda_graph_profile_manifest.json
    cuda_graph_cost_table.csv          # when cost profiling is enabled
```

The same root controls both export and later loading. Stop the server gracefully so the analyzer can export final CSVs.

## Operational Checks

- Logs containing `[TIMER]` report the seven EngineCore phases and `Total Init`.
- A startup resource hit is always accompanied by a hit/compatibility log; do not infer a hit only from elapsed time.
- Use one resource root per trust domain and protect it from untrusted writes. Generated Python/compiler artifacts are executable inputs.
- Keep `VLLM_STARTUP_DAG_MODE=off` for distributed configurations until that topology has been validated.
- Keep `VLLM_ENABLE_SUBPROCESS_COMPILE` unset unless the external PyTorch integration is installed and tested.
