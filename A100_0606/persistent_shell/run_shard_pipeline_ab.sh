#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wenbiao/vllm_my_A100
PY=/home/wenbiao/miniconda3/envs/vllm_my_A100/bin/python
MODEL=/data/wenbiao/models/Meta-Llama-3-8B-Instruct

cd "$ROOT"
for spec in on2:1 off2:0 on3:1 off3:0; do
    name=${spec%%:*}
    flag=${spec##*:}
    "$PY" A100_0606/persistent_shell/evict_model_pagecache.py "$MODEL"
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$ROOT" \
    VLLM_USE_V1=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=0 \
    VLLM_ENABLE_WEIGHT_SHARD_PIPELINE="$flag" \
    "$PY" A100_0606/persistent_shell/test_single_fresh.py \
        --model "$MODEL" \
        > "A100_0606/persistent_shell/shard_pipeline_${name}_cold.log" 2>&1
done
