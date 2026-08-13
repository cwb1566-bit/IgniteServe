#!/usr/bin/env bash
# Validate cold-start startup optimizations do not change inference outputs.
# For each model we greedily decode the SAME fixed prompt set under:
#   base  : VLLM_ENABLE_STARTUP_OPTIMIZATIONS=0  (native S0 path)
#   opt   : VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1  (S4 optimized path + caches)
#   base2 : VLLM_ENABLE_STARTUP_OPTIMIZATIONS=0  (noise-floor / determinism check)
# Then compare opt-vs-base and base2-vs-base.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PY=${PY:-$(command -v python)}
MODELS_ROOT=${MODELS_ROOT:-$REPO/models}
RES_ROOT=${RES_ROOT:-$REPO/vllm_startup_resources}
GPU="${GPU:-0}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$REPO/benchmark_results/coldstart_correctness_$STAMP}
HARNESS="$REPO/tools/validate_coldstart_correctness.py"

# Allow space-separated override via env: MODELS="A B C"
read -r -a MODELS <<< "${MODELS:-Qwen3-8B Llama2-7B OPT-6.7B}"

NUM_RANDOM="${NUM_RANDOM:-64}"
NUM_SHAREGPT="${NUM_SHAREGPT:-64}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_PROMPT_LEN=$(( MAX_MODEL_LEN - MAX_TOKENS - 16 ))

mkdir -p "$OUT"
echo "[run] out=$OUT gpu=$GPU models=${MODELS[*]}" | tee "$OUT/run.log"

common_env=(
  "PYTHONPATH=$REPO"
  "CUDA_VISIBLE_DEVICES=$GPU"
  "HF_HUB_OFFLINE=1"
  "VLLM_USE_V1=1"
  "TOKENIZERS_PARALLELISM=false"
)

gen() {  # $1=model_path $2=opt_flag(0/1) $3=prompts.json $4=out.json $5=logfile
  env "${common_env[@]}" \
      VLLM_ENABLE_STARTUP_OPTIMIZATIONS="$2" \
      VLLM_STARTUP_RESOURCE_ROOT="$RES_ROOT" \
      "$PY" "$HARNESS" gen \
        --prompts "$3" --out "$4" --max-tokens "$MAX_TOKENS" \
        --tp 1 --dtype float16 --gpu-mem-util 0.85 --max-num-seqs 64 \
        --max-model-len "$MAX_MODEL_LEN" --seed 0 >"$5" 2>&1
}

for m in "${MODELS[@]}"; do
  mp="$MODELS_ROOT/$m"
  mo="$OUT/$m"
  mkdir -p "$mo"
  echo "===== $m =====" | tee -a "$OUT/run.log"

  echo "[sample] $m" | tee -a "$OUT/run.log"
  env "${common_env[@]}" "$PY" "$HARNESS" sample \
      --model "$mp" --num-random "$NUM_RANDOM" --num-sharegpt "$NUM_SHAREGPT" \
      --input-len 256 --output-len "$MAX_TOKENS" --max-tokens "$MAX_TOKENS" \
      --max-prompt-len "$MAX_PROMPT_LEN" \
      --seed 0 --out "$mo/prompts.json" >"$mo/sample.log" 2>&1

  echo "[gen base]  $m" | tee -a "$OUT/run.log"
  gen "$mp" 0 "$mo/prompts.json" "$mo/out_base.json"  "$mo/gen_base.log" \
    || { echo "[FAIL gen base] $m (see $mo/gen_base.log)" | tee -a "$OUT/run.log"; continue; }
  echo "[gen opt]   $m" | tee -a "$OUT/run.log"
  gen "$mp" 1 "$mo/prompts.json" "$mo/out_opt.json"   "$mo/gen_opt.log" \
    || { echo "[FAIL gen opt] $m (see $mo/gen_opt.log)" | tee -a "$OUT/run.log"; continue; }
  echo "[gen base2] $m" | tee -a "$OUT/run.log"
  gen "$mp" 0 "$mo/prompts.json" "$mo/out_base2.json" "$mo/gen_base2.log" \
    || { echo "[FAIL gen base2] $m (see $mo/gen_base2.log)" | tee -a "$OUT/run.log"; continue; }

  echo "[compare opt-vs-base] $m" | tee -a "$OUT/run.log"
  "$PY" "$HARNESS" compare --ref "$mo/out_base.json" --cmp "$mo/out_opt.json" \
      --out "$mo/compare_opt_vs_base.json" | tee -a "$OUT/run.log"
  echo "[compare base2-vs-base] $m" | tee -a "$OUT/run.log"
  "$PY" "$HARNESS" compare --ref "$mo/out_base.json" --cmp "$mo/out_base2.json" \
      --out "$mo/compare_base2_vs_base.json" | tee -a "$OUT/run.log"
done

echo "[done] results in $OUT" | tee -a "$OUT/run.log"
