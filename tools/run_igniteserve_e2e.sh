#!/usr/bin/env bash
# IgniteServe end-to-end driver: materialize (Phase 1) and formal (Phase 2)
# CUDA-graph hotspot runs for Qwen3-4B / Llama2-7B over cold / warm / hot
# scenarios with burst + rps[1,2,5,10] workloads.
#
# One fresh server per hotspot key (clean state + reliable CSV export on
# graceful shutdown). Scenario order per model is cold -> warm* -> hot* so the
# torch compile cache and weight pagecache are warmed by the cold run and reused
# by warm/hot. vllm_startup_resources (best_config/kv_profile/model_info/
# transformed_code/parallel_compile_metadata + hotspot CSVs) is ALWAYS preserved.
#
# Usage:
#   tools/run_igniteserve_e2e.sh --stage material
#   tools/run_igniteserve_e2e.sh --stage formal
# Optional filters (comma lists):
#   --models Qwen3-4B,Llama2-7B   --scenarios cold,warm,hot   --rps 1,2,5,10
#   --skip-existing   (skip a key whose client CSV already exists)
set -u -o pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:-$REPO/models}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
GPU=${GPU:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8089}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
GPU_UTIL=${GPU_UTIL:-0.9}
SUFFIX="__len${MAX_MODEL_LEN}__gpuutil${GPU_UTIL}"

DATASET=${DATASET:-$REPO/benchmarks_custom/ShareGPT_validated_2000.json}
RESOURCE_ROOT=${RESOURCE_ROOT:-$REPO/vllm_startup_resources}
# Ground-truth hotspot CSVs produced by the material stage. NEVER written by
# the formal stage (formal uses a separate, seeded HOTSPOT_ROOT below).
MATERIAL_HOTSPOT_ROOT=${MATERIAL_HOTSPOT_ROOT:-$RESOURCE_ROOT/cuda_graph_hotspots}
COMPILE_CACHE_DIR=${COMPILE_CACHE_DIR:-$HOME/.cache/vllm/torch_compile_cache}
OUT_ROOT=${OUT_ROOT:-$REPO/benchmark_results/IgniteServe_e2e}

BURST_CLIENT=$REPO/benchmarks_custom/vllm_benchmark_burst.py
RPS_CLIENT=$REPO/benchmarks_custom/vllm_benchmark_rps.py

READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}
SHUTDOWN_TIMEOUT_S=${SHUTDOWN_TIMEOUT_S:-240}
KEEPALIVE_COLD_WARM=${KEEPALIVE_COLD_WARM:-120}
KEEPALIVE_HOT=${KEEPALIVE_HOT:-900}

COLD_NUM=${COLD_NUM:-100}
COLD_WORKERS=${COLD_WORKERS:-10}
RPS_NUM=${RPS_NUM:-2000}

# Defaults (overridable by flags)
STAGE=""
MODELS_FILTER="Qwen3-4B,Llama2-7B"
SCENARIOS_FILTER="cold,warm,hot"
RPS_FILTER="1,2,5,10"
SKIP_EXISTING=0

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2;;
    --models) MODELS_FILTER="$2"; shift 2;;
    --scenarios) SCENARIOS_FILTER="$2"; shift 2;;
    --rps) RPS_FILTER="$2"; shift 2;;
    --skip-existing) SKIP_EXISTING=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ "$STAGE" != "material" && "$STAGE" != "formal" ]]; then
  echo "ERROR: --stage must be 'material' or 'formal'" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset does not exist: $DATASET" >&2
  exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODELS_FILTER"
IFS=',' read -r -a SCENARIOS <<< "$SCENARIOS_FILTER"
IFS=',' read -r -a RPS_LIST <<< "$RPS_FILTER"

STAGE_OUT="$OUT_ROOT/$STAGE"
mkdir -p "$STAGE_OUT"

# Stage-specific hotspot root the SERVER reads/writes (VLLM_CUDA_GRAPH_HOTSPOT_ROOT).
# - material: write ground truth directly.
# - formal: a SEPARATE root, seeded per-key with a copy of the material CSV so
#   precapture reads the materialized Top-K while the formal export overwrites
#   only the copy (ground truth in MATERIAL_HOTSPOT_ROOT stays pristine).
if [[ "$STAGE" == "material" ]]; then
  HOTSPOT_ROOT="$MATERIAL_HOTSPOT_ROOT"
else
  HOTSPOT_ROOT="$STAGE_OUT/_hotspot_root"
  mkdir -p "$HOTSPOT_ROOT"
fi

RUN_LOG="$STAGE_OUT/run.log"
SUMMARY_TSV="$STAGE_OUT/key_status.tsv"
: >"$SUMMARY_TSV"

log_msg() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$RUN_LOG"
}

contains() {  # contains <needle> <space-array...>
  local n="$1"; shift
  local x
  for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done
  return 1
}

# ---------------------------------------------------------------------------
# Cache / pagecache helpers
# ---------------------------------------------------------------------------
clear_compile_cache() {
  log_msg "clear compile cache: $COMPILE_CACHE_DIR"
  rm -rf "$COMPILE_CACHE_DIR"
  mkdir -p "$COMPILE_CACHE_DIR"
}

drop_pagecache() {
  local model_path="$1" log_file="$2"
  log_msg "drop model pagecache: $model_path"
  "$PYTHON_BIN" -m vllm.v1.engine.weight_pagecache drop \
    --model "$model_path" >"$log_file" 2>&1 || true
}

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
kill_port() {
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      kill $pids 2>/dev/null || true; sleep 2
      kill -9 $pids 2>/dev/null || true
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  fi
}

SERVER_PID=""
start_server() {
  local keepalive="$1" server_log="$2" placeholder_model="$3"
  local stage_env=()
  if [[ "$STAGE" == "material" ]]; then
    stage_env=(VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE=1)
  fi
  kill_port
  : >"$server_log"
  log_msg "start server (keepalive=$keepalive) -> $server_log"

  setsid env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONUNBUFFERED=1 \
    VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
    VLLM_STARTUP_RESOURCE_ROOT="$RESOURCE_ROOT" \
    VLLM_CUDA_GRAPH_HOTSPOT_ROOT="$HOTSPOT_ROOT" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    ${stage_env[@]+"${stage_env[@]}"} \
    "$PYTHON_BIN" "$REPO/multi_model_server.py" \
      --host "$HOST" --port "$PORT" \
      --keepalive "$keepalive" \
      --max-concurrent-models 1 \
      --model "$placeholder_model" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      >>"$server_log" 2>&1 &
  SERVER_PID=$!
  log_msg "server pid=$SERVER_PID"
}

wait_server_ready() {
  local server_log="$1" start_ts now_ts
  start_ts=$(date +%s)
  while true; do
    if curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log_msg "server pid $SERVER_PID exited before ready"
      return 1
    fi
    if grep -qE "Traceback|EngineCore failed|RuntimeError:|AssertionError" \
        "$server_log" 2>/dev/null; then
      log_msg "server log shows fatal error before ready"
      return 1
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts > READY_TIMEOUT_S )); then
      log_msg "server ready timeout (${READY_TIMEOUT_S}s)"
      return 124
    fi
    sleep 2
  done
}

stop_server_graceful() {
  # SIGTERM the uvicorn main pid only so lifespan shutdown runs
  # (shutdown_all -> engine.shutdown -> EngineCore exports hotspot CSV).
  local pid="$SERVER_PID" start_ts now_ts
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    log_msg "graceful stop server pid=$pid (waiting for CSV export)"
    kill -TERM "$pid" 2>/dev/null || true
    start_ts=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
      now_ts=$(date +%s)
      if (( now_ts - start_ts > SHUTDOWN_TIMEOUT_S )); then
        log_msg "graceful stop timeout; sending SIGKILL to process group"
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        break
      fi
      sleep 2
    done
  fi
  kill_port
  SERVER_PID=""
}

# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------
send_warmup() {  # send_warmup <model_path> <key>
  local model_path="$1" key="$2"
  log_msg "warmup request key=$key"
  curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-VLLM-HOTSPOT-KEY: $key" \
    -d "{\"model\":\"$model_path\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":8,\"temperature\":0,\"stream\":true}" \
    >/dev/null 2>&1 || log_msg "warmup request failed (non-fatal)"
}

run_burst() {  # run_burst <model_path> <key> <num> <workers> <csv> <clientlog>
  local model_path="$1" key="$2" num="$3" workers="$4" csv="$5" clog="$6"
  "$PYTHON_BIN" "$BURST_CLIENT" \
    --base-url "http://$HOST:$PORT/v1" \
    --model "$model_path" \
    --dataset "$DATASET" \
    --num-prompts "$num" \
    --max-workers "$workers" \
    --output-csv "$csv" \
    --hotspot-key "$key" >"$clog" 2>&1
}

run_rps() {  # run_rps <model_path> <key> <num> <rate> <csv> <clientlog>
  local model_path="$1" key="$2" num="$3" rate="$4" csv="$5" clog="$6"
  "$PYTHON_BIN" "$RPS_CLIENT" \
    --base-url "http://$HOST:$PORT/v1" \
    --model "$model_path" \
    --dataset "$DATASET" \
    --num-prompts "$num" \
    --request-rate "$rate" \
    --output-csv "$csv" \
    --hotspot-key "$key" >"$clog" 2>&1
}

# ---------------------------------------------------------------------------
# Per-key runner
# ---------------------------------------------------------------------------
run_key() {
  # run_key <model_label> <scenario> <client> <key> <num> <rate> <workers> <keepalive>
  local model_label="$1" scenario="$2" client="$3" key="$4"
  local num="$5" rate="$6" workers="$7" keepalive="$8"
  local model_path="$MODEL_ROOT/$model_label"
  local full_key="${key}${SUFFIX}"
  local out_dir="$STAGE_OUT/$model_label"
  mkdir -p "$out_dir"
  local server_log="$out_dir/${full_key}.server.log"
  local client_log="$out_dir/${full_key}.client.log"
  local client_csv="$out_dir/${full_key}.csv"
  local drop_log="$out_dir/${full_key}.droppagecache.log"
  # Path the server writes/reads this stage; for material this IS the ground truth.
  local hits_csv="$HOTSPOT_ROOT/${full_key}/cuda_graph_hits.csv"
  # Ground-truth materialized CSV (never overwritten by the formal stage).
  local material_csv="$MATERIAL_HOTSPOT_ROOT/${full_key}/cuda_graph_hits.csv"

  log_msg "=== [$STAGE][$model_label][$scenario] key=$full_key ==="

  if [[ "$SKIP_EXISTING" == "1" && -s "$client_csv" ]]; then
    log_msg "skip-existing: $client_csv present, skipping"
    return 0
  fi

  # State preparation BEFORE server start.
  if [[ "$scenario" == "cold" ]]; then
    clear_compile_cache
    drop_pagecache "$model_path" "$drop_log"
  fi

  # formal: seed the (separate) hotspot root with the materialized CSV so the
  # server's precapture loads the materialized Top-K. The formal export will
  # overwrite only this copy, leaving MATERIAL_HOTSPOT_ROOT pristine.
  if [[ "$STAGE" == "formal" ]]; then
    if [[ ! -s "$material_csv" ]]; then
      log_msg "ERROR: material CSV missing for key=$full_key: $material_csv"
      printf '%s\t%s\t%s\t%s\n' "$model_label" "$scenario" "$full_key" "NO_MATERIAL_CSV" >>"$SUMMARY_TSV"
      return 1
    fi
    mkdir -p "$HOTSPOT_ROOT/${full_key}"
    cp -f "$material_csv" "$hits_csv"
    log_msg "seeded formal hotspot CSV from material: $hits_csv"
  fi

  start_server "$keepalive" "$server_log" "$model_path"
  if ! wait_server_ready "$server_log"; then
    log_msg "FAILED to start server for key=$full_key"
    stop_server_graceful
    printf '%s\t%s\t%s\t%s\n' "$model_label" "$scenario" "$full_key" "SERVER_FAIL" >>"$SUMMARY_TSV"
    return 1
  fi

  # hot: warm the engine resident with the SAME key before measuring.
  if [[ "$scenario" == "hot" ]]; then
    send_warmup "$model_path" "$key"
  fi

  local rc=0
  if [[ "$client" == "burst" ]]; then
    run_burst "$model_path" "$key" "$num" "$workers" "$client_csv" "$client_log" || rc=$?
  else
    run_rps "$model_path" "$key" "$num" "$rate" "$client_csv" "$client_log" || rc=$?
  fi
  log_msg "workload rc=$rc -> $client_csv"

  stop_server_graceful

  # Verification.
  local status="OK"
  if [[ "$STAGE" == "material" ]]; then
    if [[ -s "$hits_csv" ]]; then
      log_msg "material CSV OK: $hits_csv"
      status="CSV_OK"
    else
      log_msg "WARN: material CSV missing/empty: $hits_csv"
      status="CSV_MISSING"
    fi
  else
    local hit_report="$out_dir/${full_key}.hitcheck.json"
    if "$PYTHON_BIN" "$REPO/tools/check_hotspot_hit.py" \
        --server-log "$server_log" \
        --material-csv "$material_csv" \
        --key "$full_key" \
        --report "$hit_report"; then
      status="HIT_PASS"
    else
      status="HIT_FAIL"
    fi
    log_msg "hit-check: $status -> $hit_report"
  fi
  [[ "$rc" != "0" ]] && status="${status}+WORKLOAD_RC${rc}"
  printf '%s\t%s\t%s\t%s\n' "$model_label" "$scenario" "$full_key" "$status" >>"$SUMMARY_TSV"
  return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  log_msg "stage=$STAGE models=${MODELS[*]} scenarios=${SCENARIOS[*]} rps=${RPS_LIST[*]}"
  log_msg "dataset=$DATASET"
  log_msg "resource_root=$RESOURCE_ROOT (PRESERVED)  compile_cache=$COMPILE_CACHE_DIR"
  log_msg "out=$STAGE_OUT"

  if [[ ! -s "$DATASET" ]]; then
    log_msg "ERROR: dataset missing: $DATASET"; exit 1
  fi
  for m in "${MODELS[@]}"; do
    if [[ ! -d "$MODEL_ROOT/$m" ]]; then
      log_msg "ERROR: model dir missing: $MODEL_ROOT/$m"; exit 1
    fi
  done

  local model scenario R
  for model in "${MODELS[@]}"; do
    # cold (burst) FIRST so it warms compile cache + pagecache for warm/hot.
    if contains "cold" "${SCENARIOS[@]}"; then
      run_key "$model" "cold" "burst" \
        "${model}__burst__cold__req${COLD_NUM}__rpsnone__workers${COLD_WORKERS}" \
        "$COLD_NUM" "none" "$COLD_WORKERS" "$KEEPALIVE_COLD_WARM"
    fi
    if contains "warm" "${SCENARIOS[@]}"; then
      for R in "${RPS_LIST[@]}"; do
        run_key "$model" "warm" "rps" \
          "${model}__rps__warm__req${RPS_NUM}__rps${R}" \
          "$RPS_NUM" "$R" "0" "$KEEPALIVE_COLD_WARM"
      done
    fi
    if contains "hot" "${SCENARIOS[@]}"; then
      for R in "${RPS_LIST[@]}"; do
        run_key "$model" "hot" "rps" \
          "${model}__rps__hot__req${RPS_NUM}__rps${R}" \
          "$RPS_NUM" "$R" "0" "$KEEPALIVE_HOT"
      done
    fi
  done

  log_msg "DONE. status summary: $SUMMARY_TSV"
  cat "$SUMMARY_TSV" | tee -a "$RUN_LOG"
}

main "$@"
