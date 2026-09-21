#!/bin/bash
# Start the TGI inference service on Ascend NPU (background, multi-card via HCCL).
#
# Usage: ./start-tgi.sh [options] [-- extra launcher args...]
#
# Options:
#   --model-id PATH|ID      local model dir, HF id or ModelScope id
#                           (default: Qwen/Qwen3-0.6B, downloaded via ModelScope
#                           on first use and cached in ~/.cache/modelscope)
#   --num-shard N           tensor parallelism degree (default: 2; 1 = single card)
#   --devices LIST          ASCEND_VISIBLE_DEVICES, e.g. "0,1" (default: "0,1")
#                           note: len(LIST) should be >= --num-shard
#   --port N                HTTP port (default: 8080)
#   --max-total-tokens N    (default: 128)
#   --max-input-tokens N    (default: 100)
#   --log FILE              log file (default: /tmp/tgi.log)
#   -h|--help               show this help
#
# Examples:
#   ./start-tgi.sh                                      # 2 shards on NPU 0,1
#   ./start-tgi.sh --num-shard 4 --devices 0,1,2,3      # 4 shards
#   ./start-tgi.sh --num-shard 1                        # single card
#   ./start-tgi.sh --model-id /path/to/model -- --json-output
#
# Stop with: ./stop-tgi.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- conda env (optional, same discovery as run-npu.sh) ----------
if [[ -n "${TGI_CONDA_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    elif [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    else
        echo "TGI_CONDA_ENV is set but conda is not available in this shell" >&2
        exit 1
    fi
    if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo "cannot locate conda base environment" >&2
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$TGI_CONDA_ENV"
fi

# ---------- defaults ----------
MODEL_ID="Qwen/Qwen3-0.6B"
NUM_SHARD=2
DEVICES="0,1"
PORT=8080
MAX_TOTAL_TOKENS=128
MAX_INPUT_TOKENS=100
LOG_FILE="/tmp/tgi.log"
PID_FILE="/tmp/tgi.pid"
EXTRA_ARGS=()

usage() {
    sed -n '2,25p' "$0"
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)          MODEL_ID="$2"; shift 2 ;;
        --num-shard)         NUM_SHARD="$2"; shift 2 ;;
        --devices)           DEVICES="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --max-total-tokens)  MAX_TOTAL_TOKENS="$2"; shift 2 ;;
        --max-input-tokens)  MAX_INPUT_TOKENS="$2"; shift 2 ;;
        --log)               LOG_FILE="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        --)                  shift; EXTRA_ARGS=("$@"); break ;;
        *)                   echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if ! [[ "$NUM_SHARD" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARD" -lt 1 ]]; then
    echo "invalid --num-shard: $NUM_SHARD" >&2; exit 1
fi

# Trim trailing whitespace/newlines from MODEL_ID: a path pasted with a
# stray newline would otherwise miss the local-dir check below and break
# the snapshot_download python one-liner (unterminated string literal).
MODEL_ID="${MODEL_ID%"${MODEL_ID##*[![:space:]]}"}"

# ---------- model: download via ModelScope if it's not a local path ----------
if [[ ! -d "$MODEL_ID" ]]; then
    if ! python -c "import modelscope" >/dev/null 2>&1; then
        echo "[MODEL] installing modelscope (one-time, may take a few minutes)..."
        if command -v uv >/dev/null 2>&1; then uv pip install "modelscope>=1.37.0"
        else python -m pip install "modelscope>=1.37.0"; fi
    fi
    echo "[MODEL] downloading '$MODEL_ID' via ModelScope (cached in ~/.cache/modelscope)..."
    MODEL_ID=$(python -c "from modelscope import snapshot_download; print(snapshot_download('$MODEL_ID'))" | tail -n 1)
    if [[ -z "$MODEL_ID" ]]; then
        echo "[MODEL] model download failed (empty path); check the error above" >&2
        exit 1
    fi
    echo "[MODEL] model path: $MODEL_ID"
fi

# ---------- already running? ----------
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "TGI already running (pid $(cat "$PID_FILE")). Use ./stop-tgi.sh first." >&2
    exit 1
fi
if curl -4s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/info" 2>/dev/null; then
    echo "Port $PORT already serves a TGI instance. Use ./stop-tgi.sh first." >&2
    exit 1
fi
rm -f "$PID_FILE"

# ---------- launch ----------
echo "[START] model_id=$MODEL_ID"
echo "[START] num_shard=$NUM_SHARD  devices=$DEVICES  port=$PORT"
echo "[START] max_total_tokens=$MAX_TOTAL_TOKENS  max_input_tokens=$MAX_INPUT_TOKENS"
echo "[START] log=$LOG_FILE"

ASCEND_VISIBLE_DEVICES="$DEVICES" nohup ./run-npu.sh \
    --model-id "$MODEL_ID" \
    --num-shard "$NUM_SHARD" \
    --port "$PORT" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "[START] launcher pid $(cat "$PID_FILE")"

# ---------- wait for readiness ----------
READY_TIMEOUT=300   # seconds; dual-card cold start takes ~75s
for ((i = 0; i < READY_TIMEOUT; i += 5)); do
    if curl -4s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/info" 2>/dev/null; then
        echo "[READY] TGI is serving on http://127.0.0.1:$PORT after ~${i}s"
        echo "[READY] stop with: ./stop-tgi.sh"
        exit 0
    fi
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[FAIL] launcher exited early. Log tail:" >&2
        tail -20 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 5
done

echo "[TIMEOUT] not ready after ${READY_TIMEOUT}s. Log tail:" >&2
tail -20 "$LOG_FILE" >&2
exit 1
