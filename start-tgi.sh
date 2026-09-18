#!/bin/bash
# Start the TGI inference service on Ascend NPU (background, multi-card via HCCL).
#
# Usage: ./start-tgi.sh [options] [-- extra launcher args...]
#
# Options:
#   --model-id PATH         model directory or HF id (default: /shared/models/Qwen3-0.6B)
#   --num-shard N           tensor parallelism degree (default: 2; 1 = single card)
#   --devices LIST          ASCEND_VISIBLE_DEVICES, e.g. "2,3" (default: "2,3")
#                           note: len(LIST) should be >= --num-shard
#   --port N                HTTP port (default: 3000)
#   --max-total-tokens N    (default: 128)
#   --max-input-tokens N    (default: 100)
#   --log FILE              log file (default: /tmp/tgi.log)
#   -h|--help               show this help
#
# Examples:
#   ./start-tgi.sh                                      # 2 shards on NPU 2,3
#   ./start-tgi.sh --num-shard 4 --devices 2,3,6,7      # 4 shards
#   ./start-tgi.sh --num-shard 1                        # single card
#   ./start-tgi.sh --model-id /shared/models/Qwen3-0.6B -- --json-output
#
# Stop with: ./stop-tgi.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- defaults ----------
MODEL_ID="/shared/models/Qwen3-0.6B"
NUM_SHARD=2
DEVICES="2,3"
PORT=3000
MAX_TOTAL_TOKENS=128
MAX_INPUT_TOKENS=100
LOG_FILE="/tmp/tgi.log"
PID_FILE="/tmp/tgi.pid"
EXTRA_ARGS=()

usage() {
    sed -n '2,20p' "$0"
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
