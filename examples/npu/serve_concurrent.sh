#!/bin/bash
# Concurrent requests: serve on one NPU and fire NUM_REQUESTS /generate
# requests in parallel. TGI continuously batches them on the same model;
# every request must return a non-empty reply. Exercises the scheduler
# under concurrent load rather than a single serial request.
#
# Usage: ./serve_concurrent.sh [start-tgi.sh options...]
#   Trailing options are forwarded to ./start-tgi.sh and override this
#   scenario's defaults (last wins).
#   Env: PORT (default 8080), NUM_REQUESTS (default 4),
#        CI_OUTPUT_DIR (output dir; default /tmp).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/common.sh"

PORT="${PORT:-8080}"
NUM_REQUESTS="${NUM_REQUESTS:-4}"
OUT_DIR="${CI_OUTPUT_DIR:-/tmp/tgi-example-out}/concurrent"
mkdir -p "$OUT_DIR"

cleanup() { "$ROOT/stop-tgi.sh" >/dev/null 2>&1 || true; }
trap cleanup EXIT

cd "$ROOT"
# Default: one shard on NPU 0.
./start-tgi.sh --num-shard 1 --devices 0 --port "$PORT" "$@"

tgi_info "$PORT"

pids=()
for ((i = 1; i <= NUM_REQUESTS; i++)); do
    tgi_generate "$PORT" "The question is: #$i" 8 > "$OUT_DIR/resp_$i.txt" &
    pids+=("$!")
done

ok=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        ok=$((ok + 1))
    fi
done
for ((i = 1; i <= NUM_REQUESTS; i++)); do
    [[ -s "$OUT_DIR/resp_$i.txt" ]] || {
        echo "[FAIL] request $i produced no reply" >&2
        exit 1
    }
done

echo "[PASS] concurrent: $ok/$NUM_REQUESTS parallel requests served"
