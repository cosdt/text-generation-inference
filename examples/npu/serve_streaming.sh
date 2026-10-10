#!/bin/bash
# SSE streaming: serve /generate_stream on one NPU and verify that
# (1) at least one incremental token event arrives, and (2) the final
# streamed text equals the plain /generate reply for the same prompt
# (both are greedy by default, so the two must match exactly).
#
# Usage: ./serve_streaming.sh [start-tgi.sh options...]
#   Trailing options are forwarded to ./start-tgi.sh and override this
#   scenario's defaults (last wins).
#   Env: PORT (default 8080), CI_OUTPUT_DIR (output dir; default /tmp).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/common.sh"

PORT="${PORT:-8080}"
OUT_DIR="${CI_OUTPUT_DIR:-/tmp/tgi-example-out}"
mkdir -p "$OUT_DIR"

cleanup() { "$ROOT/stop-tgi.sh" >/dev/null 2>&1 || true; }
trap cleanup EXIT

cd "$ROOT"
# Default: one shard on NPU 0.
./start-tgi.sh --num-shard 1 --devices 0 --port "$PORT" "$@"

tgi_info "$PORT"

INPUTS="The question is:"
# tgi_generate_stream prints the token-event count on stderr and the
# final text on stdout; the token-count check is the streaming assertion.
STREAM_TEXT="$(tgi_generate_stream "$PORT" "$INPUTS" 8)"
REPLY="$(tgi_generate "$PORT" "$INPUTS" 8)"
[[ "$STREAM_TEXT" == "$REPLY" ]] || {
    echo "[FAIL] streamed text != non-streamed text" >&2
    echo "stream: $STREAM_TEXT" >&2
    echo "plain:  $REPLY" >&2
    exit 1
}
echo "[PASS] /generate_stream OK, final text matches non-streaming reply: $STREAM_TEXT"
