#!/bin/bash
# Single-card baseline: start TGI on one NPU and answer one /generate
# request. Smallest end-to-end serving scenario.
#
# Usage: ./serve_basic.sh [start-tgi.sh options...]
#   Trailing options are forwarded to ./start-tgi.sh (--model-id, --port,
#   --devices, ...) and override this scenario's defaults (last wins).
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
# Default: one shard on NPU 0. start-tgi.sh applies options in order, so
# a trailing --num-shard/--devices/... from "$@" still wins.
./start-tgi.sh --num-shard 1 --devices 0 --port "$PORT" "$@"

tgi_info "$PORT"

INPUTS="The question is:"
REPLY="$(tgi_generate "$PORT" "$INPUTS" 8)"
[[ -n "$REPLY" ]] || { echo "[FAIL] empty reply" >&2; exit 1; }
echo "[PASS] single-card /generate OK: $REPLY"
