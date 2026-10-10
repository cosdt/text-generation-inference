#!/bin/bash
# Dual-card tensor parallelism: start TGI with --num-shard 2 (HCCL
# weight sharding across two NPUs) and answer a /generate request.
# Requires at least 2 visible NPUs.
#
# Usage: ./serve_tensor_parallel.sh [start-tgi.sh options...]
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
# Default: two shards on NPUs 0,1. A trailing --num-shard/--devices/...
# from "$@" still wins (start-tgi.sh applies options in order).
./start-tgi.sh --num-shard 2 --devices 0,1 --port "$PORT" "$@"

tgi_info "$PORT"

INPUTS="The question is:"
REPLY="$(tgi_generate "$PORT" "$INPUTS" 8)"
[[ -n "$REPLY" ]] || { echo "[FAIL] empty reply" >&2; exit 1; }
echo "[PASS] tensor-parallel (2 shards) /generate OK: $REPLY"
