#!/bin/bash
# Stop the TGI inference service started by ./start-tgi.sh (or any instance
# launched by run-npu.sh on this host).
#
# Usage: ./stop-tgi.sh [--force]
#
# The launcher is sent SIGTERM first so it can shut down the shard processes
# gracefully. If leftover `text-generation` processes are still visible on the
# NPUs afterwards (e.g. the launcher was SIGKILLed earlier), a warning is
# printed; re-run with --force to kill them.
set -e

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

PID_FILE="/tmp/tgi.pid"

kill_proc() {
    local pid="$1"
    kill "$pid" 2>/dev/null || return 0
    # wait for graceful shutdown (launcher terminates shards itself)
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 2
    done
    echo "[STOP] pid $pid did not exit in 60s, sending SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
    return 0
}

stopped=0

# 1) launcher from pidfile
if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "[STOP] terminating launcher pid $pid (graceful)..."
        kill_proc "$pid"
        stopped=1
    fi
    rm -f "$PID_FILE"
fi

# 2) any other launcher on this host (e.g. started before start-tgi.sh existed)
launcher_pids=$(pgrep -f "text-generation-launcher" || true)
if [[ -n "$launcher_pids" ]]; then
    echo "[STOP] found launcher(s) without pidfile: $launcher_pids"
    for pid in $launcher_pids; do
        [[ "$pid" == "$$" ]] && continue
        kill_proc "$pid"
        stopped=1
    done
fi

# 3) verify nothing is left on the NPUs
sleep 3
leftovers=$(npu-smi info 2>/dev/null | grep -c "text-generation" || true)
if [[ "$leftovers" -gt 0 ]]; then
    echo "[WARN] $leftovers leftover text-generation process(es) still on NPUs:" >&2
    npu-smi info 2>/dev/null | grep -B1 "text-generation" || true
    if [[ "$FORCE" -eq 1 ]]; then
        echo "[STOP] --force: killing leftover server processes"
        pkill -TERM -f "text-generation-server" 2>/dev/null || true
        sleep 5
        pkill -KILL -f "text-generation-server" 2>/dev/null || true
    else
        echo "[WARN] run './stop-tgi.sh --force' to kill them" >&2
    fi
fi

if [[ "$stopped" -eq 1 ]]; then
    echo "[STOP] done."
else
    echo "[STOP] nothing was running."
fi
