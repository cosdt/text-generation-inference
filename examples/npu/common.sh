#!/bin/bash
# Shared helpers for the Ascend NPU serving examples (examples/npu/).
# Sourced by the scenario scripts; do not execute directly.
#
# Helpers (each fails the calling script on HTTP error, malformed
# response, or missing expected field):
#   tgi_info <port>                       print /info summary line
#   tgi_generate <port> <inputs> [max_new]
#                                         print the generated text on stdout
#   tgi_generate_stream <port> <inputs> [max_new]
#                                         print the final text on stdout,
#                                         token-event count on stderr

python3_here() {
    if command -v python3 >/dev/null 2>&1; then
        echo python3
    else
        echo python
    fi
}

PY="$(python3_here)"

# Build the JSON request payload with python: the inputs string may
# contain quotes or backslashes that a printf '%s' template would break.
build_payload() {
    INPUTS="$1" MAX_NEW="$2" "$PY" -c '
import json, os
print(json.dumps({
    "inputs": os.environ["INPUTS"],
    "parameters": {"max_new_tokens": int(os.environ["MAX_NEW"])},
}))
'
}

tgi_info() {
    local port="$1"
    curl -4sSf --max-time 10 "http://127.0.0.1:${port}/info" | "$PY" -c '
import json, sys
info = json.load(sys.stdin)
print("model_id:", info.get("model_id"), "| version:", info.get("version"))
'
}

tgi_generate() {
    local port="$1" inputs="$2" max_new="${3:-8}"
    curl -4sSf --max-time 120 -X POST "http://127.0.0.1:${port}/generate" \
        -H 'Content-Type: application/json' \
        -d "$(build_payload "$inputs" "$max_new")" | "$PY" -c '
import json, sys
out = json.load(sys.stdin)
text = out.get("generated_text")
if not text:
    raise SystemExit(1)
print(text)
'
}

tgi_generate_stream() {
    local port="$1" inputs="$2" max_new="${3:-8}"
    curl -4sSfN --max-time 120 -X POST \
        "http://127.0.0.1:${port}/generate_stream" \
        -H 'Content-Type: application/json' \
        -d "$(build_payload "$inputs" "$max_new")" | "$PY" -c '
import json, sys
# SSE stream: one "data: <json>" event per token. Intermediate events
# carry "token" and generated_text=null; the final event carries the
# full generated_text (and details). Keepalive lines start with ":".
tokens = 0
final = None
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line.startswith("data:"):
        continue
    payload = line[len("data:"):].strip()
    if not payload:
        continue
    ev = json.loads(payload)
    if ev.get("token") is not None:
        tokens += 1
    if ev.get("generated_text") is not None:
        final = ev["generated_text"]
if tokens == 0 or final is None:
    raise SystemExit(1)
print(f"tokens={tokens}", file=sys.stderr)
print(final)
'
}
