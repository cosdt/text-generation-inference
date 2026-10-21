# TGI Huawei Ascend NPU serving examples

End-to-end serving scenarios on Huawei Ascend NPU. Each script starts a real
TGI service (via the repo-root [`start-tgi.sh`](../../start-tgi.sh)),
sends requests, asserts the replies, and tears the service down
([`stop-tgi.sh`](../../stop-tgi.sh)). Exit code 0 means the scenario
passed.

| Script | Scenario | NPUs |
| --- | --- | --- |
| `serve_basic.sh` | Single-card `/generate` baseline | 1 |
| `serve_tensor_parallel.sh` | Dual-card tensor parallelism (`--num-shard 2`, HCCL) | 2 |
| `serve_streaming.sh` | SSE `/generate_stream`, final text must match the non-streaming reply | 1 |
| `serve_concurrent.sh` | `NUM_REQUESTS` parallel `/generate` requests (continuous batching) | 1 |

## Prerequisites

Build once and prepare the environment as described in
[docs/npu/quick-start.md](../../docs/npu/quick-start.md) (CANN toolkit,
`torch_npu`, launcher + router + server binaries in `target/release-opt`
or on `PATH`).

## Usage

```bash
# default model: Qwen/Qwen3-0.6B, downloaded via ModelScope on first
# use and cached in ~/.cache/modelscope
examples/npu/serve_basic.sh

# override model / port / devices; any start-tgi.sh option is forwarded
examples/npu/serve_tensor_parallel.sh --model-id Qwen/Qwen3-0.6B --port 8090
```

Env knobs: `PORT` (default 8080), `CI_OUTPUT_DIR` (output dir,
default `/tmp/tgi-example-out`), `NUM_REQUESTS` (concurrent only).

These scripts are watched by the Huawei Ascend docs guard (examples guard
engine, `examples_manifest.yaml`); each scenario maps to one manifest
entry. Add a new scenario by dropping a `.sh` here and classifying it in
that manifest.
