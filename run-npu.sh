#!/bin/bash
# Launch TGI on Ascend NPU using the local cann conda environment.
# Usage: ./run-npu.sh [launcher args...]
set -e

source /home/tangzizhao/miniconda3/etc/profile.d/conda.sh
conda activate cann

export PATH="/home/tangzizhao/workspace/tgi/target/release-opt:$PATH"

# 910B best default dtype; launcher picks these up via env
export ATTENTION="${ATTENTION:-flashdecoding-npu}"
export PREFIX_CACHING="${PREFIX_CACHING:-0}"
export CUDA_GRAPHS=0

exec text-generation-launcher "$@"
