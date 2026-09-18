#!/bin/bash
# Launch TGI on Ascend NPU using the local tgi conda environment.
# Usage: ./run-npu.sh [launcher args...]
set -e

source /home/tangzizhao/miniconda3/etc/profile.d/conda.sh
conda activate tgi

# CANN 9.1.0 (installed inside the tgi env)
source /home/tangzizhao/miniconda3/envs/tgi/Ascend/cann/set_env.sh

export PATH="/home/tangzizhao/workspace/tgi/target/release-opt:$PATH"

# 910B best default dtype; launcher picks these up via env
export ATTENTION="${ATTENTION:-flashdecoding-npu}"
export PREFIX_CACHING="${PREFIX_CACHING:-0}"
export CUDA_GRAPHS=0

# avoid NPU allocator fragmentation during tensor-parallel weight loading
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"

exec text-generation-launcher "$@"
