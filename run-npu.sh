#!/bin/bash
# Launch TGI on Ascend NPU. Environment discovery:
#   - conda: activate the env named by $TGI_CONDA_ENV if set; otherwise the
#     current environment is used as-is (it must provide python + torch_npu)
#   - CANN toolkit: $ASCEND_HOME_PATH if already set, else
#     $CONDA_PREFIX/Ascend/cann, else /usr/local/Ascend/ascend-toolkit
#   - launcher binaries: <repo>/target/release-opt
# Usage: ./run-npu.sh [launcher args...]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- conda env (optional) ----------
if [[ -n "${TGI_CONDA_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    elif [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    else
        echo "TGI_CONDA_ENV is set but conda is not available in this shell" >&2
        exit 1
    fi
    if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo "cannot locate conda base environment" >&2
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$TGI_CONDA_ENV"
fi

# ---------- CANN toolkit ----------
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ -f "$CONDA_PREFIX/Ascend/cann/set_env.sh" ]]; then
        ASCEND_HOME_PATH="$CONDA_PREFIX/Ascend/cann"
    elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
    fi
fi
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
fi

export PATH="$SCRIPT_DIR/target/release-opt:$PATH"

# 910B best default dtype; launcher picks these up via env
export ATTENTION="${ATTENTION:-flashdecoding-npu}"
export PREFIX_CACHING="${PREFIX_CACHING:-0}"
export CUDA_GRAPHS=0

# avoid NPU allocator fragmentation during tensor-parallel weight loading
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"

exec text-generation-launcher "$@"
