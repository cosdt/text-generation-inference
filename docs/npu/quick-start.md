# 昇腾 NPU 快速上手

在昇腾 NPU 上配置、构建并运行 Text Generation Inference（TGI）推理服务，
支持单卡与多卡 HCCL 张量并行。本文档面向使用者，覆盖 Docker 与源码构建
两条路径，并给出可直接跑通推理的 demo。

昇腾支持以 release 形式发布在
[cosdt/text-generation-inference](https://github.com/cosdt/text-generation-inference)
（TGI 官方仓库的 fork；官方 release 目前不含 NPU 支持）。

## 支持范围

| 项 | 内容 |
| --- | --- |
| 硬件 | Atlas 900 A2 PODc（Ascend 910B） |
| 后端 | v3（默认） |
| dtype | bfloat16 |
| 并行 | 单卡；多卡 HCCL 张量并行（`--num-shard N`，2/4 卡已验证） |
| 模型 | 文本生成模型（Qwen 系列等）；量化模型不支持（见[已知限制](#已知限制)） |

**验证过的软件栈**（Docker 镜像与看护 CI 均按此版本构建/测试）：

| 组件 | 版本 |
| --- | --- |
| CANN | 9.1.0 |
| Python | 3.12 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |
| transformers / accelerate | 4.57.6 / 1.15.0 |
| kernels | 0.5.0（必须固定，见[故障排查](#故障排查)） |
| 模型示例 | [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) |

## 前置条件

无论走哪条路径，机器上需要先有：

- **昇腾硬件 + 驱动**：`npu-smi info` 能列出设备（[昇腾快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- **CANN 9.1.0 工具包**（Docker 路径由镜像自带，无需单独安装）
- 若走源码构建：Python 3.12、`torch`/`torch_npu`（按[兼容矩阵](https://gitcode.com/Ascend/pytorch)与 CANN 匹配）、Rust 1.85.1

验证环境是否就绪：

```shell
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())"
```

## 方式一：Docker（推荐）

### 构建镜像

910B 主机是 aarch64 架构，**请在昇腾机器上构建**（cargo 编译产物与架构绑定）。

```shell
# 默认基础镜像：华为云 SWR ascendhub，CANN 9.1.0 + Python 3.12
docker build -f Dockerfile_ascend -t tgi-ascend:3.3.7-npu .

# 如需换基础镜像 / torch 栈（版本必须与 CANN 兼容）：
docker build -f Dockerfile_ascend -t tgi-ascend:custom \
    --build-arg CANN_IMAGE=ascendai/cann:<tag>-910b-ubuntu22.04-py3.12 \
    --build-arg TORCH_VERSION=2.9.0 --build-arg TORCH_NPU_VERSION=2.9.0.post2 .
```

镜像内已预设好 NPU 所需的运行环境变量（`ATTENTION=flashdecoding-npu`、
`PREFIX_CACHING=0`、`CUDA_GRAPHS=0` 等，见[配置说明](#配置说明)），启动时无需再传。

### 启动服务

```shell
model=Qwen/Qwen3-0.6B        # 本地目录或 HF/ModelScope 模型 id
volume=$PWD/data             # 模型与缓存卷，避免每次启动重新下载

docker run --rm -it \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    --shm-size 64g -p 8080:80 -v $volume:/data \
    tgi-ascend:3.3.7-npu --model-id $model
```

多卡张量并行（两张卡）：

```shell
docker run --rm -it \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    --shm-size 64g -p 8080:80 -v $volume:/data \
    tgi-ascend:3.3.7-npu --model-id $model --num-shard 2
```

### 发起推理

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
    -H 'Content-Type: application/json'
```

OpenAI 兼容接口：

```shell
curl 127.0.0.1:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [{"role": "user", "content": "What is 1+1?"}],
  "max_tokens": 16
}' \
    -H 'Content-Type: application/json'
```

## 方式二：源码构建

### 安装系统依赖与 Python 依赖

```shell
sudo apt-get update
sudo apt-get install -y build-essential protobuf-compiler pkg-config libssl-dev curl git python3-dev

python -m pip install -q uv
# torch / torch_npu 必须从华为昇腾源安装（与 CANN 9.1.0 匹配的 2.9 系列）
uv pip install \
  --index-url https://repo.huaweicloud.com/ascend/repos/pypi/simple \
  "torch==2.9.0" "torch_npu==2.9.0.post2"
uv pip install \
  "transformers==4.57.6" "accelerate==1.15.0" "modelscope==1.37.0" \
  "kernels==0.5.0" "grpcio-tools>=1.69.0" "mypy-protobuf>=3.6.0"
```

### 安装 Rust 工具链

```shell
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl -fsSL https://rsproxy.cn/rustup-init.sh -o /tmp/rustup-init.sh
sh /tmp/rustup-init.sh -y --default-toolchain 1.85.1 --profile minimal
mkdir -p "$HOME/.cargo"
printf '[source.crates-io]\nreplace-with = "rsproxy-sparse"\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n' \
  > "$HOME/.cargo/config.toml"
export PATH="$HOME/.cargo/bin:$PATH"
```

（rsproxy.cn 镜像用于规避国内访问 static.rust-lang.org / crates.io 的不稳定；
`rust-toolchain.toml` 固定 1.85.1。）

### 克隆、编译并安装

```shell
git clone --depth 1 --branch v3.3.7-npu \
  https://github.com/cosdt/text-generation-inference.git
cd text-generation-inference

export PYO3_PYTHON="$(command -v python)"   # PyO3 嵌入 Python 所需
export PROTOC="$(command -v protoc)"        # protobuf 代码生成所需
cargo build --profile release-opt \
  -p text-generation-launcher -p text-generation-router-v3

uv pip install --no-build-isolation -e server
make -C server gen-server-raw
```

## 推理 demo（一键跑通）

源码构建完成后，仓库自带三个脚本（含自动下载模型、就绪等待与残留进程清理）：

```shell
# 双卡 HCCL 张量并行（默认 NPU 0,1；本地无 conda 环境名要求，
# 若需指定 conda 环境：export TGI_CONDA_ENV=<env-name>）
./start-tgi.sh

# 单卡：
./start-tgi.sh --num-shard 1

# 四卡 / 自定义设备：
./start-tgi.sh --num-shard 4 --devices 0,1,2,3
```

`--model-id` 默认 `Qwen/Qwen3-0.6B`，首次启动自动经 ModelScope 下载
（约 1.2 GB，缓存于 `~/.cache/modelscope`，之后直接命中缓存；若环境里没有
modelscope 会先自动安装）。也可直接传本地路径：`--model-id /path/to/model`。

服务就绪后发起推理：

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
    -H 'Content-Type: application/json'
```

响应示例（`do_sample=false` 贪心解码，输出确定；单卡/双卡结果逐字一致）：

```json
{"generated_text":" 2\n\nThe question is: What is the sum of the numbers 1"}
```

> Qwen3 是 thinking 模型，`/v1/chat/completions` 的回复内容会包含 `<think>...</think>`
> 推理过程，属正常现象；`/generate` 同样可用。

停止服务（优雅退出并检查 NPU 上无残留进程）：

```shell
./stop-tgi.sh
# 如仍有残留进程：./stop-tgi.sh --force
```

## 多卡 HCCL 张量并行

- `--num-shard N` 指定张量并行度；启动前用 `ASCEND_VISIBLE_DEVICES` 指定可见卡，
  `start-tgi.sh` 的 `--devices` 即设置该变量。
- 多卡走 transformers 原生 TP（`tp_plan`），launcher 会自动为每个 shard 注入
  `LOCAL_RANK`——需要 **v3.3.7-npu 及之后的 release**（官方上游没有这段代码）。
- 贪心解码（`do_sample=false`）下多卡输出与单卡逐字一致，可用此特性做正确性校验。
- 实现细节与验证记录见 [hccl-multicard-design.md](./hccl-multicard-design.md)。

## 配置说明

`run-npu.sh` 与 Docker 镜像会预设以下环境变量（源码直启时需自行 export）：

| 变量 | 值 | 作用 |
| --- | --- | --- |
| `ATTENTION` | `flashdecoding-npu` | NPU attention 实现（torch-npu）；launcher 探测到 NPU 时也会自动选择它 |
| `PREFIX_CACHING` | `0` | NPU 暂不支持 prefix caching，必须关闭 |
| `CUDA_GRAPHS` | `0` | 不使用 CUDA 图 |
| `PYTORCH_NPU_ALLOC_CONF` | `max_split_size_mb:256` | 缓解张量并行权重加载时的 NPU 显存碎片 |
| `ASCEND_VISIBLE_DEVICES` | `0,1,...` | 可见 NPU 设备号 |
| `HCCL_CONNECT_TIMEOUT` | `1800`（Docker 镜像内） | 多卡 HCCL 组网超时 |

## 已知限制

- **量化不支持**：AWQ / GPTQ / Marlin / EETQ / FP8 / compressed-tensors 等量化路径
  在 NPU 上不可用（加载量化 checkpoint 会直接报错）。
- **v2 后端未适配**，请使用默认的 v3 后端。
- **prefix caching 关闭**，相关参数（`PREFIX_CACHING=1`）在 NPU 上无效。
- 滑动窗口 attention（Mistral 等）无专用 kernel，走通用路径。
- 不安装 triton（无 NPU 版）；代码已做缺失容错。

## 故障排查

### 启动报 `EJ0003 Failed to bind the IP port`

上一实例的 shard 进程残留，占用了 NPU 或端口。清理后重启：

```shell
./stop-tgi.sh --force
# 或：pkill -f text-generation-server
npu-smi info   # 确认进程列表里没有 text-generation 残留
```

### `kernels` 构建失败（找不到 `kernels.lockfile`）

TGI Python server 的构建后端依赖 `kernels.lockfile`，新版 `kernels` 删除了该文件。
必须固定 `kernels==0.5.0`，且 server 安装使用 `--no-build-isolation`（构建隔离会
重新解析出最新版 kernels）。requirements_ascend.txt 已固定此版本。

### launcher 启动后很快退出

查看日志尾行定位原因（start-tgi.sh 默认写 `/tmp/tgi.log`）：

```shell
tail -50 /tmp/tgi.log
```

常见：模型路径不存在、`import torch_npu` 失败（CANN 与 torch_npu 版本不匹配）、
多卡时卡数不足（`--num-shard` 大于可见卡数）。

### curl 连接被拒

本机 IPv6 环境下 curl 可能优先解析 `::1`，用 `-4` 强制 IPv4：
`curl -4 127.0.0.1:8080/info`。

### 多卡输出与单卡不一致

确认使用 v3.3.7-npu 及之后的版本（多卡需要 launcher 注入 `LOCAL_RANK`）；
贪心解码下输出应当是逐字一致的。
