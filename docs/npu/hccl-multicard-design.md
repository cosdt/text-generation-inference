# TGI Ascend NPU 多卡推理（HCCL）适配设计文档

- 状态：已验证（2026-09-17 双卡/四卡端到端跑通，输出与单卡基线一致）
- 日期：2026-09-17
- 分支：`feat/ascend-npu`
- 前置：单卡 NPU 适配已合入（v3 后端端到端跑通，见 commit `287d261f`）

---

## 1. 背景与目标

单卡 NPU 推理已跑通（`flashdecoding-npu` attention + 纯 torch KV cache 写入）。本文档设计**多卡（张量并行，TP）推理**的适配方案。

**目标**（跑通优先，与单卡阶段策略一致）：

1. `--num-shard 2`（及 `--num-shard 4`）在 Ascend 910B 集群上通过 HCCL 跑通 v3 推理
2. 多卡输出与单卡**完全一致**（贪心解码对比）
3. 改动最小化，复用现有平台无关的多卡架构（router 多 shard 调度、协议层）

**硬件现状**（实测）：本机 4×910B4（`npu-smi` 设备号 2/3/6/7，各 32GB HBM），`npu-smi info -t topo` 显示 4 卡**全 HCCS 互联**（无 PCIe 中转路径）。

**软件现状**（cann conda 环境，实测版本）：
- torch 2.10.0+cpu、torch_npu 2.10.0.post5、CANN 25.5.1
- transformers 4.57.6（含 `tp_plan` 原生张量并行）、accelerate 1.15.0

---

## 2. TGI 多卡架构回顾（平台无关部分）

```
text-generation-launcher (Rust)
  ├─ spawn_shards(): 每 rank 一个线程 → shard_manager()
  │    ├─ UDS socket: {uds_path}-{rank}（每 shard 独立）
  │    ├─ world_size>1 时给 server 传 --sharded
  │    ├─ 注入 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
  │    └─ 全部 shard ready（socket 文件出现）后才启动 router
  ├─ text-generation-server × N（Python，每进程一卡，torch.distributed 通信）
  └─ text-generation-router（连接 N 个 shard 的 UDS，token 级调度与汇总）
```

关键结论（上一阶段已验证）：

- **协议层无 TP 概念**：router 多 shard 调度在 CUDA 上成熟运行，NPU 无需任何协议改动
- **权重切分在 Python 侧**：本 fork 的 v3 后端走 `TransformersFlashCausalLM`（`server/text_generation_server/models/transformers_flash_causal_lm.py`）
- **分布式初始化已有 npu 分支**：`server/text_generation_server/utils/dist.py:61-70,83-89`（hccl、1800s 超时、`torch.npu.set_device`）
- **launcher 已有 npu 分支**：`launcher/src/main.rs:1061`（注入 `HCCL_CONNECT_TIMEOUT`）、`resolve_attention()` npu 分支、`num_cuda_devices()` 的 `ASCEND_VISIBLE_DEVICES` 探测

---

## 3. 关键调研结论（实测证据）

多卡路径的选型取决于 `TransformersFlashCausalLM.__init__` 中的加载逻辑（`transformers_flash_causal_lm.py:142-151`）：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ...
    device_map=device if world_size == 1 else None,   # 多卡时不指定 device_map
    tp_plan="auto" if world_size > 1 else None,       # 多卡时启用 transformers 原生 TP
)
```

即：**多卡权重切分完全由 transformers 原生 TP（`tp_plan="auto"`）完成**，TGI 自身不参与切分（`Weights.get_sharded` 是 FlashCausalLM/custom modeling 路径用的，与本路径无关）。因此 NPU 多卡 = 验证 transformers 原生 TP 在 torch_npu + hccl 上的可行性。

### 3.1 transformers 原生 TP 机制（transformers 4.57.6）

1. `integrations/tensor_parallel.py::initialize_tensor_parallelism`：
   - `device_type = torch._C._get_accelerator().type` —— **torch_npu 2.10 下实测返回 `"npu"`**（torch 2.10 已泛化 accelerator API）
   - `backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}` —— **不含 npu**，但仅在 `torch.distributed` 未初始化时才会走到建 backend 分支；**TGI 的 `dist.py` 已先初始化 hccl，此分支被跳过**，无影响
   - `init_device_mesh(tp_device.type, (tp_size,))` → `init_device_mesh("npu", (2,))`
2. 模型权重按 plan 切成 DTensor：Qwen3 的 `base_model_tp_plan` 为 q/k/v/o/gate/up/down 投影的 colwise/rowwise（`PretrainedConfig.base_model_tp_plan`，本机 transformers 内置）
3. 通信由 DTensor 语义自动完成：colwise 输出 `Shard(-1)`，rowwise 输出经 allreduce 变 `Replicate`，前向/后向 hooks 由 `add_tensor_parallel_hooks_to_module`（`post_init`）挂载

### 3.2 NPU 可行性实测（2026-09-17，双进程 hccl）

| # | 验证项 | 结果 |
|---|---|---|
| 1 | `init_process_group(backend="hccl")` 双卡 | ✅ |
| 2 | `init_device_mesh("npu", (2,))` | ✅ `DeviceMesh((2,), 'npu')` |
| 3 | DTensor `from_local(Shard(0))` / `redistribute`（all_gather） | ✅ 跨卡数据正确 |
| 4 | mesh group `allreduce` | ✅ 结果 `[3.0, 3.0, 3.0, 3.0]` 正确 |
| 5 | `AutoModelForCausalLM.from_pretrained(attn_implementation="tgi", tp_plan="auto")` 双卡加载 Qwen3-0.6B | ✅ 权重正确切分：q_proj DTensor `Shard(0)` 全局 (2048,1024) 局部 (1024,1024)；o_proj `Shard(1)`；模块 `_hf_tp_plan` = colwise/rowwise |
| 6 | 双 shard Python server 手动启动（`text_generation_server.cli serve --sharded`） | ✅ 两进程均报 `Server started at unix:///tmp/tgi-2shard-test-{rank}`，UDS socket 均创建 |
| 7 | launcher 拉起双 shard（`--num-shard 2`） | ✅ 修复 LOCAL_RANK 注入（§4.2）后启动成功，2 个 shard ready |

### 3.3 验证项结果（2026-09-17 全部通过）

| # | 验证项 | 结果 |
|---|---|---|
| V1 | DTensor 输入进入 TGI attention 的表现 | ✅ 原生 DTensor dispatch 直接跑通，**无需实施 D1**（`to_local()` 留作性能优化选项） |
| V2 | 两 rank 输出 token 一致性 | ✅ `/generate` 贪心输出与单卡基线**逐字一致** |
| V3 | 4 卡（`--num-shard 4`） | ✅ 4 shard 各占一卡，贪心输出一致，4 并发通过 |

**新增发现**：多实例并行启动时若残留进程占用 NPU/端口，hccl 报 `EJ0003 Failed to bind the IP port`（error code 7）——环境清理（`pkill` 后确认 `npu-smi` 无残留进程）即可解决，非代码缺陷。

---

## 4. 总体方案

**主路径：复用 transformers 原生 TP，只修 gap。** 理由：

1. TGI 多卡代码路径（router 调度、协议、`--sharded`、heads 除法 bookkeeping）全平台无关，已在 CUDA 成熟
2. transformers 原生 TP 的加载/切分/通信机制已实测在 NPU 上工作（§3.2）
3. 备选路径（仿 FlashCausalLM 用 `Weights.get_sharded` 手动切分 + 手写 allreduce）工作量大（需为 Qwen3 写 custom modeling），仅在主路径被证伪时启用

### 4.1 改动清单（预估）

| # | 文件 | 改动 | 行数 |
|---|---|---|---|
| 1 | `launcher/src/main.rs` | `shard_manager()` 注入 `LOCAL_RANK`（transformers TP 需要；见 §4.2） | +2 |
| 2 | `server/.../transformers_flash_causal_lm.py` | `tgi_flash_attention_forward` 入口对 DTensor 输入取局部视图（§5.3 决策 D1，**阶段 2 实测后确认是否必需**） | +5 |
| 3 | （可能）`docs/` | 本文档 | — |

预计总改动 < 20 行。其余全部复用现有代码。

### 4.2 修复点 1：launcher 注入 LOCAL_RANK

**现象**：`transformers/integrations/tensor_parallel.py:80` 读取 `os.environ["LOCAL_RANK"]` 做 `current_device.set_device()`。TGI launcher 的 `shard_manager()`（`launcher/src/main.rs:1056-1060`）只注入 `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`，导致 shard 进程启动即崩：

```
KeyError: 'LOCAL_RANK'  (tensor_parallel.py:80)
Error: ShardCannotStart
```

**修复**：`shard_manager()` 中随分布式 env 一起注入：

```rust
envs.push(("LOCAL_RANK".into(), rank.to_string().into()));
```

> 注：TGI 进程模型是"每进程一卡、rank 即本地卡号"（`dist.py:65` `device = RANK % torch.npu.device_count()`），所以 `LOCAL_RANK = rank` 与语义一致。此改动对所有平台无害（transformers TP 路径同样需要，CUDA 上 TGI 未暴露此问题是因为 FlashCausalLM 路径不走 transformers TP）。

### 4.3 无需改动的部分（确认清单）

- `dist.py`：hccl 初始化、FakeGroup 单卡捷径均已就绪
- `launcher`：`num_cuda_devices()` 的 `ASCEND_VISIBLE_DEVICES` 分支、`find_num_shards()`、`compute_type()/vram()` 按卡数累乘（`main.rs:1823-1859`）、`resolve_attention()` npu 分支、`HCCL_CONNECT_TIMEOUT` 注入
- `weights.py` / router / 协议：平台无关
- `npu.py`：per-rank 实现（每 rank 只算自己的 heads），无跨卡假设
- 单卡回归：`WORLD_SIZE==1` 走 FakeGroup + `tp_plan=None`，行为完全不变

---

## 5. 关键设计决策

### D1：DTensor 输入在 tgi attention 入口取局部视图（预判，阶段 2 实证）

**问题**：TP 下 q/k/v 投影是 colwise，输出为 `DTensor(Shard(-1))`。这些张量会原样传入 `tgi_flash_attention_forward`（`transformers_flash_causal_lm.py:20`），继续流入 `kv_cache.store` 和 `npu.py` 的 einsum。

**分析**：
- 语义上，`Shard(-1)` 的局部张量就是本 rank 的 heads 切片，**与 TGI attention 的 per-rank 语义完全一致**（`transformers_flash_causal_lm.py:181-186` 已按 world_size 除 heads 做 KV cache bookkeeping）
- 若不加处理，DTensor dispatch 对 index_put/einsum 虽支持，但引入运行时布局推断与可能的非连续/广播开销（NPU 上尤其敏感）
- rowwise o_proj 的输入 hook（`tensor_parallel.py` RowwiseParallel._prepare_input_fn）对**普通张量**会自动 `DTensor.from_local(..., Shard(-1))` 包回——所以 attention 输出普通张量也不会破坏 TP 语义

**决策**：在 `tgi_flash_attention_forward` 入口检测 `isinstance(query_states, DTensor)`，若是则取 `.to_local()`（局部视图，零拷贝）。**实施与否以阶段 2 实测为准**：若原生 DTensor dispatch 能跑通且正确，先不加（跑通优先），性能优化阶段再加。

### D2：dtype 与 KV cache

- 每 rank 默认 bf16（`transformers_flash_causal_lm.py:120-122` 已有 npu 分支）
- KV cache 每 rank 存自己的 `num_kv_heads/world_size` 份（bookkeeping 已就绪）；router 下发的 block tables 各 rank 相同、各存各的 K/V，无需改动

### D3：设备可见性与 shard 数

- `ASCEND_VISIBLE_DEVICES=2,3` + `--num-shard 2`：launcher 探测链已支持（`main.rs:1303-1311`）
- 不设 `--num-shard` 而用 `--sharded`：launcher 自动按可见设备数推导（本机全可见时 4 卡）

### D4：hcccl 环境变量

- 已有：`HCCL_CONNECT_TIMEOUT=1800`（launcher 注入）
- 保留项（若多网卡环境出现连接问题再启用）：`HCCL_SOCKET_IFNAME`、`HCCL_SOCKET_TIMEOUT`

---

## 6. 分阶段实施计划

### 阶段 1：代码改动（约半小时）

1. `launcher/src/main.rs` 注入 `LOCAL_RANK`（§4.2）
2. `cargo build --profile release-opt`（launcher 二进制，构建命令沿用单卡阶段的环境要求）
3. 单卡回归：`--num-shard 1` 确认无回归

**验收**：单卡跑通；`--env` 探测输出正常。

### 阶段 2：双卡端到端验证（核心阶段）✅ 已完成

1. `ASCEND_VISIBLE_DEVICES=2,3 ./run-npu.sh --model-id /shared/models/Qwen3-0.6B --max-total-tokens 128 --max-input-tokens 100 --port 3000 --num-shard 2`
2. 验证矩阵：

| 场景 | 验收标准 |
|---|---|
| `/generate` 贪心 | 输出与单卡基线**逐 token 一致** |
| `/v1/chat/completions` | 同上 |
| 流式 SSE | 正常出 token |
| 4 并发 | 无死锁、无错 |
| `npu-smi info` | 两卡均有进程与显存占用（权重各 ~一半） |
| 日志 | 两 shard 均 `Server started at unix://...-{rank}`，无 hccl 报错 |

3. 观察 V1（DTensor 入 attention）：若有报错/异常慢 → 实施 D1 后重测

### 阶段 3：4 卡验证 ✅ 已完成

- `--num-shard 4`（ASCEND_VISIBLE_DEVICES=2,3,6,7）重复阶段 2 矩阵：贪心输出一致、4 并发通过、4 shard 各占一卡
- 各 shard 显存占用与 2 卡时同量级（每卡权重 ~1/4 与 ~1/2 差异在 0.6B 模型上不明显）

### 阶段 4：性能与后续优化（不阻塞跑通，单独排期）

1. D1 落地（DTensor 局部视图）—— 消除 dispatch 开销
2. `npu.py` 替换为 torch-npu 原生算子（`npu_prompt_flash_attention` / `npu_incre_flash_attention`）—— 单卡同受益
3. 通信优化：rowwise allreduce 与计算的流重叠（hccl 是否支持 P2P/stream 语义，需实测）
4. 大模型实测（≥7B，两卡放不下单卡时 TP 的真正价值场景），校准 launcher FLOPs/显存表

---

## 7. 风险与回退

| # | 风险 | 概率 | 影响 | 缓解/回退 |
|---|---|---|---|---|
| R1 | DTensor 入 tgi attention 后 index_put/einsum dispatch 失败或极慢 | 中 | 阶段 2 受阻 | 实施 D1（入口 to_local，改动 5 行）；仍不行则回退到"备选路径" |
| R2 | rowwise allreduce 后两 rank logits 不一致（hccl 数值非确定性） | 低 | 输出发散 | 贪心对比定位层；torch-npu 2.10 的 hccl allreduce 是确定性的（int/bf16 sum），预期无此问题 |
| R3 | barrier 死锁（权重加载时序差异） | 低 | 启动挂起 | `dist.py` 已设 1800s 超时 + launcher 有 shard 启动超时；HCcL/HCCS 全互联拓扑下预期稳定 |
| R4 | 备选路径：手动切分（`Weights.get_sharded` + custom modeling） | — | 大 | 仅在主路径证伪后启用；预估工作量：为 Qwen3 写 `flash_qwen3_modeling.py` 级代码（~800 行），不推荐 |
| R5 | 版本漂移：用户环境 transformers < 4.52（无 tp_plan） | 低 | 加载失败 | 文档记录最低版本要求；本机已固定 4.57.6 |

**回退总原则**：主路径失败只影响多卡，不影响已跑通的单卡（`WORLD_SIZE==1` 路径完全不变）；任何阶段失败均可回退到单卡继续使用。

---

## 8. 附：验证脚本与基线

- 测试脚本：`/home/tangzizhao/workspace/test_npu/tgi/test_generate.py`、`test_chat_completions.py`（单卡基线输出已留存于此前验证记录）
- 手动双 shard 启动命令（绕过 launcher，用于隔离问题）：

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 \
ATTENTION=flashdecoding-npu PREFIX_CACHING=0 CUDA_GRAPHS=0 ASCEND_VISIBLE_DEVICES=2,3 \
python -m text_generation_server.cli serve /shared/models/Qwen3-0.6B \
  --sharded --uds-path /tmp/tgi-2shard-test
```

（rank 1 同理；两进程均报 `Server started` 即 §3.2 #6 的验证方式）

- hccl/DTensor 冒烟脚本：`/tmp/hccl_tp_smoke.py`（已通过，见 §3.2）
