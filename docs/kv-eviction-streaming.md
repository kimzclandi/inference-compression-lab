# KV Cache 淘汰实验：sink + 滑动窗口

> 状态：**已按冻结协议运行一次，两个淘汰条件都未通过采用门槛**。KV 与解码速度方面淘汰按预期生效；但长上下文基线本身接近失效，质量对比只能作弱证据。原始记录见 [`results/kv-eviction-streaming-v1/`](../results/kv-eviction-streaming-v1/)。

## 问题

长上下文问答里，KV Cache 随序列线性增长。StreamingLLM 观察到注意力会大量落在开头几个 token（attention sink）上，于是只保留"开头 sink + 最近窗口"。本实验回答：在 Qwen2.5-0.5B Q8 的文段问答上，按 50% / 25% 预算淘汰 KV，**省下多少 KV、问答质量掉多少**，是否值得采用。

## 冻结协议（`configs/kv-eviction-streaming-v1.json`）

| 项 | 设定 |
|---|---|
| 模型 | 与既有 Cache 实验相同的 Qwen2.5-0.5B Q8（文件 SHA-256 锁定），MLX 0.29.3 / mlx-lm 0.26.3 |
| 数据 | `configs/qwen-prefix/qa-dev.jsonl` 74 题（33 可答、41 不可答）；每题看到全部 9 段文段，自己的文段放在种子决定的位置 |
| 条件 | `full_cache`；`stream_50`（保留 50%，4 个 sink）；`stream_25`（保留 25%，4 个 sink） |
| 淘汰时机 | 完整 prompt prefill 后淘汰一次，再开始解码 |
| 位置编码 | 保留原位置：留下的 key 保持写入时的 RoPE，新 token 按真实位置旋转，不重编号 |
| 解码 | 贪心，最多 48 token，遇 EOS 停 |
| 指标 | 归一化 EM（总体 / 可答 / 不可答分开报）、逻辑 KV 字节、强制 32 token 的解码速度 |

**执行顺序**：来源与版本校验 → 正确性门槛（不淘汰时自写缓存路径必须与原生 MLX 贪心 token 完全一致、logits 在 1e-3 内）→ 质量（74 × 3）→ 计时（8 题，5 轮 × 2 次，条件顺序随机）→ 汇总。任何一步失败即停止并记录，不重试、不调参。

**采用门槛**（每个淘汰条件单独判断，跑结果前冻结，之后不改）：

- 可答题 EM 下降 ≤ 2 个百分点。可答只有 33 题，丢 1 题就是 3.03 点，即**不允许净丢题**；
- 总体 EM 下降 ≤ 2 个百分点；
- 逻辑 KV 字节减少 ≥ 40%；
- 解码速度中位数不慢于完整缓存，且 5 轮中至少 4 轮不慢。

## 代码

- `lab/kv_eviction.py`：淘汰策略（NumPy，CPU 可测）。
- `lab/long_context_qa.py`：长上下文 prompt 构造（确定性，不依赖分词器）。
- `lab/mlx_kv_eviction.py`：`EvictedKVCache`，物理长度短于 `offset`，只支持单 token 解码。
- `experiments/kv_eviction_streaming.py`：完整实验脚本。
- 测试：`tests/test_kv_eviction.py`、`tests/test_kv_eviction_study.py`（CPU），`tests/test_mlx_kv_eviction.py`（仅 Apple Silicon）。

## 运行（Apple Silicon Mac）

```bash
git status --porcelain   # 必须为空，脚本会检查
python -m unittest tests.test_mlx_kv_eviction -v
python -m experiments.kv_eviction_streaming --model /path/to/verified-q8 --output runs/kv-eviction-streaming-v1
```

## 边界与已知风险

- 只在 Apple Silicon / MLX 上执行；不代表 CUDA、Ascend、手机或服务端收益。
- KV 字节是缓冲区逻辑大小，不是实测设备内存。
- 滑动窗口很可能把答案所在文段淘汰掉；负结果是预期内的结果，会保留。
- 不可答题占 41/74；淘汰可能让模型更常输出 NO_ANSWER，从而抬高不可答题 EM，所以门槛以可答题为主。
- prompt 长度未在无分词器环境中核实，按题记录在 `prompts.json`。
- 不与本仓库其他实验的收益相乘或拼接。

## 结果（2026-10-10，Apple Silicon GPU，macOS 27.0，协议提交 d600792，工作区干净；本次收据未记录芯片型号）

正确性门槛：2 题不淘汰对齐，token 完全一致、logits 在 1e-3 内，通过。prompt 实际 1285–1308 token（中位 1292.5）。

| 条件 | 正确题数（74） | 可答 EM | 总体 EM | 逻辑 KV（每题均值） | 解码中位 tok/s | 相对完整缓存 |
|---|---|---|---|---|---|---|
| full_cache | 6 | 18.18% | 8.11% | 15.16 MiB | 285.5 | 1.00× |
| stream_50 | 3 | 9.09% | 4.05% | 7.58 MiB（−50.0%） | 316.2 | 1.11×（5/5 轮） |
| stream_25 | 2 | 6.06% | 2.70% | 3.79 MiB（−75.0%） | 325.3 | 1.14×（5/5 轮） |

逐题对比：stream_50 相对完整缓存丢 4 题、多对 1 题；stream_25 丢 5 题、多对 1 题。淘汰一次拷贝中位约 0.5 ms。

**按冻结门槛判定**：两个条件 KV 与速度门槛通过，可答题和总体 EM 门槛均失败，**不采用**。门槛与协议均未在结果后修改。

## 解读与限制

- **基线失效是本次最重要的发现。** 放入全部 9 段（约 1.3k token）后，完整缓存只答对 6/74：从不输出 NO_ANSWER（不可答题 0/41），且 32 条答案不是目标文段中的原文片段。这是协议设计问题：0.5B 模型撑不住该长上下文任务，质量对比落在地板附近，6→3→2 的差异只有几题，不能估计淘汰的真实质量代价。
- 解码加速来自每步注意力参与的 key 变少（约 1.3k → 650 / 325）；本实现每步仍做 concatenate 分配，加速数字只适用于本负载与本实现，不外推到服务端或其他硬件。
- 逻辑 KV 是缓冲区大小，不是实测设备内存。
- 只运行一次，没有重跑或调参。若继续，应另立 v2 协议（例如更大模型或更少干扰段，先确认完整缓存基线可用），不覆盖本次结果。

## 提交哈希说明（2026-10-10）

2026-10-10 维护者改写了这段提交历史，只删除提交信息里的工具署名行，文件内容逐字节不变。运行收据中记录的原协议提交与当前提交的对应关系：`d600792` → `c5e829d`（v1 协议）、`a22be82` → `a192bb4`（v1 结果）。收据原文保持不改。
