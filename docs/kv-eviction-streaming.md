# KV Cache 淘汰实验：sink + 滑动窗口

> 状态：**协议已冻结，尚未运行**。代码、CPU 测试和冻结协议已提交；还没有任何模型运行结果。结果出来前，本文不含质量或速度结论。

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

## 结果

尚未运行。
