# KV Cache 淘汰实验（草稿）：sink + 滑动窗口

> 状态：**草稿**。只有淘汰策略本身和 CPU 单元测试；还没有接入 Qwen/MLX，没有任何模型运行结果。协议（`configs/kv-eviction-streaming-v0.json`）尚未冻结。

## 问题

长上下文问答里，KV Cache 随序列线性增长。StreamingLLM 观察到注意力会大量落在开头几个 token（attention sink）上，于是只保留"开头 sink + 最近窗口"。本实验要回答：在本仓库已有的 Qwen2.5-0.5B 文段问答负载上，按 50% / 25% 预算淘汰 KV，**省下多少 KV payload，问答质量掉多少**，是否值得采用。

## 已实现（本草稿）

- `lab/kv_eviction.py`：`StreamingPolicy`（sink、window、按比例算预算）、`evict()`（按 token 维度选取 K/V，不改原数组）、`kv_bytes()`（逻辑 payload 字节，不等于实测显存）。
- `tests/test_kv_eviction.py`：边界、索引、切片一致性、形状校验。

## 未决问题（冻结协议前必须定）

1. 位置编码：Qwen 的 key 写入缓存时已加 RoPE。淘汰后保留原位置（本草稿做法），还是像 StreamingLLM 那样按缓存内位置重编号？两者需要不同实现。
2. 淘汰时机：只在 prefill 后淘汰一次，还是 decode 过程中持续滑动。
3. 数据筛选：只用 ≥1024 token 的题目，筛选规则须在任何运行前固定。

## 边界

- 只在 Apple M4 Max / MLX 上执行，不代表 CUDA、Ascend、手机或服务端收益。
- `kv_bytes` 是逻辑字节；实测内存需另行记录。
- 不与本仓库其他实验的收益相乘或拼接。
