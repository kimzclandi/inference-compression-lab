# KV 淘汰实验 v2：StreamingLLM 对照 SnapKV（协议已冻结，尚未运行）

## 为什么有 v2

[v1](kv-eviction-streaming.md) 按冻结协议运行一次，门槛失败、不采用，结果原样保留。v1 暴露了两个问题：

1. **基线失效**：每题放入全部 9 段（约 1.3k token）后，完整缓存只答对 6/74，无法衡量淘汰的质量代价。
2. **方法只看位置**：sink + 滑窗按位置淘汰，答案所在文段容易被整段丢掉。

v2 是一个新实验，不是 v1 的重跑：缩短上下文、加入基线有效性门槛，并加入按问题注意力选择的 SnapKV。v1 的数字不被覆盖或替换。

## 冻结协议（`configs/kv-eviction-v2.json`）

| 项 | 设定 |
|---|---|
| 模型与版本 | 与 v1 相同的 Qwen2.5-0.5B Q8（SHA-256 锁定），MLX 0.29.3 / mlx-lm 0.26.3 |
| 数据 | 同 74 题；每题 = 自己的文段 + 2 段种子决定的干扰文段 |
| 条件 | full_cache；stream_50 / stream_25（sink=4）；snap_50 / snap_25 |
| SnapKV | 观察窗口 = prompt 最后 32 token（含问题）；窗口内 query 对更早 key 的注意力，在窗口与同组 query 头上求和，最大池化（核 7），每个 KV 头各自保留得分最高的位置，窗口始终保留；打分用 NumPy（`lab/snapkv.py`） |
| 淘汰时机 / 位置编码 | 与 v1 相同：prefill 后一次；保留原 RoPE 位置 |

**执行顺序与门槛**

1. 正确性：(a) 不淘汰路径与原生 MLX 逐 token 一致、logits ≤1e-3；(b) 开启 query 捕获时模型输出与关闭时一致。
2. 先只跑 full_cache；**可答题 EM < 40% 即判定基线无效并停止**，不报告任何淘汰对比。
3. 基线有效才跑 4 个淘汰条件和计时。
4. 采用门槛与 v1 相同，运行前冻结：可答题 EM 降幅 ≤2 点（33 题 = 不能净丢题）、总体 ≤2 点、逻辑 KV 减少 ≥40%、解码中位不慢且 ≥4/5 轮不慢。SnapKV 的打分耗时单独记录，不计入解码速度。

## 运行

```bash
git status --porcelain   # 必须为空
python -m unittest tests.test_mlx_kv_eviction -v
caffeinate -i python -m experiments.kv_eviction_v2 --model /path/to/verified-q8 --output runs/kv-eviction-v2
```

## 边界

- 只在 Apple Silicon / MLX 上执行；不代表 CUDA、Ascend、手机或服务端。
- SnapKV 是已发表方法（Li et al., 2024）的复现，不是原创算法。
- 逻辑 KV 是缓冲区大小，不是实测设备内存。
- 只运行一次；结果是正是负都原样记录。

## 结果

尚未运行。
