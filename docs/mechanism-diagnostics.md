# 独立机制诊断与 Cache 收益边界

2026-10-08，Codex 辅助实现、执行与分析。旧实验、门槛、模型默认路径完全保留。

## Metal：一次明确干预，没有性能成功结论

旧 kernel 在写入各 SIMD group 的部分和之前，先清零共享数组并同步。
新候选只读取有效 group 的部分和，以零替代其他 lane，省去清零与一次 barrier。
线程数、每线程四元素、输出、精度与舍入保持不变；仍沿用 MLX 的归约结构与许可。
这一次干预同时改变 stores、同步和生成代码，不能独立测量 barrier 成本。

[协议](../configs/mechanism-diagnostics-v1.json)和实现先在 `afbde3e` 提交，再执行一次。
192 组检查覆盖两种精度、8 个宽度、3 个行数及 random/cancellation/zeros/strided。
候选与旧 custom kernel 输出逐元素相等，与 compiled native 的残差精确相等、归一化输出通过预先容差。
有限测试不构成所有输入证明，非有限或溢出输入不在契约中。

| 行数 × 896，FP16 | compiled μs | 原 custom μs | 新候选 μs | 原 custom / 新候选 |
|---|---:|---:|---:|---:|
| 1 | 104.167 | 101.708 | 103.396 | 0.984× |
| 64 | 105.875 | 103.625 | 103.708 | 0.999× |
| 512（事先主形状） | 103.729 | 102.771 | 102.792 | 1.000× |
| 2048 | 127.083 | 122.479 | 129.291 | 0.947× |

每格是 7 轮各 100 次调用中位数的中位数，随机臂顺序。预定速度门槛为 ≥1.05× 且 ≥6/7 轮更快；两种对照、所有形状均未通过。
计时包含 Python 调用、`mx.eval`、同步，排除输入创建及编译；不是纯 GPU 时间。
近似平坦的小形状耗时与调用/同步开销影响一致，但不能仅据此确认主要瓶颈。
没有 occupancy、DRAM 或寄存器实测，因此不能宣称根因已经闭环，也不能用本轮改写旧 23.8% 负结果。
候选没有接入模型、没有训练或生成质量验证，默认实现不变。

代码：[诊断 runner](../experiments/metal_mechanism_diagnostic.py)、[候选 kernel](../lab/kernels/residual_rmsnorm_masked.metal)。
记录：[逐次计时](../results/metal-mechanism-v1/timings.json)、[正确性](../results/metal-mechanism-v1/correctness.json)、[汇总](../results/metal-mechanism-v1/summary.json)、[来源](../results/metal-mechanism-v1/run.json)。

## Cache：访问局部性与容量共同决定收益

[独立协议](../configs/cache-workload-boundaries-v1.json)及最终 runner 在 `dde5b2d` 固定后执行一次。
固定本地 Qwen2.5-0.5B MLX Q8、M4 Max、串行 batch=1，每请求强制生成 8 token。
每条 trace 12 请求，每长度每模式 5 轮，两臂随机顺序，每臂每轮清空缓存。
对照是无快照的直接分段重算；不包括模型加载、分词或服务排队。

| 访问模式 | 命中 / 12 | 64-token：直接 / 缓存 | 1024-token：直接 / 缓存 |
|---|---:|---:|---:|
| 12 个不同前缀 | 0 | 0.992× | 0.999× |
| 4 前缀循环，容量 3 | 0 | 0.991× | 0.996× |
| 三组各连续访问 4 次 | 9 | 1.202× | 2.085× |
| 固定偏斜序列 | 6 | 1.125× | 1.552× |
| 3 前缀循环，字节只容 2 | 0 | 0.994× | 0.999× |

只在有命中的两种模式通过 ≥1.05× 且 ≥4/5 轮更快的预定门槛。零命中差异很小，不作显著退化结论。
600 对请求（共 1,200 条两臂输出）生成 token 相同；它们是固定合成 trace 的重复，不是 600 个独立业务样本。
计时使用每轮 12 个请求的 `total_seconds` 之和，再取 5 轮中位数；不是在线 P95/P99。
每个前缀的驻留逻辑 KV 字节为 `12288 × 前缀长度`，不是峰值 RSS。

均质成本近似可写为：缓存平均成本 `(1-h)M + hH`，直接重算成本 `D`。
若 miss 成本 M 大于 hit 成本 H，则需 `h > (M-D)/(M-H)` 才有收益。
这个关系解释为什么命中率和复制成本重要，但请求长度、内容、排队不同会破坏常成本假设；本轮没有估计通用阈值。
[单独 phase probes](../results/cache-workload-boundaries-v1/phase-probes.json)只作插桩诊断，不并入性能矩阵。

代码：[新 runner 与离线重算](../experiments/cache_workload_boundaries.py)。
记录：[固定 token 负载](../results/cache-workload-boundaries-v1/workloads.json)、[逐请求](../results/cache-workload-boundaries-v1/timings.json)、[汇总](../results/cache-workload-boundaries-v1/summary.json)。

## 离线验收

```bash
python -m experiments.metal_mechanism_diagnostic verify --output results/metal-mechanism-v1
python -m experiments.cache_workload_boundaries verify --output results/cache-workload-boundaries-v1
python -m unittest discover -s tests -v
```

验证历史字节、协议、来源归档、覆盖和算术；Linux CI 不执行 GPU，也不能独立证明测量环境没有干扰。
新硬件执行必须使用新目录，不能覆盖这两批记录。当前工作没有声称 CUDA/Ascend 实现、Attention 算子优化或生产部署。
