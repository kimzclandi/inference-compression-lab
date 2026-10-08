# 相同运算量下的 Metal 同步粒度诊断

2026-10-08，Codex 辅助实现、执行与分析。继续上一轮未通过速度门槛的 masked reduction，不再修改 kernel 参数。

## 先固定问题和协议

上一轮逐次 `eval + synchronize` 的小形状耗时接近，不能据此确认 GPU kernel 性能相同。
本轮在提交 `abcb6ff` 中先固定[协议](../configs/metal-sync-granularity-v1.json)和[runner](../experiments/metal_sync_granularity.py)，再执行一次。
每个样本均从相同输入开始，执行 **32 次有依赖的 residual-add + RMSNorm**；只改变每 1、8、32 次执行一次 eval/synchronize。
三臂为 compiled native、旧 custom、上一轮 masked 候选，3 种形状、7 轮、每格20个样本，随机化每轮9个格子的顺序。
两输出都强制求值，外层32次循环不整体compile；计时包含Python构图、GPU执行和同步，排除输入创建与预热。

27个终态正确性格子通过事先规定的原生容差；旧 custom 与 masked 在所有同步粒度下的终态输出哈希精确一致。
本轮不是新的全输入正确性证明；上一轮192个输入/形状检查仍保留。

## 已经确认：调用级计时对同步粒度高度敏感

下表是每次同步的32调用耗时，除以整组同步的32调用耗时；**不是 kernel 加速比或模型端到端加速比**。

| FP16，宽896 | compiled native | 旧 custom | masked |
|---|---:|---:|---:|
| 1行 | 12.082× | 12.600× | 12.177× |
| 512行 | 9.181× | 8.249× | 8.317× |
| 2048行 | 4.780× | 5.332× | 5.532× |

所有格子均通过事先规定的同步敏感性规则（比例≥1.1且≥6/7轮整组更快）。
同步粒度同时改变构图/提交、调度和存活输出，不能把两种耗时直接相减称为“纯CPU开销”。
这支持改进测量解释，而不是推翻冻结 kernel 研究或改成功门槛。

整组同步时，32次调用的中位耗时如下：

| 形状 | compiled native | 旧 custom | masked |
|---|---:|---:|---:|
| 1×896 | 0.312 ms | 0.308 ms | 0.313 ms |
| 512×896 | 0.441 ms | 0.481 ms | 0.487 ms |
| 2048×896 | 0.942 ms | 0.882 ms | 0.873 ms |

相对排序随形状和同步方式变化。2048行处 masked 比 compiled 的描述性比例约1.079×，但512行处约0.906×；masked相对旧custom约0.989×和1.010×。
协议没有注册新 kernel 的采用门槛，不以选取有利格子宣称优化成功；候选依然不接入模型。

记录：[逐样本计时](../results/metal-sync-granularity-v1/timings.json)、[正确性与输出哈希](../results/metal-sync-granularity-v1/correctness.json)、[汇总](../results/metal-sync-granularity-v1/summary.json)、[源码快照与来源](../results/metal-sync-granularity-v1/run.json)。

## 新采集的 Instruments 证据

在同一固定协议下，各臂执行300批×32调用、另有3批预热，形状2048×896。
每臂只采集一次，启动等待10秒后运行，进程正常退出，回执确认9,600个正式调用完成。
公开CSV按**精确PID**过滤，包含该进程初始化、预热和正式工作，不包含其他进程。
原始trace与完整XML保留本机，公开其哈希和数值投影；哈希并不能独立认证采集环境。

| 臂 | GPU Compute intervals | 区间时长之和 | 首尾区间跨度 | 区间覆盖比例 |
|---|---:|---:|---:|---:|
| compiled | 914 | 320.121 ms | 429.518 ms | 74.53% |
| original | 912 | 251.291 ms | 356.493 ms | 70.49% |
| masked | 913 | 302.531 ms | 413.358 ms | 73.19% |

9,600次调用不对应9,600个interval。覆盖比例是归属此进程的Compute区间并集/首尾跨度，**不是GPU利用率**；间隙不能全部归因于CPU或GPU空闲。
单次插桩下masked的区间时长高于original，与未插桩整组计时的小差异不能拼接为同一性能结论。
没有注册trace时长胜负门槛，不据此重新选择候选；需保留插桩扰动、初始化、调度和运行差异的解释空间。

本轮compiled trace的[计数器元数据](../results/metal-sync-trace-v1/compiled-counter-info.xml)仍只列出 **RT Unit Active**，未获得DRAM带宽、occupancy或寄存器压力实测。
因此已确定“测量对同步粒度敏感”，**仍未确定旧kernel负结果的底层硬件根因**。

实现：[区间并集与间隙算法](../lab/interval_coverage.py)、[提取和验证](../experiments/metal_sync_trace.py)。
证据：[回执与汇总](../results/metal-sync-trace-v1/receipt.json)、[全部进程过滤CSV](../results/metal-sync-trace-v1/)。

## 离线复核

```bash
python -m experiments.metal_sync_granularity verify --output results/metal-sync-granularity-v1
python -m experiments.metal_sync_trace verify --output results/metal-sync-trace-v1
python -m unittest discover -s tests -v
```

验证覆盖、来源归档、原始计时算术、PID过滤、区间并集和计数器名称，不会执行GPU或重跑性能。
Linux CI不证明模型质量、跨设备加速或用户已独立掌握；本轮没有CUDA/Ascend实现、生产部署或简历新加速表述。
