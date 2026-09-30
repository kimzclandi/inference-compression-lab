# MiniLM / M4 Max CPU 动态 INT8 与逐层误差实验

2026-09-30，真实本地执行。结论：本负载 INT8 文件更小，但推理慢于 FP32；一次按局部误差选择的浮点回退让句向量 MSE 降低 6.78%，没有充分证据说明任务质量显著提升。结果不代表 Windows、Jetson、Apple GPU、Core ML 或华为 NPU。

## 1. 环境和同源验证

Apple M4 Max，16 个逻辑 CPU，48 GiB，macOS 27.0 (26A428)，arm64；CPython 3.12.13；ORT 1.30.0；CPUExecutionProvider，intra-op=4、inter-op=1、顺序执行、ORT_ENABLE_ALL。保留 Windows 全部历史文件，哈希见 [historical-integrity.json](historical-integrity.json)。

[provenance.json](provenance.json) 中模型、tokenizer、config 和 STS-B parquet 的 revision/哈希与 Windows 历史文件全部一致。作者发布的 ONNX 文件被直接使用，没有自行从 PyTorch 导出。

- 模型：`sentence-transformers/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。
- STS-B：`sentence-transformers/stsb`，revision `ab7a5ac0e35aa22088bdcf23e7fd99b220e53308`，完整 validation 1,500 对；最长 256、batch=16、mask mean pooling、L2 归一化、Spearman。
- 量化：ORT 官方 `quantize_dynamic`，动态 U8 激活 / signed INT8 权重，只处理常量权重 MatMul，reduce_range=False。不是全模型 W8A8。
- 七个原有直接依赖版本不变；完整 Mac 依赖锁在 [requirements-macos-py312.txt](../../requirements-macos-py312.txt)。Python 3.14 的 PyArrow 源码包路径未采用。

## 2. 同协议 Mac 基线

batch=1、padding 长度 64、validation 首句；只计 `session.run`，不含分词、池化、模型加载。独立 worker；每轮预热 50、测量 200、3 轮；下表为三轮中位数的中位数。按固定 FP32 → 按张量 → 按通道顺序测量。

| 配置 | Spearman | ONNX MiB | 延迟 ms | FP32/本配置延迟 |
|---|---:|---:|---:|---:|
| FP32 | 0.867097776 | 86.22 | 2.019375 | 1.000× |
| 动态 INT8 按张量 | 0.864547651 | 55.94 | 2.277334 | 0.887× |
| 动态 INT8 按通道 | 0.865713899 | 56.04 | 2.274750 | 0.888× |

原始 600 个耗时/配置、每轮 P95、进程内存、模型 SHA256 在各配置 JSON；[summary.json](summary.json) 是摘要。按通道文件约缩小 35.0%，不代表运行内存缩小。Mac 与 Windows 数字分目录保存，不能混合成同一组重复测量，也不能仅凭这两个环境归因硬件差异。

## 3. 真实整数执行与负结果

两种量化图均含 36 个 MatMulInteger，保留 12 个浮点 MatMul；profiler 各记录到 CPU 上 18 个 DynamicQuantizeMatMul、18 个 MatMulIntegerToFloat 和 6 个 DynamicQuantizeLinear 事件。全部实际 session 仅含 CPUExecutionProvider。执行证据见 `*-execution.json`；这些是 ORT 算子级证据，不是硬件指令跟踪。

该结果与“更低位宽必定更快”相矛盾。动态范围计算、重缩放/算子调度开销，以及此形状下 FP32 内核效率，均是可能原因；尚未通过控制变量逐项归因。ORT 文档指出动态量化计算激活参数会增加推理开销，不能据此认定已经定位本机的主因。[ORT 官方量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

为检查固定顺序偏差，另做四配置循环轮换，每配置处于四个位置各一次；每次用新进程，50 预热 + 200 样本，4 轮。不把它并入上述 3 轮基线，也不假装随机化。

| 配置 | 轮换实验 ms | 相对 FP32 延迟变化 |
|---|---:|---:|
| FP32 | 1.989406 | +0.00% |
| 动态 INT8 按张量 | 2.276375 | +14.42% |
| 动态 INT8 按通道 | 2.277094 | +14.46% |
| 按通道 + 单 MatMul 保留 FP32 | 2.272167 | +14.21% |

[轮换协议和 3,200 个原始耗时](../minilm-mac-m4max-latency-crossover/) 仍支持本负载下 INT8 较慢。日常笔记本后台、温度、频率和 P/E 核分配未严格控制，未宣称普遍 CPU 结论。

## 4. 逐层误差：累计与局部必须分开

探针是 validation 前 64 对的 128 句，固定顺序、batch=16，共 8 个 batch；不是独立校准/盲测集。只统计非 padding token。36 个常量权重 MatMul、6 个 Transformer block 输出用于累计比较；另对 36 个 MatMul 使用同一 FP32 输入进行隔离的实际动态量化执行，测局部误差。

`NMSE = sum((FP32 - candidate)^2) / sum(FP32^2)`；跨 batch 累计分子分母。局部 MatMul 的量化权重、scale、zero point 全部与完整按通道模型逐项精确匹配，见 [核验记录](../minilm-mac-m4max-layer-errors-optimized/local-weight-verification.json)。

发现并保留了一次诊断偏差：最初关闭优化的插桩图，其 INT8 最终输出相对实际优化图的最大 batch NMSE 分别达 0.017316（按张量）和 0.015131（按通道）。[原始诊断记录](../minilm-mac-m4max-layer-errors/) 仅代表该未优化路径，不用于描述实际优化路径的累计误差。尚未将差异归因到单个融合算子或 CPU 指令。

改用 ORT_ENABLE_ALL 后，8 个 batch 中两种 INT8 插桩图最终输出均与原模型完全相同；FP32 最终输出 NMSE 最大 8.33e-13。此核验约束最终输出漂移，不宣称所有中间执行策略完全一样。正式分析使用 [优化后原始误差](../minilm-mac-m4max-layer-errors-optimized/raw-errors.json)、[逐层汇总](../minilm-mac-m4max-layer-errors-optimized/summary.json) 和 [插桩漂移检查](../minilm-mac-m4max-layer-errors-optimized/probe-batches.json)。调试图从不用于性能计时。

同输入局部 NMSE 最高的五个节点：

| 节点（层编号从 0 开始） | 局部 NMSE |
|---|---:|
| `/encoder/layer.4/intermediate/dense/MatMul` | 0.00037537509 |
| `/encoder/layer.5/intermediate/dense/MatMul` | 0.000355831703 |
| `/encoder/layer.5/attention/self/value/MatMul` | 0.000334807108 |
| `/encoder/layer.2/attention/self/value/MatMul` | 0.000294169846 |
| `/encoder/layer.4/attention/self/value/MatMul` | 0.000270826933 |

这些是数值敏感候选，不是已经证明的任务因果敏感性。最后一个 block 的累计 NMSE：按张量 0.0563553、按通道 0.00360385；它与局部单 MatMul NMSE 的对象和含义不同。

## 5. 一个明确假设的消融

在观察误差之前保存的选择规则：按通道局部 NMSE 最大的一个 MatMul 保持 FP32，其余配置不变。假设：这会降低完整模型句向量相对 FP32 的 MSE；Spearman 和延迟可能改善也可能恶化。

最初诊断选择 `/encoder/layer.4/intermediate/dense/MatMul`，即第 5 个 block 的 FFN 扩展层；优化路径复验后最高误差节点仍是它。消融使用最初 exclusion 文件运行，两份 selection 记录都保留；没有隐瞒这一顺序，也未声称事先锁定了整个后续分析。该节点恢复 FP32 后，完整图为 35 个整数 MatMul、13 个浮点 MatMul，profiler 也记录 35 个整数相关 MatMul 事件。

| 配置 | Spearman | 句向量 MSE vs FP32 | ONNX MiB | 本次消融 3 轮 ms |
|---|---:|---:|---:|---:|
| FP32 | 0.867097776 | 0 | 86.22 | 1.987000 |
| 动态 INT8 按通道 | 0.865713899 | 1.05404488e-05 | 56.04 | 2.276541 |
| 按通道 + 单 MatMul 保留 FP32 | 0.865746332 | 9.82537929e-06 | 57.72 | 2.278312 |

句向量 MSE 降低 6.78%，支持这一限定的数值假设；Spearman 仅增加 0.000032433。按配对 validation 行 bootstrap 1,000 次，探索性 95% percentile 区间为 [-0.000301134, 0.000365020]，跨过零，不能宣称显著质量改善。该区间未重做选层，也未处理重复句子依赖，不能当作无偏研究结论。

[完整消融原始结果](../minilm-mac-m4max-ablation/)、[bootstrap 原始差值](../minilm-mac-m4max-quality-audit/bootstrap-deltas.json)、[漂移最大的 20 对样本分数](../minilm-mac-m4max-quality-audit/largest-output-drift.json) 均保留。探针 64 对与其余 1,436 对的探索性分项指标见 [quality-audit](../minilm-mac-m4max-quality-audit/summary.json)；其余样本已出现在原始完整基线评测中，不能重新包装成未见过的测试集。

## 6. 内存、数据和贡献边界

- RSS 是质量评测与计时后进程驻留内存。macOS `ru_maxrss` 是整个 worker 生命周期的峰值，采集位置已经过图检查和 profiler，包含 Python/数据/模型等；不是推理专属峰值。没有依据宣称模型内存、显存或统一内存降低。
- 未审计模型预训练数据重叠；公开 validation 被用于探针选择及结果观察，不能声称新建盲测集或无偏泛化改善。
- 模型、原始句子、数据文件不提交；只保存 revision/哈希、行号、公开标签、预测分数、误差统计和执行/计时记录。权重与中间张量留在忽略目录。
- Codex 在用户授权下修改实验程序并运行本轮实验；量化算法核心来自 ORT。不是用户独立提出量化算法、独立实现整数内核，也不能反向证明历史 YOLOv8/TensorRT/UR5e 贡献。
- 硬件端到端应用、Jetson 真机、TensorRT/NPU、大语言模型生成吞吐与 KV cache 量化均未完成。

## 7. SmoothQuant 决策与复跑

本轮不推进 SmoothQuant。当前验证的是动态 U8S8 的常量权重 MatMul；没有建立独立校准方案、等价缩放图改写或对应静态 W8A8 的完整后端验证。SmoothQuant 的核心是离线将激活量化难度迁移到权重，不能把局部浮点回退或按通道量化改名为 SmoothQuant。后续可在校准数据、图改写等价性和真实整数路径明确后再评估，并非断言 MiniLM/ORT 无法使用该方法。[原论文](https://proceedings.mlr.press/v202/xiao23c.html)

复跑命令、输入输出形状、工程检查清单、贡献表述和 10 道理解自测见 [交接说明](../../docs/mac-reproduction.md)。本轮保留历史字节、源文件哈希、Git 提交、命令、实际环境、原始计时、预测和 profiler 节点，便于独立审查。

## 8. 本轮验证

项目 Python 3.12 环境 15 项测试通过；不安装实验依赖的标准库环境通过 13 项、跳过 2 项数值依赖测试。所有 7 份 source 记录的源文件哈希与各自 Git HEAD 对象匹配。命令行尝试写入历史 Windows 目录返回 FileExistsError，未创建工作目录；历史 14 个文件的字节与起始提交完全一致。测试日志见 [validation.log](validation.log)、[标准库日志](stdlib-validation.log) 和 [禁止覆盖验证](no-clobber-check.log)。测试通过不等于部署验收或模型泛化保证。
