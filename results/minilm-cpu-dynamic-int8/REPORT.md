# MiniLM 动态 INT8 实测报告

日期：2026-09-30。性质：真实预训练模型、真实 CPU 整数算子推理；不是 fake quant，不是 Jetson 测试。

## 方法与测量

- 模型：sentence-transformers/all-MiniLM-L6-v2，使用作者仓库现有 ONNX 文件；版本和 SHA256 见 provenance.json。没有自行从 PyTorch 导出。
- 数据：完整 STS-B validation，1,500 对句子。最大长度 256，batch=16，attention-mask mean pooling 后 L2 归一化，余弦相似度与标签计算 Spearman。
- 不训练、不选参；预先定义 FP32、按张量和按通道两种动态量化配置。动态激活范围在推理时计算，因此没有静态校准集。
- 后端：ONNX Runtime 1.30.0 CPUExecutionProvider；Windows 11，AMD Ryzen 7 5800H，intra-op=4，inter-op=1。
- 量化：调用官方 quantize_dynamic，signed INT8 权重、动态 U8 激活，reduce_range=False，仅量化常量权重 MatMul。两种粒度不是两种独立研究算法。
- 时延：batch=1、固定 padding 长度 64，使用验证集首句；仅 session.run，不含分词、池化和模型加载。每轮预热 50 次、测量 200 次，共 3 轮。
- 下表时延为三轮中位数的中位数；逐次耗时、每轮 P95 保存在各配置 JSON。

## 实测结果

| 配置 | Spearman | ONNX 文件 MiB | 推理时延 ms | 相对 FP32 倍速 |
|---|---:|---:|---:|---:|
| FP32 | 0.867098 | 86.22 | 9.17285 | 1.000 |
| 动态 INT8，按张量权重 | 0.864163 | 55.94 | 5.84955 | 1.568 |
| 动态 INT8，按通道权重 | 0.865655 | 56.04 | 6.14350 | 1.493 |

按通道配置相关系数绝对下降约 0.001442，文件缩小约 35.0%。该百分比是模型文件大小变化，不是内存下降。Spearman 是相关系数，不是准确率。

按通道配置的平均句向量余弦一致性为 0.992318，按张量为 0.964029。按通道在本次质量指标上更接近浮点，但本次计时略慢于按张量，不能简单称为全面更优。未做统计显著性分析。

## 整数执行证据

量化图各有 36 个 MatMulInteger，仍有 12 个浮点 MatMul。独立 profiler 运行记录到 CPUExecutionProvider 上 18 个 DynamicQuantizeMatMul 和 18 个 MatMulIntegerToFloat 事件。归一化、Softmax、Embedding 等未宣称量化；没有 KV cache 量化。

execution.json 保存原始图节点清单、执行算子汇总和 profiler 节点记录。计时运行关闭 profiler，另开 session 采集执行证据。

## 内存口径

分别在独立子进程运行，记录 Windows 进程生命周期峰值工作集及评测后 RSS。峰值包括 Python、数据、分词器、模型加载与推理，并不是模型张量占用或 GPU 显存，不用于声称显存压缩。原始字节数见各配置 JSON。

## 限制及未完成项

- 仅一台日常使用的笔记本 CPU；后台负载、频率和温度没有严格控制，配置按固定顺序测量，不是交错随机化的系统性能研究。
- 时延仅覆盖一个形状、一个输入；质量评测与性能评测负载不同。不代表端到端服务吞吐、GPU 或 Jetson 性能。
- 使用公开 validation 而非新建盲测集；没有审计模型预训练数据与评测集的重叠，不能声称新的泛化能力结论。
- 没有 SmoothQuant、蒸馏、剪枝、逐层激活误差或硬件指令级验证。尚未完全满足原计划中的全部研究验收项。
- U8S8 在不同 CPU 指令路径可能有数值差异；本次未通过 reduce_range 或 U8U8 对照定位误差原因。

## 贡献和简历边界

代码与实验由 Codex 在用户授权下实现并运行，量化核心使用 ORT 官方实现。这些新实验不能反向证明用户历史项目，也不自动代表用户已经独立掌握实现。

用户阅读、复跑并能解释后，可按实际参与程度表述：

> 基于 ONNX Runtime 开展 MiniLM 动态 INT8 量化对照，在 STS-B 验证集 1,500 对样本上比较按张量与按通道量化；通过执行图和 profiler 核验整数算子，在固定 CPU 负载下评估语义相似度、模型文件大小及推理时延。

不可改写为：独立提出量化算法、完成大语言模型 W8A8 全量化、Jetson INT8 加速或生产部署。

## 来源

- [模型及模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [STS-B 数据仓库](https://huggingface.co/datasets/sentence-transformers/stsb)
- [ORT 量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

模型与数据保留各自上游许可；仓库仅保存获取脚本、哈希、预测分数和实验记录，不重新发布权重或原始句子。
