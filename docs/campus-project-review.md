# MiniLM CPU runtime evidence review / CPU 运行时证据审查

本页保留历史报告使用的路径，记录 MiniLM 运行时实验的技术验收范围。代码与实验由 Codex 辅助实现和执行，底层量化与整数内核由 ONNX Runtime 提供。

| 检查项 | 实验设计与证据 |
|---|---|
| 模型与数据身份 | 固定 revision、文件摘要与数据角色；保留失败记录 |
| 公平对照 | 同一 INT8 模型比较线程配置，FP32 独立调优 |
| 配置选择与验证 | 开发输入选择配置，冻结后用后续请求复测；多形状交叉检查 |
| 质量 | 冻结配置后使用 STS-B test split；预训练重叠未知，不能称隐藏盲测 |
| 运行合同 | 可加载配置、文件哈希核验、句向量 CLI 和错误输入拒绝 |
| 结论边界 | INT8 文件更小但相对最佳 FP32 仍慢；线程收益不是量化算法收益 |

结果与命令见 [完整复跑说明](runtime-study-reproduction.md) 和 [原始报告](../results/minilm-runtime-study-v1/REPORT.md)。

单台 M4 Max 的算子级 profiler 不能完整解释所有硬件开销，也不能外推跨设备、长期服务、功耗或并发结果。MiniLM 编码器实验不验证生成式模型、Jetson/TensorRT 或自研 kernel；仓库中的后续 Qwen 实验另有独立协议和报告。
