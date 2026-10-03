# Inference Compression Lab

面向量化、推理性能分析和端侧部署的可复现实验。保留原始记录、负结果和明确的硬件/质量边界；模型权重不上传。

## 当前主线：Qwen 低比特诊断与受控回退

从真实 Q4 退化样本出发，完成首 token logits 分析、24 个 decoder block 干预、等成本 FP16 回退对照、完整 dev 评分和五轮性能复验。结果仅为反复使用的 dev 上的探索，尚无独立确认或部署质量结论。

[报告与复跑](docs/qwen-quantization-study.md) · [冻结协议](configs/qwen-quantization/study.json) · [原始结果](results/qwen-quantization-v1) · [离线重算代码](experiments/verify_qwen_quantization.py) · [学习单元](docs/qwen-quantization-learning.md) · [JD/简历证据表](docs/qwen-quantization-resume.md)

## 已有实验入口

|实验|代码与证据入口|结论边界|
|---|---|---|
|MiniLM / ORT CPU 动态 INT8、逐层误差与回退|[Mac](docs/mac-reproduction.md) / [Windows](results/minilm-cpu-dynamic-int8/REPORT.md)|INT8 并非总比 FP32 快；文件更小不等于内存/功耗同幅下降|
|线程调优与长度分桶|[运行配置](docs/runtime-study-reproduction.md) / [分桶](docs/length-bucketing-study.md)|固定编码器负载与离线吞吐，不代替生成式或线上排队结论|
|Qwen / MLX Q8 前缀 KV 缓存|[初始实验](docs/qwen-prefix-study.md) / [公平基线](docs/qwen-fair-baseline-study.md)|主要减少重复 prefill；不能把 TTFT 收益写成完整生成同比加速|
|缓存生命周期与容量失效|[报告](docs/qwen-cache-lifecycle-study.md) / [学习](docs/qwen-cache-lifecycle-learning.md)|单调用者、受控故障注入；逻辑 KV 字节不是进程内存上限|
|Jetson Orin Nano 原项目|[只读盘点脚本](experiments/jetson_inventory.py)|待原代码和硬件接入；无本仓库 TensorRT/手机/昇腾实测|

本分支由 PR #1–#5 逐层叠加而来。审查时必须查看分支/PR，不能假定默认 main 已包含全部成果。不自动合并，不更改仓库可见性。

## 最短离线检查

无需 MLX、模型或网络：

```bash
python3 -m experiments.verify_qwen_quantization results/qwen-quantization-v1
python3 -m experiments.qwen_quantization_demo
python3 -m unittest discover -s tests -v
```

真实实验的固定环境、模型身份、数据许可和全量复跑命令见各报告；所有新输出目录拒绝覆盖。`runs/` 保存本地权重及中间产物，不上传。

框架提供量化算子和推理后端；本项目实现对照、诊断、运行时改进和证据验证，不声称原创量化算法。学校/公司经历与个人实验分开归属；本人理解和复跑需另行验证。
