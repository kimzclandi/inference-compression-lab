# Inference Compression Lab

面向量化诊断与推理性能分析的个人研究实验。保留原始记录、负结果和明确的硬件/质量边界；模型权重不上传。

## 当前状态：QA 质量整改与明确停止条件

新增原文片段合同、证据偏移、拒答校准、模型/数据身份锁定和失败后关闭的本地入口。历史 block-10 回退确认仍失败，已退出部署候选。新增有限的模型/提示对照后，固定 1.5B FP16/Q8 在新校准集上各运行 128 题；原始 EM 为 74/128 与 76/128，但**均无满足预设精度、覆盖率、错误作答率和格式门槛的阈值**。按运行前提交的规则停止，不消费预留的 256 题确认集。默认入口返回 `unavailable_quality`，不能把这一保护措施写成 QA 质量已达标。

[整改报告与命令](docs/qa-remediation-study.md) · [冻结协议](configs/qa-remediation/study.json) · [校准原始记录](results/qa-remediation-v1) · [独立重算](results/qa-remediation-review-v1/independent) · [数据归属](configs/qa-remediation/ATTRIBUTION.md)

The QA remediation adds exact-span output validation, explicit abstention, fixed calibration gates, identity locking, and a fail-closed local entry point. Both 1.5B FP16 and Q8 failed the preregistered calibration requirements, so the reserved confirmation set was not evaluated. The entry point reports `unavailable_quality`; this is a reproducible negative research result and a release guard, **not deployment-quality QA**. The earlier block-restoration failure is unchanged.

## 历史主线：Qwen 低比特诊断与受控回退

从真实 Q4 退化样本出发，完成首 token logits 分析、24 个 decoder block 干预、等成本 FP16 回退对照和五轮性能复验。随后固定模块，在预先选定的 128 个新文章问题上完成一次确认：回退 EM 29/128，Q4 26/128，**未通过预设确认门槛**。保留负结果，不宣称质量已达标或原创量化算法。

[报告与复跑](docs/qwen-quantization-study.md) · [冻结协议](configs/qwen-quantization/study.json) · [原始结果](results/qwen-quantization-v1) · [离线重算代码](experiments/verify_qwen_quantization.py) · [学习单元](docs/qwen-quantization-learning.md) · [JD/简历证据表](docs/qwen-quantization-resume.md)

[新样本确认报告](docs/qwen-confirmation-study.md) · [发布验收与独立复跑](docs/release-reproduction.md) · [原始确认记录](results/qwen-confirmation-v1) · [数据许可](DATA_LICENSE.md) · [第三方归属](THIRD_PARTY.md)

## 独立审查与英文介绍

[独立技术审查与招聘判断](docs/independent-release-review.md)：从原始预测独立重算、修复完整性漏检，并以另一份推理循环复现 40 条已公布样本。原始 FP16 来源和两个回退模型的全部张量已独立核对。研究材料可复现不等于优化已确认、QA 可部署或作者已独立掌握。

A personal research artifact for low-bit inference diagnostics and controlled performance experiments. A fixed 128-question confirmation did **not** pass its quality gates: block-10 restoration scored 29/128 EM versus 26/128 for Q4, with a zero lower confidence bound and an excessive F1 drop from FP16. The review independently recomputed saved outputs and reproduced 40 published generations on the same M4 Max. MLX/ONNX Runtime provide the kernels and quantizers; this project contributes controlled experiments, cache lifecycle handling, diagnostic code, and verifiable evidence. No deployment-quality QA, novel quantization algorithm, or cross-device speedup is claimed.

## 已有实验入口

|实验|代码与证据入口|结论边界|
|---|---|---|
|MiniLM / ORT CPU 动态 INT8、逐层误差与回退|[Mac](docs/mac-reproduction.md) / [Windows](results/minilm-cpu-dynamic-int8/REPORT.md)|INT8 并非总比 FP32 快；文件更小不等于内存/功耗同幅下降|
|线程调优与长度分桶|[运行配置](docs/runtime-study-reproduction.md) / [分桶](docs/length-bucketing-study.md)|固定编码器负载与离线吞吐，不代替生成式或线上排队结论|
|Qwen / MLX Q8 前缀 KV 缓存|[初始实验](docs/qwen-prefix-study.md) / [公平基线](docs/qwen-fair-baseline-study.md)|主要减少重复 prefill；不能把 TTFT 收益写成完整生成同比加速|
|缓存生命周期与容量失效|[报告](docs/qwen-cache-lifecycle-study.md) / [学习](docs/qwen-cache-lifecycle-learning.md)|单调用者、受控故障注入；逻辑 KV 字节不是进程内存上限|
|Jetson Orin Nano 原项目|[只读盘点脚本](experiments/jetson_inventory.py)|待原代码和硬件接入；无本仓库 TensorRT/手机/昇腾实测|

本分支包含 PR #1–#6 的叠加成果、PR #7 独立审查修复及后续 QA 整改。审查时必须查看分支/PR，不能假定默认 main 已包含全部成果。不自动合并，不更改仓库可见性。

## 最短离线检查

无需 MLX、模型或网络：

```bash
python3 -m experiments.verify_qwen_quantization results/qwen-quantization-v1
python3 -m experiments.verify_release
python3 -m experiments.verify_qa_remediation --root results/qa-remediation-v1
python3 -m experiments.qwen_quantization_demo
python3 -m unittest discover -s tests -v
```

真实实验的固定环境、模型身份、数据许可和全量复跑命令见各报告；所有新输出目录拒绝覆盖。`runs/` 保存本地权重及中间产物，不上传。

发布候选支持从 pinned upstream 本地重建五个模型；已在全新 Python 环境和无 Git 源码归档完成真实推理。当前仓库保持私有、PR 保持草稿；根代码复用许可证等待维护者选择。

框架提供量化算子和推理后端；本项目实现对照、诊断、运行时改进和证据验证，不声称原创量化算法。学校/公司经历与个人实验分开归属；本人理解和复跑需另行验证。
