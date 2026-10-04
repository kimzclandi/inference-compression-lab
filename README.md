# Inference Compression Lab

面向量化诊断与推理性能分析的个人研究实验。保留原始记录、负结果和明确的硬件/质量边界；模型权重不上传。

## 从这里开始 / Start here

本项目研究：在固定本地 CPU 成本下，量化、抽取解码和拒答策略如何分别影响延迟、答案正确性与覆盖率。
This project studies the separate effects of quantization, span decoding and abstention on local CPU cost, answer correctness and coverage.

- [系统结构、实现取舍与证据导航](docs/qa-system-overview.md)：按输入→推理→解码→排序→门控定位代码。
- 无需模型的完整核验与精简双语报告（需 `numpy==2.2.6`，输出目录必须全新）：

```bash
python -m experiments.review_qa_evidence --output-dir runs/my-evidence-review
```

命令从原始记录完整核验后输出 `review.md`、`review.json`、`acceptance.json` 与 `run.json`。失败会保留记录、返回非零状态，不生成通过报告；不会运行新评估或检查远端发布状态。实际模型演示另见[无 Git 复现指南](docs/release-reproduction.md)。

[新增第三特征固定消融](docs/qa-span-gap-ablation.md)：仅使用已公布训练/校准材料。两窗反例可区分机制，但真实数据0行特征改变，未产生质量提升；默认策略保持不变。

## 当前状态：历史样本通过；扩大训练后的新校准失败

[后续非线性与不确定性特征实验](docs/qa-nonlinear-study.md)：两项固定候选也未通过完整开发门槛，预留评估仍未运行。Two subsequent fixed ranking challengers also failed development gates; the reserved evaluation remains unused.

最新一次固定实验把 INT8 五特征正确性排序头的训练集从 256 题扩大到 2,304 题，保持基础模型、特征、优化器和阈值网格不变。训练在 8 次迭代后收敛，但原头与扩大训练后的头均未通过新的 192 题校准门槛，因此没有运行预留的 192 题评估，也没有提高已验证的可用覆盖率。[完整协议、曲线与复跑](docs/qa-expanded-ranking-study.md)。

The latest fixed experiment expanded the five-feature INT8 correctness-head training set from 256 to 2,304 rows. Fitting converged in eight iterations, but both the original and expanded heads failed the new 192-question calibration gates. The reserved 192-question evaluation was not run. This exposes limited transfer of the earlier result; it does not establish improved usable coverage or deployment quality.

### 历史固定样本结果 / Earlier fixed-cohort result

新增专用抽取模型与固定五特征正确性排序头。INT8 在校准集固定阈值后，对四篇新文章的 128 题只评估一次：接受 27 题，27 题全部 EM 正确；可回答覆盖率 27/64，64 个不可回答问题全部拒答，五项预设点门槛通过。支持有输入范围和严格身份验证的本地原型，**不证明通用部署质量**。精度区间下界约 87.54%，仍有 37 个可回答问题被拒答。

[RC4 无 Git 使用、依赖与发布访问](docs/release-reproduction.md) · [固定排序实验、命令与限制](docs/qa-risk-study.md) · [专用模型 v1 与 CPU 性能](docs/qa-specialist-study.md) · [128 题原始 logits](results/qa-risk-v2/evaluation-int8) · [独立验收](results/qa-risk-review-v1/verification.json)

The frozen INT8 correctness-ranking pipeline passed five empirical point gates on 128 locally reserved public-benchmark questions: 27/27 accepted answers correct, 27/64 answerable coverage, and 0/64 false acceptance on unanswerable questions. This supports a bounded local supplied-passage QA prototype, not a general deployment claim or population-risk guarantee. No model weights or learned ranking parameters are distributed. Earlier failures remain unchanged.

## 先前 QA 整改：保留的失败与停止条件

新增原文片段合同、证据偏移、拒答校准、模型/数据身份锁定和失败后关闭的本地入口。历史 block-10 回退确认仍失败，已退出部署候选。新增有限的模型/提示对照后，固定 1.5B FP16/Q8 在新校准集上各运行 128 题；原始 EM 为 74/128 与 76/128，但**均无满足预设精度、覆盖率、错误作答率和格式门槛的阈值**。按运行前提交的规则停止，不消费预留的 256 题确认集。默认入口返回 `unavailable_quality`，不能把这一保护措施写成 QA 质量已达标。

[整改报告与命令](docs/qa-remediation-study.md) · [冻结协议](configs/qa-remediation/study.json) · [校准原始记录](results/qa-remediation-v1) · [独立重算](results/qa-remediation-review-v1/independent) · [数据归属](configs/qa-remediation/ATTRIBUTION.md)

The QA remediation adds exact-span output validation, explicit abstention, fixed calibration gates, identity locking, and a fail-closed local entry point. Both 1.5B FP16 and Q8 failed the preregistered calibration requirements, so the reserved confirmation set was not evaluated. The entry point reports `unavailable_quality`; this is a reproducible negative research result and a release guard, **not deployment-quality QA**. The earlier block-restoration failure is unchanged.

## 历史主线：Qwen 低比特诊断与受控回退

从真实 Q4 退化样本出发，完成首 token logits 分析、24 个 decoder block 干预、等成本 FP16 回退对照和五轮性能复验。随后固定模块，在预先选定的 128 个新文章问题上完成一次确认：回退 EM 29/128，Q4 26/128，**未通过预设确认门槛**。保留负结果，不宣称质量已达标或原创量化算法。

[报告与复跑](docs/qwen-quantization-study.md) · [冻结协议](configs/qwen-quantization/study.json) · [原始结果](results/qwen-quantization-v1) · [离线重算代码](experiments/verify_qwen_quantization.py)

[新样本确认报告](docs/qwen-confirmation-study.md) · [发布验收与独立复跑](docs/release-reproduction.md) · [原始确认记录](results/qwen-confirmation-v1) · [数据许可](DATA_LICENSE.md) · [第三方归属](THIRD_PARTY.md)

## 独立审查与英文介绍

[独立技术审查](docs/independent-release-review.md)：从原始预测独立重算、修复完整性漏检，并以另一份推理循环复现 40 条已公布样本。原始 FP16 来源和两个回退模型的全部张量已独立核对。研究材料可复现不等于优化已确认、QA 可部署。

A personal research artifact for low-bit inference diagnostics and controlled performance experiments. A fixed 128-question confirmation did **not** pass its quality gates: block-10 restoration scored 29/128 EM versus 26/128 for Q4, with a zero lower confidence bound and an excessive F1 drop from FP16. The review independently recomputed saved outputs and reproduced 40 published generations on the same M4 Max. MLX/ONNX Runtime provide the kernels and quantizers; this project contributes controlled experiments, cache lifecycle handling, diagnostic code, and verifiable evidence. No deployment-quality QA, novel quantization algorithm, or cross-device speedup is claimed.

## 已有实验入口

|实验|代码与证据入口|结论边界|
|---|---|---|
|MiniLM / ORT CPU 动态 INT8、逐层误差与回退|[Mac](docs/mac-reproduction.md) / [Windows](results/minilm-cpu-dynamic-int8/REPORT.md)|INT8 并非总比 FP32 快；文件更小不等于内存/功耗同幅下降|
|线程调优与长度分桶|[运行配置](docs/runtime-study-reproduction.md) / [分桶](docs/length-bucketing-study.md)|固定编码器负载与离线吞吐，不代替生成式或线上排队结论|
|Qwen / MLX Q8 前缀 KV 缓存|[初始实验](docs/qwen-prefix-study.md) / [公平基线](docs/qwen-fair-baseline-study.md)|主要减少重复 prefill；不能把 TTFT 收益写成完整生成同比加速|
|缓存生命周期与容量失效|[报告](docs/qwen-cache-lifecycle-study.md)|单调用者、受控故障注入；逻辑 KV 字节不是进程内存上限|
|Jetson Orin Nano 原项目|[只读盘点脚本](experiments/jetson_inventory.py)|待原代码和硬件接入；无本仓库 TensorRT/手机/昇腾实测|

本分支包含 PR #1–#6 的叠加成果、PR #7 独立审查修复及后续 QA 整改。审查时必须查看分支/PR，不能假定默认 main 已包含全部成果。不自动合并，不更改仓库可见性。

## 最短离线检查

历史材料重算无需 MLX、模型或网络：

```bash
python3 -m experiments.verify_qwen_quantization results/qwen-quantization-v1
python3 -m experiments.verify_release
python3 -m experiments.verify_qa_remediation --root results/qa-remediation-v1
python3 -m experiments.qwen_quantization_demo
python3 -m unittest discover -s tests -v
```

RC4 完整验收新增小排序头重建，需要 `numpy==2.2.6`，运行 `python -m experiments.verify_release_rc4`。这与上面历史标准库验收的范围不同。

真实实验的固定环境、模型身份、数据许可和全量复跑命令见各报告；所有新输出目录拒绝覆盖。`runs/` 保存本地权重及中间产物，不上传。

发布候选支持从 pinned upstream 本地重建五个模型；已在全新 Python 环境和无 Git 源码归档完成真实推理。当前仓库保持私有、PR 保持草稿；本分支尚未包含根代码 LICENSE，尚未创建 tag 或 Release。

框架提供量化算子和推理后端；本项目实现对照、诊断、运行时改进和证据验证，不声称原创量化算法。代码与实验由 AI 辅助实现和执行。

专用模型基础流水线在 M4 Max CPU 的固定 8 输入、6 进程轮换中为 72.052 ms FP32 / 54.188 ms INT8（1.330×）；带正确性排序器和拒答门控的 INT8 流水线另测 69.600 ms。两者计时范围不同，不把前者加速倍数套到后者。新 QA 原型也没有新的 FP32 配对质量非劣证据。

按文章检查暴露了明显分布差异：Civil_disobedience 的 32 题全部拒答（可回答覆盖率 0/16），Ctenophora 为 6/16，Harvard_University 为 11/16，Yuan_dynasty 为 10/16。被拒的 37 个可回答问题中，有 19 个 raw span 本来 EM 正确。整体点门槛通过不能外推到每一篇文章或业务领域；排序器换来了精度，也丢弃了有效答案。

[系统结构与验证入口](docs/qa-system-overview.md) · [第三特征消融](docs/qa-span-gap-ablation.md)。

[不重叠竞争片段覆盖率实验](docs/qa-coverage-gap-study.md)：实际改变多数样本特征，但候选校准失败，保留原策略。 / Disjoint-competitor features changed most development rows but failed calibration; the default policy is retained.

[扩大排序头训练实验](docs/qa-expanded-ranking-study.md)：2,304 题训练收敛，新校准失败；包含全部 raw→feature→label 审计、存档矩阵重建和禁止失败后评估的回归检查。
