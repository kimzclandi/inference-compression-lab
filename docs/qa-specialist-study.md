# Task-trained extractive QA study / 专用抽取式问答研究

**Specialist v1 的固定评估失败，回答入口保持 `unavailable_quality`。** 上游专用模型和抽取式输出合同解决了任意生成文本的问题，但固定阈值在评估数据上的答案精度及覆盖率均未达标。动态 INT8 在本机 CPU 的固定工作负载上测得约 1.33× 请求加速，文件也更小；这些结果不等于问答系统可部署。下文保留 v1 的完整负结果；一次新的监督风险评分头实验仍为 **pending**，不得用它覆盖或提前改写 v1 结论。

**The frozen specialist-v1 evaluation failed; the answer entry remains `unavailable_quality`.** A task-trained model and an extractive output contract constrain answer format, but the selected policies missed both precision and coverage requirements on evaluation. Dynamic INT8 achieved about 1.33× request speedup on the measured local CPU workload and reduced ONNX file bytes. This does not establish deployable QA quality. The one supervised risk-head follow-up is **pending**, and does not change the v1 result.

## Model, quantization and provenance / 模型、量化与来源

本实验使用已有 SQuAD2 监督训练的 `deepset/roberta-base-squad2`，固定 revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`。项目没有训练该基础 QA 模型。原始 safetensors、配置、tokenizer 文件及上游模型卡均有逐文件 SHA256；ONNX FP32 从该原始模型导出，再由 ONNX Runtime 动态量化生成 INT8 图。[固定上游身份](../configs/qa-specialist/upstream.json)、[资产与导出证据](../results/qa-specialist-assets-v1/assets.json)、[固定版本模型卡](https://huggingface.co/deepset/roberta-base-squad2/blob/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4/README.md)。

The base QA model is upstream work, already supervised on SQuAD2. The project exports a pinned checkpoint to FP32 ONNX and applies ORT dynamic quantization; it does not claim to have trained the task model or invented the quantizer. The pinned model card declares CC BY 4.0. Dataset excerpts retain CC BY-SA 4.0 with [separate attribution](../configs/qa-specialist/ATTRIBUTION.md). Neither notice selects the repository's root code license; model weights remain local and are excluded from release artifacts.

量化设置为动态 activation UINT8 / per-channel weight INT8，`reduce_range=False`、`MatMulConstBOnly=True`。固定 QA projection head 保持 FP32，此规则在质量评估前确定，没有根据评估样本搜索回退层。图中有 72 个 `MatMulInteger`、48 个 `DynamicQuantizeLinear`，仍有 25 个浮点 `MatMul`；不能称整个模型都以 INT8 执行。它不是 MLX Q8，也不是 FP8。[构建实现](../experiments/prepare_qa_specialist_assets.py)。

Quantization uses dynamic UINT8 activations and per-channel INT8 weights for eligible constant-weight operations, with a predeclared FP32 QA head. Integer operators demonstrate a real quantized graph, but do not themselves prove acceleration. Three synthetic export checks compared original PyTorch and FP32 ORT start/end logits at sequence lengths 21, 24 and 315; maximum absolute error was about `2.60e-5`, below the fixed `1e-3` tolerance. These are bounded export-parity checks, not quality evaluation or proof for every shape.

运行环境记录为 Python 3.12.11、PyTorch 2.8.0、Transformers 4.56.2、ONNX 1.19.0、ONNX Runtime 1.22.1、NumPy 2.2.6、tokenizers 0.22.0。推理严格校验资产清单、tokenizer/模型哈希和运行依赖版本，采用 `CPUExecutionProvider`、`ORT_ENABLE_ALL`、intra-op 4 线程、inter-op 1 线程、sequential execution。[依赖锁定](../requirements-qa-specialist.lock.txt)、[运行时](../lab/qa_specialist_runtime.py)。

## Fixed decoding and acceptance policy / 固定解码与接受策略

输入是英文问题和调用方提供的 passage；输出候选必须是该 passage 的精确子串，并携带 Python Unicode 字符偏移 `[start, end)` 与 context SHA256。问题最多 64 tokens，context 最多 12,000 字符，最多 8 个 384-token 窗口，重叠 stride 为 128。超限直接失败，不静默丢弃窗口或截断问题。question、special、padding token 全部不能进入答案。

The decoder exhaustively checks legal context spans of at most 30 tokens. Zero-character tokenizer offsets cannot be answer endpoints, but may occur inside a span and count toward its token budget. Boundary whitespace is trimmed while adjusting character offsets. For window `w`, let `S_w` be its largest legal start-plus-end logit sum and `N_w` its CLS start-plus-end sum. The selected window maximizes `S_w − N_w`; ties prefer lower window, start and end indices. `sigmoid(S_w − N_w)` is an uncalibrated ranking score, not a correctness probability. Gold answers and answerability labels do not enter decoding.[实现](../lab/extractive_qa.py) / [Implementation](../lab/extractive_qa.py).

这套窗口聚合规则与 Transformers 官方 QA example postprocessor 的“全局最高 span 与跨窗最低 null 比较”不同，不是官方 `pipeline` 数值行为的复刻，也没有证明本策略更准确。多个窗口取最大值仍可能放大误接受风险。[独立机制审查与反例](../results/qa-specialist-review-v1/independent/decoder-review.md)。

候选解码始终返回 span；后续策略低于固定阈值时才输出 `NO_ANSWER`。因此表中 raw EM 不含拒答策略，不能把 raw 的不可回答得分为零误读为服务实际对全部问题作答。无论格式多正确，精确子串都可能回答错误问题，不能用 substring 检查替代语义正确性评估。

The raw decoder always proposes a span; only the separately frozen threshold policy turns low-score candidates into `NO_ANSWER`. Exact substring provenance constrains format, not truth. The same passage can contain a perfectly valid span that does not answer the question.

## Fixed calibration and evaluation / 固定校准与评估

[协议](../configs/qa-specialist/study.json)在推理前锁定模型身份、代码、数据、阈值网格和验收规则。calibration 为 4 篇文章、128 题、103 个 context 家族，evaluation 为另 8 篇文章、256 题、189 个家族；两组各有一半不可回答。标题、ID、规范化精确 context/question 与已盘点旧材料互斥。两个变体分别从预设网格选择满足全部条件的最小阈值；只有两者均通过校准才允许评估。实际选择和评估授权在评估推理前提交，未按 evaluation 结果更换阈值。[数据清单](../configs/qa-specialist/dataset/manifest.json)、[阈值与完整校准曲线](../results/qa-specialist-v1/selection.json)、[评估授权](../results/qa-specialist-v1/evaluation-protocol.json)。

This is a fixed local assessment on **public SQuAD2 development data**, not an official hidden test or model-unseen confirmation. The upstream specialist was trained on SQuAD2 and its model card already reports public-dev results. Project-level exact disjointness cannot remove upstream benchmark exposure, pretraining contamination, semantic near-duplicates or selection bias. Article quotas, unique-context constraints and 4,000-character sample limits also restrict generalization. Published evaluation samples become reproduction data on subsequent runs.

固定质量门槛 / Frozen empirical point gates:

| Metric / 指标 | Definition / 分母 | Requirement |
|---|---|---:|
| Accepted precision / 已回答精度 | exact-correct accepted / all accepted | ≥90% |
| Answerable answer coverage / 可回答覆盖率 | accepted answerable / all answerable | ≥40% |
| Correct answerable coverage / 正确可回答覆盖率 | exact-correct accepted / all answerable | ≥35% |
| Unanswerable false acceptance / 不可回答误接受率 | accepted unanswerable / all unanswerable | ≤10% |
| Invalid output rate / 非法输出率 | invalid candidates / all questions | ≤5% |

These are sample point-estimate gates. Reported Wilson intervals are descriptive only, do not adjust for article/context dependence, and are not a population-risk guarantee. Zero accepted answers gives undefined precision and cannot pass.

| Phase / 阶段 | Variant | Frozen margin threshold | Accepted precision | Answerable coverage | Correct answerable coverage | Unanswerable false acceptance | All five gates |
|---|---|---:|---:|---:|---:|---:|---|
| Calibration, 128 | FP32 | 10 | 27/27 = 100% | 27/64 = 42.19% | 27/64 = 42.19% | 0/64 = 0% | pass |
| Calibration, 128 | INT8 | 8 | 28/31 = 90.32% | 30/64 = 46.88% | 28/64 = 43.75% | 1/64 = 1.56% | pass |
| Evaluation, 256 | FP32 | 10 | 45/51 = 88.24% | 49/128 = 38.28% | 45/128 = 35.16% | 2/128 = 1.56% | **fail** |
| Evaluation, 256 | INT8 | 8 | 48/54 = 88.89% | 51/128 = 39.84% | 48/128 = 37.50% | 3/128 = 2.34% | **fail** |

所有四个运行的 invalid rate 都是 0；这没有挽救评估失败。FP32/INT8 在 evaluation 的 system EM 分别为 `171/256 = 66.80%`、`173/256 = 67.58%`，system F1 为 `0.680196`、`0.685379`。始终拒答基线为 50% EM。raw answerable EM 为 `107/128`、`108/128`，与上述包含拒答的 system EM 分母不同，不能混写。完整原始输出、逐 token logits、阈值、评分和失败门槛见[验收结果](../results/qa-specialist-v1/verification.json)及四个 run 目录。

INT8−FP32 的配对 system EM 差为 `+0.78125 pp`，95% 区间 `[-1.6000, +3.4884] pp`；F1 差为 `+0.51839 pp`，区间 `[-2.09191, +3.24474] pp`。统计采用文章分层、context-family 为采样单位、5,000 次 bootstrap，文章本身不重采样。EM 非劣下界满足预设 `−2 pp`，F1 下界略低于 `−2 pp`，而两者质量门槛也失败，故**完整压缩验收未通过**。不能因 INT8 的点估计略高就写“保持精度”或“优化已确认有效”。

The independent auditor imports neither the project decoder nor its scorer/gate modules. It re-enumerated **3,310,578 legal spans across 772 windows and 768 predictions**, reproduced chosen candidates, scores, calibration selection and fixed-policy evaluation. This verifies arithmetic and evidence consistency; it is not another model inference, a new evaluation or proof of production readiness.[Independent audit](../results/qa-specialist-review-v1/independent/evaluation-audit.json).

## CPU performance and resource boundaries / CPU 性能与资源边界

固定性能协议使用 8 个由 ID 哈希选出的 calibration 输入，与答案标签和模型结果无关；6 个独立进程按 FP32、INT8、INT8、FP32、FP32、INT8 顺序执行。每进程先 warm up 8 次，再对每个输入测量 5 次，共 40 次测量；每个精度各 3 个进程。运行记录为单台 macOS ARM64 主机，无并发。[性能协议与原始记录](../results/qa-specialist-performance-v1/protocol.json)、[汇总](../results/qa-specialist-performance-v1/summary.json)。

| Measured quantity / 实测对象 | FP32 | INT8 |
|---|---:|---:|
| Median of process median request times | 72.052 ms | 54.188 ms |
| ONNX file bytes / 文件字节数 | 496,445,889 | 242,243,831 |
| Process lifetime peak RSS range / 全进程峰值 RSS 范围 | 1,390,460,928–1,408,745,472 bytes | 935,559,168–976,928,768 bytes |
| Initialization range / 初始化时间 | 1.634–1.708 s | 1.554–1.630 s |

请求计时包含 tokenizer、所有窗口的同步 ORT CPU 推理、穷举解码；不含模型加载、JSON 证据写入和质量证据验证。`72.052 / 54.188 = 1.3297×` 是该固定单调用方工作负载的实测比值。全部 8 个性能样本实际均为单窗口、135–256 tokens，不能据此推断 8 窗口上限或更长 passage 的吞吐。三进程重复不构成线上延迟 SLO，也没有并发、能耗、设备间或生产规模证据。

The ONNX size ratio is `0.487956` (about 51.20% fewer file bytes). Peak RSS is a separate, process-lifetime high-water mark including initialization, tokenizer, ORT and model execution. Neither metric is logical KV storage, isolated weight allocation or the other's proxy. There is no autoregressive generation here, so these measurements must not be described as TTFT or decode-token throughput. The numbers also exclude the pending correctness-head feature extraction and scoring cost.

## Contribution and limits / 项目贡献与限制

| Existing framework/upstream capability | Project code and experimental contribution |
|---|---|
| SQuAD2-trained RoBERTa QA model and fast tokenizer | Pinned model/tokenizer identity, source/license records, bounded input contract and exact character-span provenance |
| PyTorch ONNX export and ORT dynamic INT8 kernels | Fixed head exclusion, export-parity checks, graph/file evidence and controlled FP32/INT8 comparison |
| Start/end QA logits and ordinary null-answer scoring | Explicit exhaustive decoding and window rule, mask/zero-offset handling, independently reimplemented arithmetic audit |
| Standard metrics, bootstrap and logistic regression methods | Frozen data roles, threshold-only calibration, precision/coverage gates, dependency-aware paired statistics, preserved failures and no-overwrite evidence |

本项目的价值是把模型能力、量化实现、置信排序、拒答策略、性能和发布证据放到同一套可审计协议中，区分“文件更小”“特定请求更快”“质量非劣”“系统能上线”。它没有提供原创量化器、CUDA kernel、训练加速、蒸馏、稀疏、token 压缩或分布式服务成果；文档/测试数量本身不是简历贡献。代码由 AI 辅助完成也不证明作者已能独立解释或复跑。

This study is separate from the historical Qwen block-10 fallback experiment. Different architecture, precision scheme, task head, datasets and acceptance policies prevent a causal before/after comparison. It neither repairs nor re-confirms that failed fallback hypothesis. The historical Qwen result and poor-generative-QA results remain intact; any CV statement must attribute this work to the personal project and retain its measured scope.

## One supervised risk-head follow-up — pending / 唯一监督风险评分头追加实验——待验收

**本节截至本草稿提交时仅记录冻结机制和数据角色，未宣称训练、校准或新评估通过。** v1 原 calibration 128 题及已经失败的 evaluation 256 题全部降为 development。按固定文章哈希分成 train 256 题和 calibration 128 题；另选剩余 4 篇文章共 128 题作为这次后续假设的固定 evaluation。这是观察到失败之后提出的新假设，适应性研究历史必须保留；不能把重新分配的旧 evaluation 再称为未见验证。[数据分配与限制](../configs/qa-risk/ATTRIBUTION.md)。

The single follow-up holds the base model, quantization and frozen span decoder unchanged. A five-feature linear logistic head will rank candidate correctness: selected span/null margin; selected joint start/end log score normalized over context plus CLS; gap to the best differently normalized candidate across windows; `log1p(answer token count)`; and `log1p(window count)`. Features receive no gold labels. Training labels are supplied separately; means/stds are fitted on training rows only. The fixed objective is summed binary logistic loss plus `0.5 × ||weights||²`, with an unregularized intercept, at most 100 Newton steps and fixed Armijo backtracking. Missing alternative candidates, malformed evidence, non-finite parameters or failed convergence are rejected.[Fixed head implementation](../lab/qa_risk_calibration.py).

风险分数是拟合得到的 logistic score，不保证为校准后的正确概率。剩余评估只有四篇文章，仍是已被上游使用的公开 benchmark；即使通过也不能证明业务部署质量。此草稿不填写尚未独立核验的风险 head 结果、资源开销、发布状态或简历性能数字；最终记录必须由实际运行与重算补齐，并保留这次实验与 v1 的区分。

首个训练进程因数值运行时问题失败，原目录保留。独立合成复现显示：记录环境 NumPy 2.2.6 使用 Accelerate BLAS，32/128 行矩阵可拟合，而有限的 256/512/1,024 行矩阵在零初始化 `matmul` 报 `divide by zero`。将固定六维乘积改为 `einsum(optimize=False)` 后，五种规模均收敛且重复结果逐值一致，独立标准库重算目标值误差小于 `1.2e-13`。数学目标、特征、训练标签、正则项和优化门槛均未改变；这支持实现层 workaround，未定位上游库的完整根因，也不是质量提升证据。[原始失败](../results/qa-risk-v1/training/run.json)、[独立合成复现](../results/qa-risk-review-v1/numerical-fault/recorded-run/report.json)。

The initial fit failure is preserved. A synthetic regression reproduced the matrix-size-dependent runtime exception and verified the explicit-contraction fix without inspecting new evaluation outcomes. Before/after source snapshots, NumPy build details and a repeatable script are retained with the numerical report. This implementation repair does not authorize another model, feature or threshold search.

## Reproduction / 复现入口

以下命令只重算公开的现有证据，不调用模型、不重新选阈值；独立 audit 的输出路径必须不存在。成功运行验证器表示证据可重算，报告中的 `overall_success: false` 仍代表实验未通过。

```bash
python3 -B -m experiments.verify_qa_specialist \
  --root results/qa-specialist-v1 \
  --study configs/qa-specialist/study.json
python3 -B results/qa-specialist-review-v1/independent/audit.py \
  --root results/qa-specialist-v1 \
  --output runs/specialist-independent-recheck.json
```

Actual model reproduction requires the pinned local source weights/assets and the locked environment. Use new output directories; never overwrite the archived runs. The selected policies and model identities are already public project evidence, so these commands are reproductions, not new confirmations.

```bash
python -B -m experiments.qa_specialist \
  --asset-root runs/qa-specialist-assets-v1 \
  --spec configs/qa-specialist/study.json \
  --data configs/qa-specialist/dataset/evaluation/data.jsonl \
  --variant int8 --selection results/qa-specialist-v1/selection.json \
  --output-dir runs/specialist-int8-reproduction
python -B -m experiments.benchmark_qa_specialist \
  --asset-root runs/qa-specialist-assets-v1 \
  --spec configs/qa-specialist/study.json \
  --output-dir runs/specialist-cpu-reproduction
```
