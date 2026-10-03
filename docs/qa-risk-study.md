# Fixed correctness ranking / 固定正确性排序实验

**INT8 + 固定排序头在新的 128 题评估上通过了五项预设样本门槛。它支持有边界的本地研究原型，不证明通用部署质量。** 这不是把旧失败改成成功：Qwen block-10 的确认失败、Qwen 自审校准失败、专用抽取 QA v1 的固定评估失败全部保留；新方法、新数据角色和新结论单列。

**The fixed INT8 correctness-ranking pipeline passed all five predeclared empirical point gates on 128 newly evaluated public-benchmark questions. This supports a bounded local research prototype, not general deployment readiness.** Every earlier failed study remains unchanged. The earlier FP32/INT8 quality noninferiority claim is still unsupported.

## Problem → hypothesis → fixed test / 问题与固定验收

问题是高 span/null margin 只能支持“相比拒答更倾向此片段”，不能确保片段精确回答了问题。固定假设：冻结已训练的 RoBERTa QA 模型，以旧样本监督训练一个五特征 logistic correctness ranker，可能更好地识别该模型给出的错误片段。标签是 normalized exact match (EM)，不是普遍的语义真伪标签。

五特征依次为：选中 span/null margin；选中窗口内 start/end 两个 context+CLS softmax 的 log probability 之和；选中答案与全部窗口中最佳不同 normalized answer 的 margin 差；`log(1+answer_token_count)`；`log(1+window_count)`。第三项不能只找每窗 winner；第二项不是对合法 span 集合归一化的联合概率。gold 只用于训练目标和离线评分，不进入特征或推理。无不同答案候选时失败关闭。

The five label-free features are fixed in [source](../lab/qa_risk_calibration.py) and [protocol](../configs/qa-risk/study.json). Each precision has its own train-only standardization and logistic head. Optimization minimizes the **sum** binary logistic loss plus `0.5 * ||w||²`, with no intercept regularization, fixed Newton/Armijo settings, and no hyperparameter search. These are project-trained small ranking heads; the QA backbone, quantizer and inference kernels are upstream capabilities. Different heads mean any paired comparison would concern complete pipelines, not the isolated quantizer.

旧 specialist 的全部 384 题明确降为开发材料：按预先固定规则分成训练 256 题/8 篇、校准 128 题/4 篇。训练数据只能拟合 scaler/coefficients；校准只能从 `[.5,.6,.7,.8,.85,.9,.95,.975,.99]` 选择最小合格阈值。余下四篇合格文章组成新评估 128 题、96 个 context 家族、64 可回答/64 不可回答。筛选和协议在拟合前提交，阈值在新评估前提交。只执行这一个追加方法，不继续搜索特征/模型直到正结果。

New evaluation articles: Ctenophora, Civil_disobedience, Yuan_dynasty and Harvard_University. This is public SQuAD2 dev, **not an official hidden test or model-unseen confirmation**. The upstream specialist already uses SQuAD2; exact article/context/question exclusion cannot remove upstream benchmark exposure, pretraining contamination, semantic duplicates, repeated project hypothesis decisions or four-article external-validity limits. Future runs on these published samples are reproductions.

## Actual result / 实际结果

| Phase | Pipeline | Threshold | Accepted precision | Answerable coverage | Correct answerable coverage | Unanswerable false accept | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Calibration 128 | FP32 + fitted head | none feasible | — | — | — | — | fail; no new evaluation |
| Calibration 128 | INT8 + fitted head | 0.7 | 33/35 = 94.29% | 34/64 = 53.13% | 33/64 = 51.56% | 1/64 = 1.56% | pass |
| Evaluation 128 | INT8 + frozen head | 0.7 | 27/27 = 100% | 27/64 = 42.19% | 27/64 = 42.19% | 0/64 = 0% | five point gates pass |

评估 invalid rate 为 0；回答 27/128，拒答 101/128，其中 37 个可回答问题被拒答。system EM 为 `91/128 = 71.09%`（27 正确答案 + 64 正确拒答），始终拒答为 `64/128`。因此不能写“128 题全部答对”“准确率 100%”或“高覆盖 QA”。正确率 100% 的分母只有被接受的 27 个答案。precision 的描述性 Wilson 95% 下界约为 87.54%，还低于 90%；该区间也未修正文章/context 依赖。通过预设样本点门槛不代表总体错误风险得到统计保证。

FP32 failed calibration and did not consume the new evaluation. Consequently there is **no new FP32 policy comparison, no paired compression noninferiority result, and no evidence that INT8 is inherently more accurate**. The INT8 task gate is evaluated independently of the compression claim. The earlier base-only CPU benchmark and the new full risk-pipeline performance measurement must be reported separately.

## Implementation defects and evidence / 实现缺陷与证据

首次训练在 Apple NumPy 2.2.6 的 BLAS matmul 对有限矩阵零初始化时报浮点异常，原始失败目录完整保存。独立合成测试复现规模相关异常；用固定 `einsum(optimize=False)` 替代矩阵产品，保留目标/solver/收敛门槛；修复后 5 个规模均通过，与 `math.fsum` 独立目标重算一致。另修复评分 API tuple 解包。它们发生在任何校准阈值结果和新评估之前，没有按结果修改方法。

[Failed attempt](../results/qa-risk-v1/training) · [Numerical diagnosis](../results/qa-risk-review-v1/numerical-fault) · [Successful fit and calibration](../results/qa-risk-v2/training) · [One fixed evaluation, including logits](../results/qa-risk-v2/evaluation-int8) · [Independent feature arithmetic](../results/qa-risk-review-v1/independent-feature-audit.md).

训练参数及标准化参数不进入 Git 或发布包。只分发特征矩阵、EM 标签、训练 loss/gradient trace、配置和参数摘要；用同版本 NumPy 在本地重建参数。参数摘要使用预先固定的十位小数表示，运行轨迹按数值容差核验；不声称任意平台逐位一致。原始模型权重也只保留本地，发布工具同时拒绝模型二进制和带训练参数的 head JSON。

## Reproduce / 复跑

下列验证不执行模型推理，但重建小排序头，需要 NumPy 2.2.6：

```bash
python -m pip install numpy==2.2.6
python -m experiments.verify_qa_risk --root results/qa-risk-v2
python results/qa-risk-review-v1/evaluation-independent.py
```

完整模型环境见 `requirements-qa-specialist.lock.txt`。先按 `docs/qa-specialist-study.md` 准备本地固定 ONNX 资产，再复现训练与**已公布**样本：

```bash
python -m experiments.qa_risk train --spec configs/qa-risk/study.json \
  --head-dir runs/my-risk-heads --output-dir runs/my-risk-training
python -m experiments.qa_risk evaluate --spec configs/qa-risk/study.json \
  --training-dir results/qa-risk-v2/training \
  --selection-sha256 b2999477ee82318a98266ef0674c22a526cddea1eec541c591a7bec7e354b74a \
  --asset-root runs/qa-specialist-assets-v1 --variant int8 \
  --output-dir runs/my-published-risk-reproduction
```

新输出目录必须不存在；旧记录不能覆盖。无 Git 源码归档中也应能重建小 head、验证证据并使用显式本地模型路径运行。上游权重下载和导出需要另外的模型环境，不依赖作者其他仓库。

## Complete risk-pipeline cost / 排序器完整开销

在同一台 M4 Max CPU、同 8 个固定单窗口输入上，另以 3 个独立 INT8 进程、每进程 8 warmup + 40 实测请求测得：进程中位数再取中位数为 **69.600 ms**，范围 69.192–69.992 ms；进程生命周期峰值 RSS 为 942,833,664–1,016,201,216 bytes。初始化/小 head 重建约 1.62–1.64 s，启动时完整证据核验另外计时。120 个测量包含 tokenizer、同步 CPU 推理、span 解码、遍历全部替代 span 的五特征计算、logistic score 和阈值决策；不含 JSON I/O 和进程启动。基础流水线 54.188 ms 的数字不包含这些新增成本，不能把其 1.33× 加速倍数套到本原型。[原始测量](../results/qa-risk-performance-v1)。

Three isolated processes measured the complete warm INT8 ranking pipeline at 69.600 ms median-of-process-medians on the eight published single-window inputs. This is not a concurrent service benchmark, tail-latency SLO, TTFT or token-generation speed. Startup identity/evidence verification and model load are separate operations.

按文章检查暴露了明显分布差异：Civil_disobedience 的 32 题全部拒答（可回答覆盖率 0/16），Ctenophora 为 6/16，Harvard_University 为 11/16，Yuan_dynasty 为 10/16。被拒的 37 个可回答问题中，有 19 个 raw span 本来 EM 正确。整体点门槛通过不能外推到每一篇文章或业务领域；排序器换来了精度，也丢弃了有效答案。

## Local entry and actual failure exercise / 本地入口与实际故障验收

`python -m experiments.serve_qa_specialist --asset-root runs/qa-specialist-assets-v1 --input-json request.json` 接受恰好两个字段 `context`、`question`；不允许附加 gold/labels。模型、tokenizer、协议和selection身份不符则关闭；超过字符/token/窗口上限不截断后悄悄作答。未实现可靠语言检测，英文是已经声明的适用范围，不是已自动识别的安全保证。

Actual reproduction exercised one answer, one unanswerable refusal, one answerable refusal, extra-gold-field rejection, over-token-limit rejection, empty-context rejection, and missing-asset rejection. All seven paths matched their contracts. [Raw functional run](../results/qa-prototype-reproduction-v1/run.json).

本次实测首次启动约 **18.04 s**（完整证据重算16.44 s、本地模型加载1.60 s、小head重建约0.002 s）。命令行每启动一次都会付出此成本；Python `load_service(...)` 后可以复用对象回答多个请求。69.60 ms 是热流水线范围，不含这一启动成本，也不含CLI JSON结构解析。该入口是本地研究工具，没有多用户HTTP服务、并发调度、认证、业务域适配或上线SLO证据。历史 `serve_grounded_qa` 仍保持关闭。
