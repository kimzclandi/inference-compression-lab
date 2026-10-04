# Frozen semantic correctness head / 冻结语义表征正确性头

**投影一致性已修复；开发质量门槛仍失败；预留评估未运行。**
**The training/inference projection mismatch is repaired, but development quality gates still fail. The reserved evaluation was not run.** No coverage-improvement or deployment claim is made.

## 动机、协议与身份

五特征树和十六特征树均未通过门槛后，本研究检查冻结模型的语义表征是否提供置信度摘要以外的信息。它是顺序、适应性开发，已经受前面开发集结果影响，不是独立确认。基础模型、INT8 权重、QA projection 与答案抽取不训练、不替换。

初始协议和实现先提交于 `92e9d35`；评估算法在拟合前另行提交于 `81e43a9`。从同一 ONNX 图末层 LayerNorm 添加输出，提取所选窗口 CLS 向量和所选 span 的 mean pooling，各 768 维。2,304 训练 + 192 开发行共提取 2,496 行；所选窗口的起止 logits 与历史记录最大差异均为 0。

语义输入经仅在训练集拟合的 StandardScaler 和固定 64 维随机化 PCA，拼接另行标准化的原五特征，得到 69 维 logistic 输入。固定 C=0.1、L-BFGS、max_iter=1000、tol=1e-6、seed=2026100417；无维数/C/阈值搜索。这里的分类器是线性的；继承协议中“nonlinear challenger”的措辞已在拟合前用 [说明](../configs/qa-semantic/CLARIFICATION.md) 澄清。

[原协议](../configs/qa-semantic/study.json) · [评估协议](../configs/qa-semantic/evaluation.json) · [原始表征与训练](../results/qa-semantic-v1)

## 发现并修复的实际缺陷

原 sklearn Pipeline 在拟合时调用随机化 PCA `fit_transform`，预测时调用 `transform`。同一训练样本两个路径的数值最大差异为 **0.34327576**；这是该近似分解路径的观察，不是宣称所有 PCA 都存在此差异。分类器在训练路径上的最大梯度约 `7.69e-7`，切换到实际推理投影后约 `8.48e-4`。

修复方案在 `859b62b` 提交后执行：先拟合训练变换，再统一通过 `transform` 产生训练输入，最后拟合分类器。保持相同表征、PCA 维数、随机种子、C、停止配置、数据和质量门槛。原版结果完整保留。回归测试捕获分类器实际收到的训练特征，核对与随后 `transform` 的结果一致。

修复版推理路径上独立计算的最终 logistic 目标最大梯度约 **9.47e-7**。该数值属于本新模型目标；不能改写旧五维 Newton solver 的固定 `1e-8` 门槛或原数值边界。

[修复协议](../configs/qa-semantic-fixed/study.json) · [实现](../experiments/qa_semantic_fixed.py) · [修复版记录](../results/qa-semantic-fixed-v1) · [回归测试](../tests/test_qa_semantic_projection.py)

## 开发结果与停止

沿用全部原门槛：precision ≥90%、可回答覆盖 ≥40%、正确覆盖 ≥35%、不可回答误接受 ≤10%、invalid ≤5%。沿用九个阈值，选最低合格点；两版均无合格点。

| 方法 / 阈值 | 接受 / 正确 | 接受精度 | 可回答覆盖 | 正确覆盖 | 不可回答误接受 |
|---|---:|---:|---:|---:|---:|
| 原语义头 / 0.8 | 50 / 42 | 84.00% | 43/96 = 44.79% | 42/96 = 43.75% | 7/96 |
| 修复语义头 / 0.8 | 50 / 42 | 84.00% | 43/96 = 44.79% | 42/96 = 43.75% | 7/96 |
| 原语义头 / 0.9 | 34 / 30 | 88.24% | 30/96 = 31.25% | 30/96 = 31.25% | 4/96 |
| 修复语义头 / 0.9 | 35 / 31 | 88.57% | 31/96 = 32.29% | 31/96 = 32.29% | 4/96 |

Invalid 均为 0。相同阈值 0.8 的聚合计数相同不证明得分/机制等价；完整逐题分数和九点曲线均保存。修复数值一致性不自动保证质量提高。选择失败后评估入口在读取本地 head 或加载模型前关闭；没有使用预留样本寻找更好的阈值。

The hidden states were extracted from the same frozen upstream model. Only the PCA/scalers/classifier were fitted locally. Existing public-benchmark exposure, reused development data, few article groups and balanced answerability limit external validity. No upstream-model-unseen or production guarantee is established, and failure does not prove semantic correctness modeling is universally ineffective.

## 数值告警、独立核验与成本

macOS 上的 NumPy/BLAS 路径出现 `matmul` divide-by-zero/overflow/invalid RuntimeWarning；这些告警保留，不把“未出现 sklearn ConvergenceWarning”写成“完全无数值告警”。末态特征和分数均为有限值，使用非 BLAS 的 `einsum` 重算投影/概率，与修复版分数最大差约 `4.44e-16`。这些检查不能解释或证明所有中间告警均无害，也不能保证任意平台重新提取 hidden states 后训练稳定。

独立按 ID 哈希固定选 8 行检查：末层 hidden 经原 FP32 QA projection 可恢复 logits（最大差约 `1.34e-5`），独立 span pooling 与存档向量差小于 `6.36e-7`。这是实现/数值抽查，不是新增质量评估。全部向量的身份、形状和有限性、训练来源以及 head 重建分别核验。

跨平台存档重建采用显式概率差 `1e-6`，并额外要求九个阈值的逐题决策完全相同。它不是精确字节相同的参数保证；也未更改质量门槛。原/修复版告警、差异和最终梯度见 [审计](../results/qa-semantic-review-v1)。

NPZ 文件是冻结模型对公开样本产生的 activations，不是模型、PCA 或分类器参数。新增 ONNX 输出图和学习参数只存放本地忽略的 `runs/`，不分发。表征提取和为评估准备的额外前向会增加成本，本轮未做候选端到端延迟验收，不能沿用历史 69.600 ms 或 1.33×。

## 无模型重建

使用 [固定依赖](../configs/qa-nonlinear/requirements.txt)，输出必须全新。CLI 仅重建已保存的失败实验：

```bash
python -m pip install -r configs/qa-nonlinear/requirements.txt
OMP_NUM_THREADS=1 python -m experiments.verify_qa_semantic --output runs/semantic-original-audit.json
OMP_NUM_THREADS=1 python -m experiments.verify_qa_semantic --fixed \
  --root results/qa-semantic-fixed-v1 --output runs/semantic-fixed-audit.json
python -m unittest discover -s tests -v
```

真实表征抽查另需原固定模型资产和 ONNX 运行环境，调用 `experiments.check_qa_semantic_tap`。该命令不能在没有资产时被称为已执行。代码与实验由 AI 辅助实现和执行；默认本地 QA 策略保持不变，历史失败没有改判。

## Linux CI failure retained

At commit `b517530`, Linux Python 3.11 and 3.12 both failed the original
semantic head's fixed `1e-6` score-reproduction tolerance. The repaired-head
check was not reached in that run. Local replay success is therefore not a
cross-platform pass. CI now executes both checks separately, retains their
maximum score differences and changed-decision counts, and still fails on the
same tolerance violation. It does not relax a numerical or quality gate.
The semantic branch is not fully technically accepted while this remains
unresolved; the earlier nonlinear branch has separate passing checks.
