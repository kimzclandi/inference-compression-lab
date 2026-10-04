# Expanded correctness-head training / 扩大正确性排序头训练

**训练收敛，校准失败，未运行预留评估；没有提高已验证的可用覆盖率。**

**Training converged; calibration failed; the reserved evaluation was not run. No improvement in validated usable coverage is established.** This is a controlled local ranking study with retained negative results, not a newly trained language model or a production QA service.

## 固定问题与实现 / Fixed question and implementation

原头只用 256 条五维特征训练。此次只扩大监督数据，检验更广的文章和正确/错误样本能否改善排序。基础模型仍是冻结的 `deepset/roberta-base-squad2`、revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`，使用相同 ORT 动态 INT8、FP32 QA projection、解码和五个特征。上游提供已训练 QA 模型，ORT 提供量化/推理实现；本项目实现数据隔离、特征与标签构建、小 logistic 头训练、校准、门控及证据核验。代码与执行由 AI 辅助完成。

The experiment adds official-train data, changing both sample count and source distribution. It cannot isolate a pure sample-size effect. It keeps the fixed five features, optimizer, L2 coefficient, numerical tolerance and threshold grid. Because the objective sums losses, increasing sample count also reduces the relative weight of the unchanged L2 penalty; it is not a constant-per-example-regularization comparison.

输入是原文、问题及冻结模型的窗口 logits；训练头的输入形状是 `(2304, 5)`，标签形状是 `(2304,)`。标签为 raw span 是否与某个 gold 答案规范化后完全匹配，不可回答题标签为 0。推理特征不读取 gold。头输出一个 logistic 分数，不能解释成有保证的正确概率。

选择器、数据、runner、协议与验收在推理前提交：`3556ee76a377e10ca949759b0a96e2463d7558ac`。协议 SHA256：`5d7d55aec9307c6c5d7f16ad10af90175778f29d215680e3529cf68dc38c37fb`。本轮原头重建一次、扩大头拟合一次；后续重建仅检验存档输入可复现性，不是额外搜索。

## 数据隔离及暴露边界 / Data roles and exposure

| 角色 | 题数 | 文章数 | 来源与用途 |
|---|---:|---:|---|
| 原训练 | 256 | 8 | 保留原 INT8 训练矩阵 |
| 新增训练 | 2,048 | 64 | 官方 train；每篇每类 16 题，每段每类最多 1 题 |
| 新校准 | 192 | 4 | 官方 dev；每篇每类 24 题，每段每类最多 2 题 |
| 预留评估 | 192 | 4 | 官方 dev；同校准配额，未推理 |

各角色按文章隔离；新增角色排除历史 inventory 和所有原头 cohort 的相同 ID、大小写/空白规范化后的原文及问题。新增训练排除全部历史文章标题。公开 dev 剩余文章不足以完全排除所有旧 Qwen 标题，因此在推理前明确：新校准/评估排除所有旧正确性头 cohort 的文章，但允许旧 Qwen 研究的标题，仍排除其已记录的相同原文和问题。此次校准/评估的八篇文章标题均在历史 Qwen inventory 中；这不是完全新的文章级外部测试，也不是临时根据结果更换样本。

Upstream RoBERTa was already trained on SQuAD2 train and assessed on dev. These local roles do not establish upstream-model-unseen evaluation. Selection does not exclude semantic near-duplicates, and class balancing is not deployment prevalence. Only four calibration articles limit transfer. The published reserved set remains unused by this experiment, but future use must disclose that it has been published.

## 预设门槛 / Prespecified acceptance

阈值网格固定为 `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99`，选最低合格值。原门槛保持不变：接受精度 ≥90%、可回答覆盖 ≥40%、正确回答覆盖 ≥35%、不可回答误接受 ≤10%、invalid ≤5%。

预定比较原头固定 0.7、原头在新校准集重新选阈值、扩大头在同一新校准集选阈值。扩大头校准通过才允许评估。评估阶段原计划要求：全部质量门槛通过，正确覆盖比同样重校准的原头高至少 5 个百分点，并且按文章/原文族配对 bootstrap 的 95% 下界大于 0。由于校准失败，没有计算评估指标、bootstrap 或宣布提升。

## 实际训练和校准 / Observed training and calibration

原头 7 次迭代收敛，最大梯度 `4.46e-10`；扩大头 8 次迭代收敛，最大梯度 `6.33e-10`，均低于固定 `1e-8`。训练成功在这里仅指优化器收敛，不等于质量验收成功。不同训练样本数的总 loss 不能直接比较大小。

新校准集 96 题可回答、96 题不可回答；raw span 有 80 个 EM 正确，即正确覆盖的事后 oracle 上限为 80/96。排序只能决定接受/拒答，不能把剩余 16 个抽错答案修正。下表展示原网格中的代表点，**没有任何点成为被选中阈值**；完整 18 点曲线和每篇文章计数均保留。

| 方法 / 阈值 | 接受 / 正确 | 接受精度 | 可回答覆盖 | 正确覆盖 | 不可回答误接受 | 门槛 |
|---|---:|---:|---:|---:|---:|---|
| 原头 / 0.7 | 56 / 47 | 83.93% | 48/96 = 50.00% | 47/96 = 48.96% | 8/96 | 失败 |
| 扩大头 / 0.7 | 58 / 48 | 82.76% | 49/96 = 51.04% | 48/96 = 50.00% | 9/96 | 失败 |
| 原头 / 0.8 | 35 / 32 | 91.43% | 33/96 = 34.38% | 32/96 = 33.33% | 2/96 | 失败 |
| 扩大头 / 0.8 | 45 / 39 | 86.67% | 39/96 = 40.63% | 39/96 = 40.63% | 6/96 | 失败 |
| 扩大头 / 0.9 | 26 / 24 | 92.31% | 24/96 = 25.00% | 24/96 = 25.00% | 2/96 | 失败 |

所有点 invalid 均为 0。固定 0.7 逐题变化：新增 2 个正确和 2 个错误接受，丢失 1 个正确和 1 个错误接受；净增加 1 个正确答案不构成质量提升。原头在新材料也失败，说明历史 128 题样本通过不能外推。不同 cohort 的 42.19% 与本表覆盖不能当作配对提升。

Both heads failed every predefined calibration threshold. The expanded head did learn a converged function, but it did not solve the precision/coverage trade-off on the new calibration articles. This does not establish that more representative data or other methods can never help. It rejects this particular fixed expansion as a replacement policy. The runtime default is unchanged; the historical prototype remains limited to its documented scope.

## 独立核验、复现及停止 / Audit, reproduction and stopping

全部 2,496 条旧/新训练与新校准记录独立枚举合法 spans，检查解码、特征、标签；另验证三个新角色共 2,432 条输入的排除/配额和官方源文件对应关系。核验后使用原存档浮点矩阵重放训练，比较参数摘要、训练轨迹和独立公式计算的校准分数。这个边界不扩展成“任意平台重新生成浮点特征再训练均稳定”。学习参数只保存在本地忽略的 `runs/` 或内存中。

无需模型、无需 Git、仅需 `numpy==2.2.6` 的完整核验：

```bash
python -m experiments.verify_qa_expanded --replay-fit --output runs/expanded-audit.json
python -m experiments.diagnose_qa_expanded --output-dir runs/expanded-diagnosis
python -m unittest discover -s tests -v
```

如本地已取得归属说明中的原始 train/dev 文件，可为 verifier 增加 `--source-root /path/to/squad`，逐条核对原始来源；CI 不联网下载这些源文件。所有输出必须使用新路径。

只重建存档训练头（不推理、不重新选择数据）：

```bash
python -m experiments.qa_expanded rebuild \
  --training-dir results/qa-expanded-v1/training \
  --head-dir runs/expanded-rebuilt-heads --output-dir runs/expanded-rebuild
```

要复现整条开发链，需要 [固定模型资产及推理环境](release-reproduction.md)。在新目录收集，再训练：

```bash
python -m experiments.qa_expanded collect --split train_new \
  --asset-root /path/to/assets --output-dir runs/expanded-train-collection
python -m experiments.qa_expanded collect --split calibration \
  --asset-root /path/to/assets --output-dir runs/expanded-cal-collection
python -m experiments.qa_expanded train \
  --train-dir runs/expanded-train-collection --cal-dir runs/expanded-cal-collection \
  --head-dir runs/expanded-local-heads --output-dir runs/expanded-training
```

当前 selection 不合格，`evaluate` 在加载 head 或基础模型前拒绝执行，回归测试验证不会创建评估目录。不要为了产生评估结果手动改 selection、阈值网格、质量或收敛门槛。已保存失败必须保留。

[协议](../configs/qa-expanded/study.json) · [数据归属](../configs/qa-expanded/ATTRIBUTION.md) · [训练/校准原始证据](../results/qa-expanded-v1) · [逐题诊断与完整曲线](../results/qa-expanded-review-v1/diagnosis.json) · [独立核验](../results/qa-expanded-audit-v1/verification.json) · [验证器](../experiments/verify_qa_expanded.py)
