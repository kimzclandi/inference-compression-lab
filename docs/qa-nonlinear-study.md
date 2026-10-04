# Nonlinear ranking challengers / 非线性排序候选

两项新增候选均完成训练，但没有通过原开发校准门槛。预留 192 题没有推理，默认策略没有改变。Coverage alone passes at some thresholds; the joint precision/coverage gate does not. No improvement or deployment claim is made.

本轮针对原五特征 logistic 头扩大训练后仍失败，分两步检验表示能力：先固定小型 histogram gradient boosting classifier；失败后追加一个明确标为后续开发实验的 richer-feature 候选。第二项沿用完全相同的树参数，只增加 11 个不读取 gold 的特征。两项均只拟合一次，没有模型、树深、随机种子或阈值网格搜索。它们是顺序、适应性开发，不能算两个独立确认；不掩盖研究者已看过开发结果。

## 冻结与环境

第一项实现/协议在 `df9ef975514f44e664ae4840041f8e7b0b66bdf4` 冻结。第一次尝试在导入 SciPy 1.15.3 的 macOS 二进制时失败，未进入拟合，完整记录位于 `results/qa-nonlinear-v1/training/`。在独立依赖目录验证 SciPy 1.16.2 可以导入后，仅更新依赖版本并再次提交，未改变数据、树参数、网格或质量门槛；成功执行记录位于 `results/qa-nonlinear-v2/`。

此导入症状与 [SciPy 上游记录](https://github.com/scipy/scipy/issues/25635) 相符；这里仅声称本机 1.16.2 导入与训练通过，不泛化为所有 macOS 的通用修复。原推理环境未被修改。依赖固定在 [requirements](../configs/qa-nonlinear/requirements.txt)。

采用 scikit-learn 1.6.1 [HistGradientBoostingClassifier](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)：log loss，150 次 boosting、学习率 0.05、最多 7 叶/深度 3、叶最少 30 行、L2=1、63 bins、关闭提前停止、固定 seed=2026100416、限制单线程。框架提供树训练实现；项目贡献是受控比较、数据/特征链路和验收。固定迭代完成不是梯度收敛证明。

第二项在看到第一项失败后另行冻结于 `7d18527`，见 [协议](../configs/qa-rich/study.json)。输入从五维变成十六维，新增起点/终点各自的归一化熵、CLS log probability、选中端点 log probability、与最强其它端点的 logit 差，以及问题/答案词集合重叠、问题词数 log1p、答案相对字符位置。特征只读问题、原文与原始输出，不使用 gold 或 answerability；原五列、标签及 raw 答案保留。新增信号可能区分原聚合特征没有体现的情况，但实验失败不能证明信息完全无用或某项信号是唯一瓶颈。

## 数据与验收

训练仍为原 256 + 新 2,048 = 2,304 题。使用已公布的 192 题开发校准；它已经影响研究选择，不是新评估。仅候选通过全部旧门槛才固定 selection，并在上一轮未推理的 192 题评估一次。本轮目标是绝对点门槛：precision ≥90%、可回答覆盖 ≥40%、正确覆盖 ≥35%、不可回答误接受 ≤10%、invalid ≤5%；没有把绝对门槛改写成配对提升或总体风险保证。

九个阈值沿用 `0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99`，选最低合格值。没有合格值即停止该候选。公开 SQuAD、模型上游暴露、旧 Qwen 文章标题重复、样本均衡及少数文章的边界沿用 [扩样研究](qa-expanded-ranking-study.md)。

| 开发方法 / 阈值 | 接受 / 正确 | 接受精度 | 可回答覆盖 | 正确覆盖 | 不可回答误接受 |
|---|---:|---:|---:|---:|---:|
| 五特征树 / 0.85 | 43 / 38 | 88.37% | 39/96 = 40.63% | 38/96 = 39.58% | 4/96 |
| 五特征树 / 0.9 | 25 / 23 | 92.00% | 23/96 = 23.96% | 23/96 = 23.96% | 2/96 |
| 十六特征树 / 0.85 | 45 / 39 | 86.67% | 40/96 = 41.67% | 39/96 = 40.63% | 5/96 |
| 十六特征树 / 0.9 | 27 / 24 | 88.89% | 24/96 = 25.00% | 24/96 = 25.00% | 3/96 |

所有点 invalid=0，完整网格保留于各 `selection.json`。只看覆盖列会误报成功。No candidate satisfied all prespecified development gates, so no reserved evaluation or default-policy replacement occurred. These failures do not establish that nonlinear ranking, richer features or more data are universally ineffective.

## 重建和独立审计

在隔离 Python 环境安装上述固定 requirements。标准库 tests 不需要新训练依赖；重建需要这些依赖。源码归档不包含模型权重或树参数，训练 pickle 只留在忽略的 `runs/`。不加载外部 pickle。

```bash
python -m pip install -r configs/qa-nonlinear/requirements.txt
OMP_NUM_THREADS=1 python -m experiments.qa_nonlinear verify \
  --folder results/qa-nonlinear-v2 --output runs/nonlinear-audit.json
OMP_NUM_THREADS=1 python -m experiments.qa_rich verify \
  --folder results/qa-rich-v1 --output runs/rich-audit.json
python -m experiments.verify_qa_rich_inputs --output runs/rich-input-audit.json
python -m unittest discover -s tests -v
```

前两个命令用存档矩阵重建，并重算开发分数、完整门槛选择和评估是否被禁止；不是新推理或新调参。最后一个验证器用独立 softmax/熵公式逐行核对全部 2,496 行新增特征，并确认原五列、标签、答案保持不变。原五列的完整 raw 独立核验仍由已有扩样 verifier/CI 执行。

[五特征树记录](../results/qa-nonlinear-v2/training) · [十六特征树记录](../results/qa-rich-v1/training) · [新增特征矩阵](../results/qa-rich-input-v1) · [特征实现](../lab/qa_rich_features.py)。代码和执行由 AI 辅助完成。没有重新测延迟，也没有新的 FP32 配对量化非劣证据。
