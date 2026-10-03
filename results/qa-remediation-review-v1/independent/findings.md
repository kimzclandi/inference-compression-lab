# 独立统计复核与失败案例 / Independent statistical audit and failure cases

审计脚本 `audit.py` 只使用 Python 标准库，未导入项目的评分、门禁或验证函数，也没有读取 confirmation 数据。先在已公开开发集检查，再在两个完整 calibration run 上独立运行。脚本使用独立的规范化/词频交集公式重算逐题 EM/F1，核对原始预测清单与 run 内预测完全一致，并以 Yes/No 原始 logit 的双类别 softmax 重算全部拒答排序分。

The audit imports no project scorer or gate. It reads the named completed development/calibration runs only; the held-out confirmation data was not opened. The Yes/No score is an uncalibrated ranking signal, not an estimated correctness probability.

## 结果 / Results

| 项目 | FP16 | Q8 |
|---|---:|---:|
| Calibration 原始 EM | 74/128 = 57.8125% | 76/128 = 59.375% |
| 原始 F1 | 0.6107762897 | 0.6264012897 |
| 非抽取/非法输出 | 8/128 = 6.25% | 8/128 = 6.25% |
| 满足全部五项约束的预设阈值 | 0/9 | 0/9 |
| 阈值0.995的返回答案EM精度 | 12/13 = 92.31% | 12/13 = 92.31% |
| 阈值0.995的可回答题覆盖率 | 13/64 = 20.31% | 13/64 = 20.31% |
| 阈值0.995的正确可回答题覆盖率 | 12/64 = 18.75% | 12/64 = 18.75% |
| 记录的参数物理存储字节 | 3,087,428,608 | 1,640,332,288 |

所有阈值均失败。非法输出率6.25%本身就超过预设5%上限；提高阈值可提高返回答案的EM精度，但0.995时两种覆盖率均明显低于40%与35%的要求。阈值0.999虽然达到100%样本精度，仅保留FP16的3个、Q8的2个答案，不构成可用系统的证据。

All nine fixed thresholds fail. Raising the threshold improves observed precision by refusing more requests, at the cost of inadequate coverage. Therefore this audit independently supports stopping before confirmation and keeping the answer entry point closed.

Q8/FP16参数字节比为 **0.5312939978**。此数值由run里保存的各dtype字节计数重新相加并核对总数，属于对实际运行布局记录的独立算术审查；不是重新加载张量、测量RSS或证明端到端加速。两个原始分数更高的Q8样本不能替代预设质量门槛，也不是新的确认性非劣效证明。

## 三个具体错误 / Three concrete errors

以下均来自 **calibration-q8**；完整原文、问题、参考答案、预测、logits和提示哈希见`calibration-audit.json`的`runs.q8.examples`。这些例子用于解释已观察到的失败，不用于修改本轮提示、阈值或评分。

1. **关系方向错误，原文抽取和高分仍不够。**

   ID：`5ad0483977cf76001a686f8a`。原文说Orange County包含Downtown Santa Ana等区域；问题却问Downtown Santa Ana包含哪个county。模型返回原文中的`Orange County`，同模型自审给出排序分 **0.990140**，Yes/No合计质量也高达约0.999891。按固定数据标注，此题不可回答。关键词和span均在原文中，但包含关系方向倒置；同模型再次判断保留了同类错误。不能把该分数解释为“99.01%概率正确”。

   The candidate is an exact substring, but it answers an inverted relationship. A second call to the same model can repeat the first call's reasoning error even when both binary-label mass and the conditional Yes score are high.

2. **语义大致正确，答案粒度违反固定任务。**

   ID：`571cc8815efbb31900334df0`。问题问极地水域为何支持更多生命，参考span为`higher oxygen content`等。模型复制了包含主语、结果与原因的整段分句，自审分 **0.998218**；原始F1为 **0.6**，EM为 **0**。这类失败不应被描述成虚构事实：它暴露的是“答案受到原文支持”和“输出符合最短抽取答案要求”不是同一判定目标。不能通过临时宽松评分把本轮EM提升；若未来产品允许长答案，需要新的预先固定任务定义。

   This example is largely supported semantically but overlong for the fixed exact-answer contract. A support judge does not necessarily enforce answer granularity. The historical scorer and current thresholds remain unchanged.

3. **词形改写破坏严格原文span合同。**

   ID：`5726da89dd62a815002e92b4`。问题询问GPhC的主要职责。模型输出`regulating the practice ...`，而参考原文使用`regulates the practice ...`。两者词语高度重合，原始F1为 **0.857143**，但生成文本不是原文连续片段，合同将其标为`invalid`，不进入自审，也不给正确拒答的分数。此处合同做对了安全抑制，但抑制不能说明模型任务质量已经合格。

   A minor grammatical rewrite is still nonextractive under the frozen span contract. Suppressing it is correct contract enforcement, not evidence that the model reliably produces valid answers.

## 复跑 / Reproduce

```sh
python3 results/qa-remediation-review-v1/independent/audit.py \
  --run fp16=results/qa-remediation-v1/calibration-fp16 \
  --run q8=results/qa-remediation-v1/calibration-q8 \
  --output /tmp/qa-independent-calibration-audit-new.json
```

必须使用不存在的新输出文件；脚本拒绝覆盖报告。结果含study、脚本和输入文件SHA256、完整九阈值曲线、五门槛逐项结果及上述原始案例。报告没有改变旧block10确认失败，也没有把本地prototype称为生产部署系统。
