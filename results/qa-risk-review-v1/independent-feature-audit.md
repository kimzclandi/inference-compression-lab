# Independent correctness-feature audit / 独立正确性特征审计

结论：3 个独立闭式手算样例以及固定抽取的 2 个真实训练样例均与实现一致，未发现五特征计算或该两条 EM 标签错误。审计没有拟合 head、修改特征或阈值，也没有运行或读取新的 risk evaluation。

五个输入特征依次是：选中 span 与同窗 CLS 的 logit 差；选中窗 start/end 的 context+CLS 两个 softmax 对应 log 概率之和；选中 margin 与**跨全部窗口、规范化后不同答案**的最高 margin 之差；`log(1+答案token数)`；`log(1+窗口数)`。标签不进入特征提取或线上排序。

第二项的名称需要精确解释：它是两个端点分布概率的乘积取 log，归一化集合包括所有 context token 及 CLS，排除 question/special/pad；**不是对所有合法 span 的联合归一化**，也没有给非法 start/end 配对重新归一化。zero-offset context token 虽不能作答案端点，仍进入这个 token 归一化集合。该设计是固定特征定义，不能把这个值或后续 logistic score 称为已校准的答案正确率。

三个手算反例覆盖：

1. `cat dog`：start 的 CLS/cat/dog 权重为 2/3/1，end 为 2/1/4。选中 `cat dog` 的 span/null margin 为 `log(12/4)=log3`；第二项为 `log(12/(6×7))`。额外 question token 的 logit=100 必须被排除。竞争答案 `dog` 的 margin 为 0，因此 gap 为 `log3`。
2. `The cat cat dog`：窗口 A 选中 `The cat`，score=64、null=1；窗口 B 的 `cat` score=16，但与 `The cat` 规范化后相同，不能充当不同答案。真正最高竞争者是窗口 B 的 `cat dog`，score=12；gap=`log(64/12)`。这同时验证跨窗口搜索和规范化去重。
3. `a  b`：中间零长度 context token 的 start/end 权重均 100。它不能作端点，但两个 softmax 分母均为 106；选中 span 的第二项为 `log(16/(106×106))`，答案跨度含 3 个 token，因此长度特征为 `log4`。

真实记录按事先固定的 `SHA256("independent-feature-audit-" + variant + id)` 最小值选取，未按预测好坏筛选：

| Precision | ID | Margin | Joint endpoint log score | Alternative gap | EM target |
|---|---|---:|---:|---:|---:|
| FP32 | `5ad248f7d7d075001a428b8f` | −9.57249737 | −9.61285531 | 0.10482633 | 0 |
| INT8 | `5a7b485921c2de001afe9e52` | −4.16217032 | −4.77949320 | 0.01552251 | 0 |

两条均为 `is_impossible=true`，无 gold span，target=0。审计 JSON 中 `answerability` 字段直接复制该 `is_impossible` 布尔值；它不是“可回答=true”的意思。另有手算标签例子验证大小写、标点和冠词归一化匹配，以及部分答案不得给 EM=1。

The training target is exact match after the existing SQuAD-style normalization: lowercasing, ASCII punctuation removal, article removal, whitespace collapse. Matching any annotated gold yields 1; an impossible question's empty gold list yields 0. This target measures benchmark EM correctness, not semantic truth or support strength. The chosen span can be a wrong answer despite appearing in the passage. The alternative gap compares answer strings under that same normalization, so semantically equivalent paraphrases may still be distinct, and normalized-identical spans at different locations are pooled.

Expected feature values were computed without importing the project feature extractor or metric helpers; synthetic values additionally have explicit closed forms. The extractor was called only as the implementation under test. The real check independently re-enumerates legal spans from raw logits. Two real samples are an audit sample, not exhaustive verification of all training labels, optimizer convergence, population calibration, or deployment acceptance.

```bash
python3 -B results/qa-risk-review-v1/independent-feature-audit.py \
  --training-dir results/qa-risk-v2/training \
  --output /new/nonexistent/feature-audit.json
```

Evidence: `independent-feature-audit.json` binds the successful training protocol `365a1ff260d9f887558bc5ce5421e4028702ad435717aefc8f17787cea08081b`, feature-table hashes, current source hashes and fixed sample rule. The separate synthetic record was saved before the real-sample inspection.
