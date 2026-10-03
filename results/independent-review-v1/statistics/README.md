# Independent confirmation audit / 确认实验独立统计审查

This directory contains a new audit of **existing disclosed predictions**, not a
new confirmation dataset or additional model inference. `audit_statistics.py`
uses only the Python standard library and imports none of the project scorer,
selector or bootstrap implementation. Historic evidence was not modified.

本目录是对已经公布的预测进行独立复算，不是新确认集，也没有新增模型推理。
脚本只用 Python 标准库，未导入本项目的评分、选样或 bootstrap 实现，历史证据未修改。

## What was actually rerun / 本轮实际执行

```bash
python3 results/independent-review-v1/statistics/audit_statistics.py \
  --raw /path/to/squad-dev-v2.0.json \
  --output /tmp/confirmation-independent-audit-new
```

Use a nonexistent output directory. The pinned raw SQuAD source is identified by
SHA256 `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`.
Omit `--raw` for prediction/scoring/bootstrap checks without cohort reconstruction.
Git is optional: archive runs skip the Git chronology check and retain all other
checks. This audit command does not download data or model weights.

输出目录必须不存在。传入固定 SQuAD 源文件可复核选样，不传 `--raw` 则只复算预测、
评分、统计和记录内的排除关系。无 Git 源码包会明确跳过 Git 时序核对，其余可运行。

Recorded result: [run-20261004/audit.json](run-20261004/audit.json).
All 640 row scores, all variant aggregates, and all three seeded 5,000-replicate
paired intervals matched the frozen reports. Independent cohort reconstruction
matched all 128 records and the article order exactly. The dataset has 64
answerable and 64 unanswerable questions, 85 context families and four articles.

实跑结果：640 条逐题 EM/F1、分类与汇总全部匹配；三个配对比较的 5,000 次分组
bootstrap 区间匹配；从固定原始 SQuAD 文件独立重建出的 128 条记录及文章顺序完全一致。
新统计脚本没有发现会改变确认结论的算术缺陷。

|Variant / 变体|EM count / 128|F1|
|---|---:|---:|
|FP16|33|0.3251811990|
|Q4|26|0.2810016760|
|Q8|34|0.3290874490|
|Q4 + original FP16 block 10|29|0.2834784527|
|Q4 + original FP16 block 22|25|0.2757368977|

Block 10 gains 3 and loses 0 against Q4: +2.34375 percentage points, with the
prespecified interval [0, 5.072463768] percentage points. Its F1 drop versus FP16
is 4.170274627 percentage points, exceeding the two-point limit. The conjunction
of the three prespecified criteria is **false**. This does not prove zero benefit.

block 10 相比 Q4 新增正确 3 题、丢失 0 题；预设区间下界为零且 F1 退化超过门槛，
确认失败保持不变。失败不等于已证明干预完全无效。所有模型在 64 个不可回答问题上
正确拒答数都是零，整体 EM 低于始终拒答基线的 50%；此系统未达到问答部署质量。

## New interpretation checks / 本轮新增解释核对

- Two of the three EM gains shorten an answer-bearing full sentence to its gold
  span (temperature and Trout River). The third changes `4` to `42`; the original
  source accepts `28` or `42`. Report this as exact-extraction/answer-output
  improvement on three items, not uniformly recovered factual reasoning.
- Normalized EM and the exact-substring output contract measure different things.
  Q4 and block 10 each have two normalized-EM-correct outputs that fail the
  substring contract. Thus Q4's 26 EM matches and 24 `correct` categories are not
  an arithmetic inconsistency. Preserve both measures; do not rename
  `format_valid` to human-judged formatting quality.
- Seven protocol/data/scorer/runner artifacts match commit `5ca4191` byte for
  byte. Its local commit timestamp precedes the recorded inference start by
  36.151966 seconds. This supports the recorded ordering but is not independent
  timestamp attestation or proof of no unrecorded trials.
- The paired family resampling and its changing denominator implement the
  stated protocol. Articles themselves are not resampled, so the interval is
  conditional on these four articles. Exact-context families do not exhaust all
  semantic dependencies.
- Exact exclusion is verified against the recorded known-data inventory. This
  cannot establish complete exposure inventory, semantic decontamination or
  absence of pretraining contamination. Published samples remain disclosed on
  every future rerun.

3 题收益中 2 题主要是输出收敛为精确答案片段；归一化 EM 与原文子串约束应分别展示。
本地时序与哈希是可追溯证据，不是第三方审计认证。四篇文章和精确 context 的范围
不能扩展为全领域有效性。未验证的旧数据接触、预训练污染与语义近重复仍是限制。

## Recruiting interpretation / 招聘价值边界

The defensible contribution is a controlled, traceable diagnostic experiment:
frozen hypotheses, matched intervention/control, paired family uncertainty,
original predictions, and a decision to retain a failed confirmation. It is not
a new quantization algorithm or verified general-purpose quality improvement.
Framework quantization itself is supplied by MLX; code volume, documentation
volume and check counts do not establish algorithmic novelty or user mastery.

可支持的招聘价值是固定干预、等成本对照、选择与确认分离、配对分组不确定性和保留
失败结论；不能支持新量化算法、通用质量增益、部署质量或本人已掌握。单看本统计
主线，适合支撑模型压缩评测与工程判断；作为整个加速岗位核心项目的判断还必须结合
真实性能、模型身份、缓存机制和用户亲自复跑证据。

The existing per-article gain/loss implementation exercise in
`docs/confirmation-learning.md` remains for the user to perform and explain.
This audit does not mark it completed or claim the user authored this audit.

既有“本人新增逐文章 gain/loss 并亲自复算解释”的任务仍未完成；本轮审查不将其代记为
本人掌握。原始预测、独立逐题分数与全部 bootstrap 复制样本均保存供复查。
