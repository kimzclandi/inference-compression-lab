# Coverage experiment: disjoint competitors / 不重叠竞争片段覆盖率实验

**结果：候选训练收敛，但固定校准网格没有合格方案；没有提高可用覆盖率，不切换默认策略。**

**Result: fitting converged, but no prespecified calibration threshold passed all quality gates.** The candidate does not improve usable coverage and is not enabled in the default runtime. This is a development experiment, not an independent confirmation.

## 问题与固定假设 / Problem and hypothesis

原 INT8 排序策略在已公布的 128 题中接受 27 题且全部 EM 正确，可回答覆盖率 27/64。37 个被拒的可回答问题中，19 个 raw 答案正确；其它 18 个 raw 答案错误，需要改进抽取本身，不能只靠接受策略变成正确答案。当前 raw 输出固定时，正确回答覆盖的 oracle 上限为 46/64 = 71.88%；这是事后使用 gold 的诊断，不是可实现的部署策略。

19 个正确拒答中，18 个的最强不同答案与选中 span 在字符范围上重叠。原第三特征会把稍长/稍短的边界变体视为竞争者。固定假设：只和不重叠的不同答案竞争，可能减少不必要的低置信度；但重叠答案也可能代表真实边界歧义，删除这种信号可能增加错误接受。

The hypothesis was informed by already published outcomes. Those 128 rows are explicitly reused diagnostic data and can no longer support an independent evaluation claim for this experiment. An overlap observation alone does not establish that the feature causes incorrect refusals: 14 of the 18 raw-wrong answerable refusals also have overlapping competitors.

## 实现与验收 / Implementation and acceptance

- 只改第三列：`selected_margin - max(disjoint_alternative_margin)`。所有窗口仍参与竞争；不同规范化答案之外，再要求与选中答案的半开字符区间不重叠。相邻区间允许；同文本异位置仍按 normalized 答案排除。
- raw 答案、其它四个特征、训练标签、256/128 划分、训练器、L2、收敛门槛和九个校准阈值保持不变。没有模型推理、模型搜索或权重下载。
- 独立核对 raw/features/labels 后，非目标列保留原存档 binary64 数值。存档矩阵重放的数值边界没有扩大到任意平台重新生成浮点输入的训练稳定性。
- 如果不存在不重叠的不同答案，候选失败；不填 0、不删样本、不临时退回原定义。
- INT8 基线重建和候选拟合各一次。候选需通过全部原点门槛（precision ≥90%、可回答覆盖 ≥40%、正确回答覆盖 ≥35%、不可回答误接受 ≤10%、invalid ≤5%），且两种覆盖率均比校准基线增加至少 5 个百分点，才进入已公布 128 题的回顾性重放。
- 没有满足标准就停止，不追加模型、特征、优化器或阈值搜索。默认模型、门控、阈值及历史结果不变。

协议与实现先提交于 `016462caa689c30740e1c12e518008d55f1a8e51`，随后执行一次真实训练/校准流程。两臂均正常收敛（7 次迭代，最终最大梯度分别约 `4.46e-10` / `7.05e-10`，固定上限 `1e-8`）。学习参数仅保留内存；记录五列矩阵、标签、非参数训练轨迹、分数与参数摘要。代码与实验由 AI 辅助实现和执行。

源码：[`lab/qa_nonoverlap_gap.py`](../lab/qa_nonoverlap_gap.py)；[`experiments/qa_coverage_gap.py`](../experiments/qa_coverage_gap.py)。实验复用原五维数值 solver/scorer，公开记录另行标明候选特征语义；它不是兼容默认 serving policy 的新 head。

## 校准结果 / Calibration results

同一阈值 0.7；128 题中 64 可回答、64 不可回答。这不是历史后续评估的 42.19% 覆盖率分母，不能跨集合直接比较。

| 指标 | 原特征 | 不重叠竞争特征 |
|---|---:|---:|
| 接受 / EM 正确 | 35 / 33 | 36 / 32 |
| 接受答案精度 | 94.29% | 88.89% |
| 可回答覆盖 | 34/64 = 53.13% | 34/64 = 53.13% |
| 正确回答覆盖 | 33/64 = 51.56% | 32/64 = 50.00% |
| 不可回答误接受 | 1/64 | 2/64 |
| invalid | 0 | 0 |
| 完整网格选出的合格阈值 | 0.7 | 无 |

训练 241/256 行、校准 117/128 行的第三特征实际变化，未定义样本为 0。与上次选中窗口消融不同，本次不是零干预：实现变化确实传递到了特征、重拟合得分和决策。

逐题对照：新增接受 1 个正确答案和 2 个错误答案，同时丢失 2 个原先正确接受的答案；总接受数 +1，正确数 -1。可回答覆盖总数不变也掩盖了样本得失抵消。完整 [逐题变化](../results/qa-coverage-gap-review-v1/paired-at-0.7.json) 保留，不只展示恢复成功的那一道题。

候选阈值 0.6 时覆盖为 42/64，但精度仅 39/44 = 88.64%，仍低于原门槛；阈值 0.85 时接受 14 题且全对，但覆盖仅 14/64。完整预设网格均失败，未插入新的阈值。该结果反对在本批开发数据上采用这一候选，不能推出所有不重叠竞争方法普遍无效。

The candidate gains one correct answer, adds two incorrect answers and loses two previously correct answers at 0.7. Larger gaps do not imply larger fitted scores or better ranking: the scaler and coefficients are refitted. The calibration failure prevents even the planned retrospective replay; no new evaluation, quality-improvement or compression-noninferiority claim is made.

## 复现与审计 / Reproduction and audit

需要记录的 Python 环境和 `numpy==2.2.6`，输出必须使用新目录：

```bash
python -m experiments.qa_coverage_gap --output-dir runs/coverage-gap-reproduction
python -m experiments.verify_qa_coverage_gap \
  --root runs/coverage-gap-reproduction --output runs/coverage-gap-audit.json
```

第一个命令重建基线并训练候选；第二个独立枚举合法 span、核对五列及标签、重算完整阈值曲线和是否允许后续重放，不训练、不推理。保存的训练轨迹仅做一致性核验，不冒充独立重训证明。CI 执行历史记录的审计，不搜索新方案。

[冻结协议](../configs/qa-coverage-gap/study.json) · [本次完整结果](../results/qa-coverage-gap-v1) · [独立审计](../results/qa-coverage-gap-review-v1/audit.json) · [审计代码](../experiments/verify_qa_coverage_gap.py)

本轮保留原策略。后续若继续追求覆盖，应先扩充代表性训练/校准材料并保留新的验证数据，分别处理“抽错”和“正确但拒答”，再预先定义下一项实验。已公布材料只能复现或开发使用；已有少数文章不能证明业务分布的可靠性。增加覆盖必须同时报告正确回答覆盖、精度、误接受及逐题得失。
