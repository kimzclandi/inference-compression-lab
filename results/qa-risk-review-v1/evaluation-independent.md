# Independent final 128-row assessment / 最终 128 题独立验收

独立重算确认：INT8 + 已冻结正确性排序 head 在阈值 **0.7** 下，通过预设的五项**整体点估计**门槛。128 条 candidate 的原文 span/offset/context hash、head score 阈值比较、原始和处理后 EM/F1、全部拒答记录均与发布摘要一致。该审计仅用标准库，没有导入项目 scorer/gate，也没有推理或搜索其它阈值。

| Metric | Recomputed value |
|---|---:|
| 接受答案 / 接受且 EM 正确 | 27 / 27 |
| 接受答案 EM precision | 100%（27 个观察） |
| 可回答覆盖率 / 正确可回答覆盖率 | 42.1875% / 42.1875% |
| 不可回答误接受 | 0/64 |
| 无效 span | 0/128 |
| 系统 EM / F1 | 91/128 = 0.7109375 / 0.7109375 |
| 始终拒答基线 EM | 64/128 = 0.5 |
| 总拒答 / 其中可回答 | 101 / 37 |

37 条可回答拒答中，**19 条原始 candidate 本来已满足 gold EM**。这是真实的精度—覆盖取舍，不能把总 EM 上升解释为所有问答能力都增强。原始 candidate 仍是 128 条全部出 span；安全拒答来自额外排序 head 和阈值，而非骨干模型突然学会了拒答。

| Article（每篇 16 可回答 + 16 不可回答） | 接受且正确 | 可回答覆盖率 | 拒答可回答 |
|---|---:|---:|---:|
| Civil_disobedience | 0 | **0%** | 16 |
| Ctenophora | 6 | 37.5% | 10 |
| Harvard_University | 11 | 68.75% | 5 |
| Yuan_dynasty | 10 | 62.5% | 6 |

**Civil_disobedience 整篇全部拒答。** 整体门槛通过不等于每个文章分布达到同样覆盖率。按文章结果是预先固定数据上的描述性切片；本轮没有事后增加新通过门槛、替换题目或调阈值。四篇公开 SQuAD dev 文章仍不足以证明跨领域或业务部署质量。

The 27/27 accepted precision has a descriptive two-sided 95% Wilson interval of approximately **[0.875445, 1]**; 0/64 false acceptance has an upper bound of approximately **0.056624**. These binomial intervals ignore context/article dependence and are report-only. In particular, empirical 100% does not establish a population precision of at least 90%. The result supports a bounded local public-benchmark prototype under the frozen protocol, not production or model-unseen confirmation.

选择记录 SHA256 `b2999477ee82318a98266ef0674c22a526cddea1eec541c591a7bec7e354b74a` 与推理前提交 `c0e28ffdacc1c7bd5a6910020b3771195f4317f3` 的 Git 对象一致。FP32 无可行校准阈值仍明确保持失败；不能把 INT8 管线通过说成相对已通过 FP32 管线完成量化非劣验证。head 分数本身由另一核验器重建，本脚本只独立验证其有限数值、0.7 阈值决策和结果评分。

可供现场演示的三例按各类别内 `SHA256("qa-risk-demo-2026100408" + id)` 最小值选择：

| Category | ID | Candidate / emitted | Head score |
|---|---|---|---:|
| 接受且正确 | `5726400589a1e219009ac5ef` | `tentilla` / `tentilla` | 0.782387 |
| 正确拒答不可回答 | `5a669ee5f038b7001ab0c06b` | `punished` / `NO_ANSWER` | 0.093439 |
| 拒答可回答，展示代价 | `5728eef92ca10214002daab2` | `solidarity` / `NO_ANSWER` | 0.519485 |

These three examples are explicitly chosen **after outcomes for illustration**. They are not evaluation evidence or a representative random sample. The JSON includes their questions, gold spans, exact offsets and context hashes so the demo can read the same fixed input by ID.

分发检查：扫描当前 Git tracked 以及 configs/results 的 776 个 JSON，未发现已学习 head weights/intercept/scaler；无已跟踪模型二进制。两个本地 head 仅位于 `runs/qa-risk-heads-v2/{fp32,int8}.json`，已忽略且未跟踪。此检查的范围是当前仓库候选文件；最终 ZIP 仍须再做 payload 验收，不能据此声称检查了外部存储或尚未生成的发布附件。

```bash
python3 -B results/qa-risk-review-v1/evaluation-independent.py \
  --output /new/nonexistent/evaluation-audit.json
```

`evaluation-independent.json` 保存全部 128 条独立评分、按文章统计、固定门槛、baseline、演示清单、选择提交绑定及分发检查。输出文件必须尚不存在。
