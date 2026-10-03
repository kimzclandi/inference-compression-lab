# 已公布 QA 失败的独立诊断 / Independent diagnosis of published QA failures

范围：旧 74 题与公布的 128 题，共 202 个题目、五变体的 1,010 条输出。仅离线读取历史输出；没有运行模型、修改历史输出或重标参考答案。`recompute.py` 只用 Python 标准库，独立实现已公开的评分定义并逐条核对历史 EM、F1、拒答及原文 span 合规性；1,010 条全部一致。它同时核对五变体每题 prompt SHA256、输入 token SHA256 和 token 数相同。

Scope: 202 already-published cases and 1,010 outputs across five variants. This is an offline independent recomputation, not a new confirmation or a model run. No historical data or labels were changed.

## 先区分三个失败 / Separate three failures

| 问题 | 当前证据 | 可采取动作 | 不能得出的结论 |
|---|---|---|---|
| Q4 的新增退化 | FP16 52/202、Q4 42/202；F1 0.316426 对 0.267598；原文 span 合规 181/202 对 122/202 | 用 Q8 作为精度保护候选，并实测其质量、延迟及资源成本 | 量化必然无效，或 Q8 必然更快 |
| block10 回退确认不通过 | block10 47/202，且已公布128题仍未满足原定确认门槛 | 保留负结果；不将该回退作为服务默认策略 | 继续在128题选层后可以把同一实验称为确认成功 |
| 基础 QA / 拒答能力不足 | 五变体均未在105个不可回答题上输出 `NO_ANSWER`；FP16在97可回答题仅52题EM正确 | 新的提示/模型候选与拒答机制必须单独评估 | 消除量化退化就等于达到部署质量 |

在已公布202题中，Q8恰好保持52题EM正确，F1为0.317251；相对FP16仅7条原始输出变化，1题EM变好、1题变差。该结果支持将Q8作为本轮有限比较的精度保护候选，不支持声称它已通过新的未见数据确认或任何部署验收。

Q8 preserves the observed aggregate FP16 quality on this published set, but the base QA failure remains. Precision protection and QA readiness require separate acceptance decisions.

## 主要假设与证据 / Hypotheses and evidence

1. **拒答指令遵循/问题语义判断是首要瓶颈。** 已有 system prompt 明确要求仅在有答案时抽取，否则输出 `NO_ANSWER`，但五变体零拒答。示例 `5a1c60f0b4fb5d0018714611` 询问“美国第三大海港”，原文只说“佛罗里达第三大”，FP16仍答 `Port of Jacksonville`。这不是简单输出格式问题；一个答案确实出现在原文中，并不说明它回答了问题。此处是从输出推得的可验证机制假设，尚不能区分模型容量、提示模板与任务训练的相对因果贡献。
2. **输出预算不足不是主要解释。** 五变体共同输入范围107–443 token，明显低于2048限制；输入未截断，且FP16 202条只有1条达到48-token输出上限。延长预算不能解释或修复大量简短而错误的抽取和拒答失败。
3. **量化造成的提示遵循退化与基础语义失败并存。** Q4的原文span合规率显著低于FP16/Q8，部分输出变成整句解释。block10恢复了一部分行为，但不足以恢复原始质量。因为选层使用已知错误，这只能形成探索假设。
4. **格式门只能作为必要防护。** 离线固定反事实：遇到非原文span一律拒答，Q8可修复15/105个不可回答题，仍有90个不可回答题给出原文中存在但不受问题支持的片段。该门不能单独保证事实性，也不能代替答案正确性评估。对Q4还会损失4个原来规范化EM正确、但严格原文span不合规的答案。所有反事实统计均保留在`recomputed.json`，不是新的模型实测。
5. **参考答案可能存在个别歧义，不能据此重写评分。** 例如`570967c4ed30961900e840bc`原文先写1998年发射使用Astra 2A、后写新增Eurobird 1；问题表述与参考答案可能引发解释分歧。本轮仍按既有参考评分，不将人工审查结果回填成更好的EM。若未来审查数据，需单独版本和双人审查，不得混入本轮确认。

## 有限修复候选 / Bounded remediation candidates

在新评估输出生成前固定候选、选择规则和停止规则。已有202题现在只是开发/回归集，不能恢复成“未见确认集”。建议至多比较以下候选，避免反复改提示直到得到好结果：

- **A：0.5B Q8 + 固定抽取/拒答提示。** 使用少量自行构造、与评估语料无关的合成示例，明确地理范围、时间和实体替换造成无答案的情况；输出仍仅允许原文span或`NO_ANSWER`。不把已知评估题的答案或近似示例放入提示。
- **B：已有本地1.5B或3B + 同一个固定提示。** 此比较检验更强模型能否解决当前容量/语义瓶颈，并记录代价；它不能伪装成0.5B算法加速。可先在固定开发子集比较有限候选，确定一个后只进行一次最终评估。
- **统一防护：** 对空输出、超预算、非span输出、模型身份不符和运行失败返回有类型的拒绝/错误状态；不得把运行错误默默改写成正确拒答。若需要把模型生成的解释变成span，处理规则必须固定、不能访问gold，并以真实端到端输出评分。

A larger local model or a fixed task prompt is a legitimate new system candidate, not a repair that retroactively turns the failed block-restoration confirmation into a success. A substring guard is a format contract, not an entailment verifier.

## 固定验收建议 / Acceptance recommendations

这些是本轮设计建议，尚不是已满足的门槛。最终采用的具体数值需在最终评估前写入协议。

- **质量：** 同时报告总体EM/F1、可回答EM/F1、不可回答召回率、错误拒答率、回答覆盖率，以及返回答案中的EM精度。只提高总体EM可能只是更多拒答；不能只用超过“始终拒答”基线作为部署条件。
- **量化保真：** 新选择的系统分别以其FP16与Q8对照，预设允许的EM/F1退化上限；使用相同输入、模板、生成和评分。不拿新模型Q8与旧模型FP16差异冒充纯量化效果。
- **性能/资源：** 将预处理、首token、decode、完整请求计时及同步范围固定；若加入第二次验证模型调用，必须把该调用算入端到端成本。RSS、MLX峰值、文件字节和KV逻辑字节分别报告。
- **泛化：** 最终集合按context family划分，避免共享原文；预先确定公开来源、文章和排除规则。已公开数据可能进入预训练，仍不得称官方隐藏测试。最终集合结果不达标就停止并保留失败。
- **服务范围：** 即使有限QA基准通过，也只支持“固定上下文抽取/拒答原型在该评估上达到预设门槛”。并发、SLO、真实业务样本、越权/提示注入、回滚和长期监控尚需独立证据，不能直接称一般生产部署质量。

## 可复跑入口 / Reproduction

```sh
python3 results/qa-remediation-review-v1/diagnosis/recompute.py
```

输出：`recomputed.json`含输入文件SHA256、逐数据集/变体汇总、配对增损与格式拒答反事实；`all-records.jsonl`保留1,010条独立逐题评分、原始答案及问题。此步骤不依赖MLX、模型文件、网络或GPU。
