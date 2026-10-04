# Portfolio evidence and personal practice / 招聘证据与亲自实践

当前判断：**适合作为 AI Infra/推理优化岗位的补充项目**。它有真实量化图、独立进程性能、数值/拒答机制和冻结实验的证据，但尚不能独撑加速算法研究岗位的全部要求。若本人能独立解释、修改并复跑，它也可以作为面试中的主要工程案例；目前没有这份掌握证据。没有当前简历 DOCX/PDF，本文件只是候选表达，不修改任何历史简历，不替用户确认个人职责。

This is a useful supporting portfolio project for inference infrastructure, especially experimental discipline, performance boundaries and reproducibility. It is not evidence of original quantization kernels, broad algorithmic research success, production service ownership, training acceleration or independent author mastery. All work belongs to the personal project, separate from ByteDance evaluation, NUS research and historical NUSRI/Jetson work.

## The two strongest pieces of evidence / 最有价值的两项证据

1. **受控量化与运行成本判断。** 上游任务模型导出 FP32、固定 QA projection 浮点、ORT 动态 INT8；验证原始模型/export logits、真实整数图和全部身份。固定 8 输入、6 独立轮换进程共 240 次测量：72.052 ms → 54.188 ms，ONNX 文件 496,445,889 → 242,243,831 bytes。强调这是基础流水线；完整排序门控 INT8 另测 69.600 ms。不能附带“无精度损失”，因为相应质量非劣门槛未通过。
2. **从失败到有边界的质量策略。** 原始高分会错答，先固定校准/评估发现失败；随后只追加一次固定监督排序实验，严格训练/校准/新评价角色，保留所有负结果。新 128 题接受 27 题全部 EM 正确，覆盖率 27/64；独立审计发现一整篇文章全拒答与 19 个正确 raw 答案被压制，因此只开放本地受限原型。不是原创 logistic regression，也不是部署质量或理论风险保证。

代码和原始材料：`experiments/prepare_qa_specialist_assets.py`、`lab/extractive_qa.py`、`lab/qa_risk_calibration.py`、`experiments/serve_qa_specialist.py`；`results/qa-specialist-assets-v1/`、`results/qa-specialist-performance-v1/`、`results/qa-risk-v2/`、`results/qa-risk-performance-v1/`。框架负责模型、导出器、量化算子与 CPU kernels；本项目贡献合同、固定对照、诊断、失败处理、监督排序集成、独立重算与发布核验。文档页数、测试数量、ZIP 文件数不应写成业务或算法成果。

## Candidate wording and fact checks / 候选改写与事实表

| 原表述 | 候选改写 | 代码与原始证据 | 修改理由 | 本人需核实 |
|---|---|---|---|---|
| 无当前简历原文 | 个人项目：构建 ONNX Runtime CPU 动态 INT8 与可拒答 QA 实验链路，固定模型/tokenizer、输入和线程，独立进程轮换复测基础流水线 72.05→54.19 ms；保存逐 token logits、配对质量对照和全部负结果。 | prepare_qa_specialist_assets；qa-specialist-performance-v1 的 6 进程/240 次计时；qa-specialist-v1 原始 logits | 体现真实推理测量与范围，不能把文件缩小当加速、不能附带无损质量 | 是否能解释 activation/weight 量化、projection exclusion、CPU同步和统计单位；自己具体完成什么 |
| 无当前简历原文 | 固定五特征正确性排序与拒答策略，在新选四篇公开文章的 128 题评估中接受 27 题均 EM 正确，可回答覆盖率 42.19%；建立身份/输入/质量门禁，并保留文章失效与覆盖损失。 | qa_risk_calibration；serve_qa_specialist；qa-risk-v2/evaluation-int8；evaluation-independent.json | 有明确分母、数据来源和代价，不冒充通用准确率/生产部署 | 能否独立重建 head、推导特征、解释数据角色与27个样本的不确定性 |
| 不支持的候选：恢复量化精度、QA上线、原创INT8优化、学校/公司部署 | 删除或降级为“研究诊断/局部实测/有限本地原型” | Qwen确认失败、specialist-v1非劣失败、risk无新FP32配对 | 这些事实无法支持更强结论 | 历史Jetson后端、模型、版本、本人职责仍需原始材料 |

建议简历压缩为两句，保留个人归属：

> 个人项目：构建 ONNX Runtime CPU 动态 INT8 与可拒答 QA 实验链路，完成模型身份、逐 token logits、独立进程性能和冻结质量评估；固定基础负载延迟由 72.05 ms 降至 54.19 ms（单机 CPU）。加入监督正确性排序后，在 128 题公开文章评估中接受 27 题均 EM 正确、可回答覆盖率 42.19%，完整 INT8 排序流水线约 69.60 ms；保留量化质量非劣失败及覆盖率限制。

暂不写“独立负责”，不把 AI 辅助代码提交等同于本人已掌握。字节、NUS、NUSRI 和个人项目绝不合并。

## Minimal interview demonstration / 最小现场展示

安装固定环境、准备本地资产后：

```bash
python -m experiments.verify_release_rc4
python -m experiments.reproduce_qa_prototype \
  --asset-root runs/qa-specialist-assets-v1 --output-dir runs/my-interview-reproduction
```

展示一条真实回答、一个不可回答拒答、一个本可回答却被拒答的代价案例，再展示超长问题/额外 gold 字段/缺失模型的失败关闭。所用三个问题是**已公布后挑出的演示样本**，不作为新的评估。讲清初始化核验耗时与热请求耗时的区别。

30 秒回答：项目的重点是把量化成本和任务质量分开验。INT8 基础链路在固定 CPU 负载更快，但早期质量门槛失败；我保留失败并固定一个小监督排序器，以覆盖率换取已接受答案精度。最终只得到公开数据上有限原型的证据，不能据此宣称生产 QA 或无损量化。实际本人讲述应先补齐亲自复跑和职责核实。

## Learning map / 学习主干

顺序：原始 token/logits/offset → 合法 span 与 null → 量化误差及真实运行成本 → 分数排序/阈值/覆盖率 → 数据角色与置信区间 → 门禁与可复现包。必须掌握每层输入输出和失效方式；进阶是风险校准、分组置信保证和多域迁移；本轮可暂缓 CUDA kernel、分布式服务与业务扩展，因为没有相关实测。

五个思维模型：

| 模型 | 为什么有用与何时失效 |
|---|---|
| 对照隔离原因 | 固定其它变量才能解释差异；两个独立训练 head 的差异不能归因于量化器。面试用它辨别无效“前后对比”。 |
| 分数不等于正确概率 | logits/softmax服务于训练目标；高margin仍可能选错短语。分布变化后校准也可能失效。 |
| 精度—覆盖率交换 | 更严格阈值可减少输出，但始终拒答没有实用覆盖。非单调校准曲线不能凭直觉外推。 |
| 证据链逐层成立 | 图有INT8、文件更小、请求更快、质量通过分别需要证据；测试通过不能替代任何一层。 |
| 数据角色有记忆 | 选过层/看过错误的样本已经影响方法，不能重新称未见确认；公开新文章也不保证模型未见。 |

三个实际争议：统一更大生成模型与任务专用模型之间是通用性/成本/任务匹配的取舍；宽松拒答与严格拒答之间是错误成本/有效覆盖的取舍；样本点门槛与总体风险保证之间是工程筛选/统计外推的取舍。没有适用于所有业务的单一优胜者。本项目只支持特定英文 supplied-passage 公共数据的实测边界，真实业务需要先定义可接受错误成本和新目标分布，再做新验收；不能把争议写成确定结论。

主动回忆（不附答案）：

1. context 内的精确子串为什么仍能错误回答问题？
2. span/null margin 为20，能否说答案正确率接近100%？
3. 为什么 alternative 要排除 normalized 相同的答案？
4. 为什么每个窗口自己的 null 分数会改变最终选择？
5. 0字符 token 为什么不能当端点却要计入长度和softmax？
6. 为什么此次27/27不证明总体precision≥90%？
7. 为什么同一份旧evaluation重新划分后只能叫development？
8. 为什么1.33×不能套到带排序器的69.60ms流水线？
9. 为什么INT8 head通过而FP32失败，不证明量化提高模型准确率？
10. 若一个领域全拒答而总体门槛通过，你怎样判断能否上线？

**必须由本人完成的任务，Codex 不代做：** 在一个新分支新增 `experiments/my_span_gap_ablation.py`，把第三特征从“全部窗口的最佳不同 normalized answer”改为“仅选中窗口内最佳不同答案”。亲自写一个两窗反例，解释两个gap为何不同；仅用已公开的训练/校准材料重训并比较固定门槛下 precision/coverage，保存新目录，不改默认策略、历史结果或阈值网格。若再跑已公开evaluation，只能称复现/消融，不得称新确认。交付代码 diff、命令、原始记录和一段失败分析后，再判断是否真正掌握。
