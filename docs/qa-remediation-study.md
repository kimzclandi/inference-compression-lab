# QA 质量整改：修复已落地，模型质量仍未过门槛

## 结论 / Decision

**工程缺陷已修复；QA 部署质量尚未解决；原量化回退的确认失败不能被改写。** 本轮实际执行了四个完整开发对照和两个新校准对照，保存失败运行、全部预测与原始 logits，并由不导入项目评分器的脚本独立重算。QA 入口默认关闭，返回 `unavailable_quality`，不是伪装成正常 `NO_ANSWER`。这能阻止不合格配置作为可用系统交付，不能替代模型能力提升。

Engineering defects were repaired and exercised, but the QA quality goal remains unmet. A bounded model/prompt study followed by frozen FP16/Q8 calibration found no feasible operating threshold. The local answer entry point remains disabled. The historical block-restoration result is unchanged; this remediation does not validate that algorithm.

|需分别判断的事项|本轮判断|
|---|---|
|可复现研究材料|原始证据、失败记录、固定协议和离线重算可交付；发布仍受根许可证/可见性决定约束|
|block10 回退优化|历史确认未通过；退出部署候选。不能说已证明彻底无效，也不能改称成功|
|新 Q8 压缩|实际参数张量存储减少46.87%；未获得确认集质量非劣效结论，无本轮加速结论|
|QA 可部署|不通过本地研究门槛；默认入口不可用。业务分布、安全、并发、延迟SLO未验收|
|目标JD全面匹配|不覆盖训练加速、蒸馏、稀疏、token压缩、CUDA kernel、TensorRT或多设备部署|
|用户本人掌握|没有新增证明；Codex辅助实现和自动复跑不等于本人独立实现|

## 问题、假设、固定验收和实际动作

1. **问题：旧0.5B所有不可回答题都作答。** 对历史开发74题和确认128题合并只读分析，五变体均在105/105不可回答题上失败。FP16同样失败、输入未截断，说明不能把问题仅归因于Q4。Q8仅对少数输出产生变化；把非原文答案一律抑制也不能挡住大量“合法但错误”的span。证据：[独立诊断](../results/qa-remediation-review-v1/diagnosis/diagnosis.md)。
2. **假设：有限容量/提示对照可能改善拒答，输出合同可阻止非抽取答案。** 在已公开128题上固定少量候选。四个完整运行后停止模型/提示搜索：0.5B grounded、1.5B legacy、1.5B grounded、一次已存在3B grounded容量对照。开发选择及原因见[decision.json](../results/qa-remediation-development-v1/decision.json)。3B与1.5B grounded原始EM均85/128，且3B研究许可不适合作为默认商业部署候选。选择1.5B进行下一阶段，不把开发结果当独立确认。
3. **固定验收：** 正式推理前在提交 `cbdaad7b83b00db589524a69b8ccf2f1be78ce91` 冻结模型、数据、提示代码哈希、精度、9个阈值、五个门槛、失败停止规则和条件确认方案。此前 `19a5a92` 固定初始开发预算；`5217c08` 在看到较小候选失败后明确追加一次容量对照，该追加属于开发，不称原始预注册。Git时间线是本地工作记录，不是第三方预注册平台证明。
4. **修改：** 精确原文span及字符偏移、明确拒答/非法/运行异常分类、一次独立调用但同一模型的答案自审、有限标量与输入长度检查、精确模型/数据注册、运行失败保留、质量门禁和证据绑定CLI。没有微调、训练adapter或新量化kernel。
5. **实际运行：** 同一M4 Max 48GiB、复用现有Python3.12.11/MLX0.29.3/MLX-LM0.26.3环境，离线从本地1.5B原始BF16权重加载并转FP16，另一个独立进程执行MLX affine group64 Q8。各128个新校准问题，完整256条生成。`run.json`保存版本、文件哈希、prompt哈希、token、logits、计时、模型布局和逐题评分。没有权重下载、付费算力或新环境安装。
6. **分析与保存：** 两者所有阈值失败；严格执行停止规则，不运行预留256题确认，不补抽样、不换prompt、不调门槛。完整证据见[校准目录](../results/qa-remediation-v1)，独立实现重算见[审计报告](../results/qa-remediation-review-v1/independent/findings.md)。

## 数据与协议

新 calibration128题（64可回答/64不可回答、4篇文章、95个context家族），预留 confirmation256题（128/128、8篇文章、190个context家族）。排除盘点过的103个标题、706个ID及规范化精确context/question；两组之间也互斥。原文重选和答案offset复核完成，详情见[数据归属与选择偏差](../configs/qa-remediation/ATTRIBUTION.md)。

这是SQuAD公开dev的本地拆分，不是官方隐藏测试；不证明无预训练污染、语义近重复或其他未盘点材料曝光。文章数少、选择规则及英文抽取任务限制外推。预留确认数据本轮没有推理，但已随研究材料披露，后续不能仅凭“本轮未消费”把它称为秘密/从未曝光测试。

固定质量门槛为：返回答案EM精度≥90%、可回答题作答覆盖≥40%、正确可回答题覆盖≥35%、不可回答题错误作答率≤10%、原始非法输出率≤5%。阈值网格为0、0.5、0.8、0.9、0.95、0.98、0.99、0.995、0.999。门槛是本地研究的工程要求，不是用户业务验收或行业统一标准。Wilson区间仅用于描述，未处理context/文章依赖，不能称统计风险保证。

每种精度只能选择最小可行校准阈值。**任何一种没有可行阈值就停止确认。** 不使用最大EM作为备用。未触发的确认方案原本要求：固定两种阈值后提交，再对Q8−FP16的系统EM/F1进行5000次按文章分层context家族bootstrap，两项95%区间下界均≥−2个百分点，参数张量存储比≤0.60，且两种精度都通过质量门槛。本轮没有执行这一确认比较；统计分支仅有合成合同验证。

## 真实结果及其含义

|实验/数据角色|原始EM|原始不可回答题正确拒答|说明|
|---|---:|---:|---|
|历史0.5B FP16 / 已公布128题|33/128|0/64|旧确认，现作为开发参考|
|0.5B FP16 grounded / 同128题|49/128|5/64|本轮开发|
|1.5B FP16 legacy / 同128题|73/128|22/64|本轮开发容量对照|
|1.5B FP16 grounded / 同128题|85/128|38/64|本轮开发提示对照，选定候选|
|3B FP16 grounded / 同128题|85/128|35/64|一次追加容量检查，仍失败|
|1.5B FP16 / 新校准128题|74/128|35/64|F1=0.6107762897；无可行阈值|
|1.5B Q8 / 新校准128题|76/128|37/64|F1=0.6264012897；无可行阈值|

开发集85/128相对历史33/128的变化包含模型容量和提示协议变化，不能归因于量化回退，更不能把不同数据上的85/128与74/128视为同条件回归。

|新校准阈值|FP16精度 / 可回答覆盖|Q8精度 / 可回答覆盖|判断|
|---|---|---|---|
|0.95|75.00% / 56.25%|73.81% / 53.13%|精度不足|
|0.995|92.31% / 20.31%|92.31% / 20.31%|覆盖不足；正确覆盖仅18.75%|
|0.999|100% / 4.69%|100% / 3.13%|只剩3个/2个答案，不能宣称可用|

两种精度的原始非法输出均8/128=6.25%，超过门槛。非法输出被抑制仍记为非法，不以`NO_ANSWER`替换后奖励分数。可回答问题被过度拒答会损失覆盖，始终拒答不能通过门槛。

实测参数张量存储：FP16 **3,087,428,608 bytes**，Q8 **1,640,332,288 bytes**，后者为前者 **53.1294%**。包含scale/bias等参数张量，不包括KV、激活、allocator缓存或进程其他内存；不是权重文件大小或RSS。KV及norm保持浮点，Q8不是FP8。没有将本轮QA计时升级为性能收益：自审增加第二次prefill，`total_pipeline_seconds`包含它；生成的TTFT/decode时间不等于整个QA管线延迟。

## 新修复和遗留项

|缺陷|修复/实际证据|仍然不能推出|
|---|---|---|
|合法span被误当正确答案|分离输出合法性、精度、覆盖、不可回答错误作答；重算全部原始行|substring保证语义正确|
|同模型高分被当概率|保留Yes/No logits、conditional score和binary mass；独立重算|概率校准、独立judge或无相关错误|
|空/非法答案抑制后虚增拒答分|非法保持独立状态，评分不奖励；合同测试|生成模型已学会拒答|
|只看EM忽略几乎全拒答|五项联合门槛，失败不选阈值|业务可用性已验收|
|运行中反复独占创建run.json崩溃|追加predictions.jsonl，终态只写一次；原失败目录和旁置postmortem保留|首次运行没有失败|
|预检/模型/数值异常无清晰终态|从预检开始捕获失败，保留current ID、已完成输出；实际故障注入|所有Metal故障、OOM或进程强杀已恢复|
|模型/数据配置混用|locked allowed_runs精确匹配，拒绝预量化源伪装FP16；完整文件hash|文件hash本身是外部可信签名|
|未达标配置仍能作为可用入口|disabled policy在模型导入前返回unavailable_quality；启用还需完整校准与确认、代码/模型/证据绑定|当前有可用QA服务，或本地门禁是防恶意维护者的安全边界|
|历史材料可能被新工作覆盖|只追加结果目录，旧result逐字节保护；发布归档来自固定commit|main已经合并、仓库公开或许可证已选|

目前真正未解决的是任务能力与可靠覆盖。最小下一阶段应是**另立协议**评估适配抽取+answerability的任务模型或监督训练，并用目标业务语料和事先约定的风险/覆盖要求验收。不能继续在这批已公布问题上修改提示直到通过，再宣称独立成功。本轮没有执行新模型下载/训练，也没有业务输入或部署验收要求，因此不承诺已解决这部分。

## 复跑与现场演示

无模型、无网络、无需作者旧目录，在解压源码根目录执行：

```bash
python3 -B -m experiments.verify_release
python3 -B -m experiments.verify_qa_remediation --root results/qa-remediation-v1
python3 -B -m experiments.qa_remediation_demo
python3 -B -m experiments.serve_grounded_qa --policy configs/qa-remediation/policy.json
```

最后一条预期exit2和`unavailable_quality`，不是命令安装失败。`verify_release`与CI通过表示**失败证据可复核、门禁行为正确**，输出中的`qa_remediation_quality_passed`仍为false。

真实校准复跑仅需Apple Silicon/Metal、仓库固定MLX环境和下述原始本地snapshot；不依赖作者其他仓库。该数据已公布，复跑不是新的确认。新目录不能存在：

```bash
python -m experiments.qa_remediation \
  --spec configs/qa-remediation/study.json \
  --data configs/qa-remediation/dataset/calibration/data.jsonl \
  --model /path/to/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --label calibration-fp16 --mode grounded --output-dir runs/qa-replay/calibration-fp16
python -m experiments.qa_remediation \
  --spec configs/qa-remediation/study.json \
  --data configs/qa-remediation/dataset/calibration/data.jsonl \
  --model /path/to/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --label calibration-q8 --bits 8 --mode grounded --output-dir runs/qa-replay/calibration-q8
python -m experiments.verify_qa_remediation --root runs/qa-replay --write-selection
python -m experiments.verify_qa_remediation --root runs/qa-replay
```

源目录需要与协议九个文件及内容完全匹配；缺LICENSE/README/tokenizer文件也会失败，应补齐原始指定revision文件，不能改协议hash以掩盖身份问题。数值跨版本/设备不一致时保留差异，不挑选通过的一次。

面试最小案例：演示 `qa_remediation_demo`，解释一个原文中的合法span为何仍因关系方向倒置而错误，接着展示完整阈值曲线中精度和覆盖的冲突，再运行关闭入口。讲清楚“机制诊断→固定验收→真实运行→停止结论”，不是展示一个挑出的正确问答。

## 招聘表述与个人验收

最有价值的仍是两类证据：已有受控量化/回退诊断及公平性能/缓存生命周期实验；本轮把QA失败分解为任务、格式、拒答、容量和精度覆盖的实验判断。框架负责模型、量化器、Metal/ORT算子；本项目贡献合同、固定对照、校准停止规则、身份/证据验证。文档/测试数量、始终关闭的入口本身不作为算法业绩。

|原表述|候选改写|代码/原始证据|理由|本人需核实|
|---|---|---|---|---|
|无当前简历原文；不采用“量化回退恢复精度并完成部署”|个人推理压缩实验：建立MLX Q4/Q8数值诊断与等成本FP16模块回退对照，固定样本确认未通过质量门槛并保留负结果；进一步实现抽取QA输出合同、拒答校准与失败关闭机制，对1.5B FP16/Q8各128题校准并独立重算，识别精度—覆盖率不可兼得的失败配置。|`experiments/qwen_quantization.py`、`lab/selective_qa.py`、`lab/qa_gate.py`；`results/qwen-confirmation-v1/`与`results/qa-remediation-v1/`|支持实验设计与工程判断，不支持成功算法或生产部署|个人完成的部分、能否独立复跑、时间与项目归属；代码由Codex辅助不代表掌握|

English candidate: Built controlled MLX low-bit diagnostics and equal-cost FP16 block-restoration experiments; retained a failed confirmation instead of claiming restored quality. Added extractive QA contracts, abstention calibration, and fail-closed evidence checks, auditing 128-question FP16/Q8 calibration runs to expose the precision–coverage limitation. Use only after personally validating the claimed contribution.

当前更适合作为**补充项目**，或经本人讲解/复跑验证后作为以实验与Infra工程为主的核心项目；不宜作为已成功加速算法或部署QA的核心成果。没有当前简历DOCX/PDF，本报告只是候选，未修改历史简历。学校、NUSRI、字节及个人项目保持分开归属。

本人必须亲自完成、此次没有代做的任务：从 `predictions.jsonl` 和 `data.jsonl` 实现按文章报告“答案精度、正确覆盖、错误作答率”的小工具，拒绝重复/缺失ID，并写一个关系方向反转的反例；在已公开校准集复跑该分析，对比全局结论。提交本人修改diff、命令和输出后，再解释为什么这不是新的独立确认。此前学习单中的个人任务亦未被自动标记掌握。
