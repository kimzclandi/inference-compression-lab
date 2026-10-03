# 数据来源与再现内容

本文件不授予代码或模型权重许可。数据及其再现内容保留原许可。

## SQuAD 2.0

`configs/qwen-prefix/qa-dev.jsonl`、`configs/qwen-quantization/bench-inputs.json`、`configs/qwen-confirmation/dataset/data.jsonl` 以及 results 中这些文本/答案的副本来自 SQuAD 2.0，继续遵循 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)。归属：Pranav Rajpurkar、Robin Jia、Percy Liang；文章来自 Wikipedia 贡献者。

[官方项目及许可](https://rajpurkar.github.io/SQuAD-explorer/)；[原始公开 dev 文件](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)；论文：Rajpurkar, Jia, Liang (2018), *Know What You Don't Know: Unanswerable Questions for SQuAD*。

历史 74 题来自个人 Domain QA Lab 已有 Computational_complexity_theory 子集；本轮新样本从同一公开 dev 中按预先固定规则抽取 Rhine、Victoria_(Australia)、Jacksonville,_Florida、Sky_(United_Kingdom) 共 128 题。保留问题 ID、原文与答案，增加 family/source/split 字段，并确定性筛选、排除已知旧数据。原始文件哈希、变更规则、排除项与选样脚本均保留在 `configs/qwen-confirmation/`。原文是 SQuAD 收录的历史版本，不是今天 Wikipedia 网页的快照。

模型生成内容若再现源文段，仍保留上述来源与许可。旧 attribution 中的 SmallModelQAFinetuningAndQuantization 是更早历史入口；此次直接复用的本地源仓库是 `kimzclandi/domain-qa-lab`，见 `results/qwen-quantization-history/origin.json`。不改写历史证据以统一名称。

这些都是公开 dev 的本地拆分，不是官方隐藏测试；新样本仅相对已盘点的个人项目数据未使用，不保证未进入模型预训练语料，也不保证不存在语义近重复。

## STS-B 与其他工作负载

MiniLM 实验使用 sentence-transformers/stsb 固定 revision `ab7a5ac0e35aa22088bdcf23e7fd99b220e53308`；来源和文件哈希在 MiniLM provenance 中。原始 parquet 与 STS-B 原句不随发布包分发，results 保存数值预测、标签和统计。STSbenchmark 原始分数保留 CC BY-SA 4.0；本项目不对其重新授予代码许可。归属：Eneko Agirre、Daniel Cer、Mona Diab、Iñigo Lopez-Gazpio、Lucia Specia，SemEval-2017 Task 1。Sentence Transformers 将分数归一化到 0–1；记录保留所用数据 revision、样本索引、筛选与变更规则。

2026-10-04 核验：[Microsoft 官方再分发说明](https://microsoft.github.io/nlp-recipes/DatasetReferences.html) 明确区分分数的 CC BY-SA 4.0 与各来源文本自身条款；[Sentence Transformers 的数据说明](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/sts/README.md) 说明分数归一化。原始 STS wiki 本轮无法访问，因此没有把第三方模型卡的许可标签当作整个文本集的统一授权。需要原句的复跑由使用者按上游条款自行取得数据。

前缀缓存固定长度工程输入为合成负载，不是线上业务流量。模型权重、学校或公司材料均不包含在发布包中。

QA 整改追加的本地 calibration/confirmation 拆分及变更说明见 [configs/qa-remediation/ATTRIBUTION.md](configs/qa-remediation/ATTRIBUTION.md)，继续遵守 SQuAD/Wikipedia 的原始许可；它们不是官方隐藏测试。
