# Fixed specialist QA assessment / 固定专用 QA 模型评估

The calibration and evaluation files are selected, reformatted excerpts of **SQuAD 2.0 public development data**, retaining **CC BY-SA 4.0**. Attribution: Pranav Rajpurkar, Robin Jia, Percy Liang; passages by Wikipedia contributors. This notice does not license the root code or model weights.

- [SQuAD dataset, authors and upstream license](https://rajpurkar.github.io/SQuAD-explorer/)
- [Original public development JSON](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [CC BY-SA 4.0 legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
- [Specialist model card and upstream benchmark results](https://huggingface.co/deepset/roberta-base-squad2)
- Source SHA256: `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`.

Changes: deterministic filtering and local splits, preservation of original ID/context/question/answer strings, deduplication of identical answer strings, and addition of `source_title`, normalized exact `family_id`, and `split`. No model predictions or scores enter selection. The selector imports only standard-library code and prior dataset/evidence helpers.

先累计排除 `qa-remediation` 既有排除清单及该轮所有 384 道 calibration/confirmation 数据（包括未运行但已公布的 confirmation），再用固定 seed `2026100405` 选择 calibration 4 篇文章共 128 题和 evaluation 8 篇文章共 256 题。每篇文章可回答与不可回答各 16 题，每个类别内 context 唯一，类别之间允许共享 context。两组的标题、ID、规范化精确 context/question 全部互斥。先选 calibration，再从剩余材料选 evaluation；不足时失败，不更换 seed 或放宽配额。

**这是 public benchmark 上的 fixed local assessment，不是模型未见确认实验。** `deepset/roberta-base-squad2` 使用 SQuAD 2.0 监督训练，模型卡也报告 public dev 结果；本轮项目内部排除不能消除上游 benchmark 暴露、可能的模型选择偏差、预训练污染或语义近重复。不能把这 256 道题称为官方隐藏测试或模型从未见过的数据。新选文章只相对本项目已盘点材料互斥，不等于业务分布代表性。

The article filter is conservative: any previously used passage excludes its entire article. Unique-context quotas and a 4000-character context limit bias selection toward articles that satisfy those constraints. Statistics must account for context-family dependence. Calibration may select only the predeclared threshold. Evaluation is one frozen-policy public-benchmark assessment. It cannot alone establish deployment quality or confirm the historical block-10 quantization hypothesis. Subsequent runs on these published samples are reproductions.

```bash
python3 -B -m experiments.prepare_qa_specialist \
  --raw /path/to/squad-dev-v2.0.json \
  --output-dir runs/reselected-qa-specialist
python3 -B -m unittest discover -s tests -p 'test_qa_specialist_selection.py' -v
```

The output directory must not exist. `selection.json`, `exclusions.json`, and the dataset manifests fix selection and content identities. `results/qa-specialist-selection-v1/verify_source.py` independently checks the selected fields and every answer offset against the raw upstream JSON without importing the selector.
