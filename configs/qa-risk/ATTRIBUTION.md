# One supervised risk-head allocation / 一次有监督正确性排序实验的数据分配

These selected and reformatted excerpts of SQuAD 2.0 public development data retain **CC BY-SA 4.0**. Attribution: Pranav Rajpurkar, Robin Jia, Percy Liang; passages by Wikipedia contributors. No root code or model license is granted by this notice.

- [SQuAD source, authors and license](https://rajpurkar.github.io/SQuAD-explorer/)
- [Original public development JSON](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [CC BY-SA 4.0 legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
- Raw source SHA256: `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`.

上一轮 specialist 的 calibration 128 题和**已失败的 evaluation 256 题全部降为 development**，不是再次使用原 evaluation 证明独立确认。对其 12 篇文章按 `SHA256(str(2026100408)+title)` 排名，前 8 篇、256 题作 train，后 4 篇、128 题作 calibration；逐行只改 `split`，原 ID/context/question/答案和 answerability 保持一致。旧数据与旧失败结果不改动。

为这唯一一次后续假设固定的新 evaluation 采用原累计排除规则，另外排除上述 384 题的全部标题、ID、规范化精确 context/question。剩余合格文章恰好四篇：Ctenophora、Civil_disobedience、Yuan_dynasty、Harvard_University。每篇可回答/不可回答各 16 题，类别内 context 唯一，共 128 题。没有更换 seed、缩样本或根据预测挑题。不同 split 之间四轴互斥；类别之间可共享 context，不能把所有题视为独立观测。

Changes relative to upstream: deterministic selection, answer-string deduplication, and addition of `source_title`, normalized exact `family_id`, and local `split`. Existing specialist rows change only their role field; new evaluation rows preserve raw question/context/answer text. The selector reads no model output. This allocation is for one fixed supervised correctness-ranking hypothesis, motivated by an already observed failure; it does not erase that adaptive project history.

This remains a **fixed local public-benchmark assessment**, not an official hidden test or model-unseen confirmation. The specialist was supervised on SQuAD2 and upstream already reported dev results. Exact disjointness cannot remove benchmark exposure, pretraining contamination, semantic near-duplicates or distribution shift. Only four articles remain, with 4000-character limits and unique-context quotas; business deployment quality and generalization cannot be inferred from passing them. Future use of the published evaluation is reproduction, not newly unseen evaluation. This round predeclares no further feature/model search after this single follow-up fails.

```bash
python3 -B -m experiments.prepare_qa_risk_data \
  --raw /path/to/squad-dev-v2.0.json \
  --output-dir runs/reselected-qa-risk
python3 -B -m unittest discover -s tests -p 'test_qa_risk_selection.py' -v
python3 -B results/qa-risk-selection-v1/verify_source.py --raw /path/to/squad-dev-v2.0.json
```

The output directory must not already exist. The independent source check validates every answer offset against raw SQuAD and verifies that all 384 old rows are reused exactly once with only `split` changed.
