# New local SQuAD cohorts / 新本地数据拆分

The calibration and confirmation JSONL files are selected and reformatted excerpts of SQuAD 2.0 public development data, retaining **CC BY-SA 4.0**. Attribution: Pranav Rajpurkar, Robin Jia, Percy Liang; passages by Wikipedia contributors. This data notice does not grant any root code or model-weight license.

- [Dataset, authors and upstream license](https://rajpurkar.github.io/SQuAD-explorer/)
- [Original public development file](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [CC BY-SA 4.0 legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
- Source SHA256: `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`.

Changes: deterministic filtering and local splits; preserve original ID, context, question and answer text, deduplicate identical answer strings, and add `source_title`, normalized exact `family_id`, and `split`. No model output is used in selection. `selection.json`, cumulative `exclusions.json`, and `dataset/*/manifest.json` record identities and selection rules. All 74 old development plus 128 published confirmation questions are excluded alongside the pre-existing inventory; those 202 questions are development material for the remediation work.

先固定 calibration 4 篇文章、128 题，再从排除 calibration 的剩余文章固定 confirmation 8 篇文章、256 题。每篇文章可回答、不可回答各 16 题；每个类别内部 context 唯一，类别之间允许共享 context，因此统计单位不能默认是独立的每道题。两组之间标题、ID、规范化精确 context/question 全部互斥。

标题排除是保守的：旧材料只使用某文章一段，也排除整篇文章。精确 question 排除会排除不同文章中相同的通用问题；每类 16 个不同 context 和 4000 字符上限会偏向篇幅较长、结构适合这套限制的文章。严格全局排除或上游材料变化可能导致合格文章不足，脚本会报错，不降低样本数或改变 seed。选择这些限制不证明目标业务分布得到覆盖。

These are local splits of **public dev, never the official hidden test**. Their novelty is relative only to the inventoried personal project data. Model pretraining contamination, semantic near-duplicates, and exposure in other un-inventoried sources are unknown. Calibration can choose the predeclared threshold, but cannot justify changing the frozen prompt or model. Confirmation is reserved for the final frozen-policy evaluation; after its first evaluation/publication, further runs are reproductions rather than newly unseen confirmation. The published historical failure remains unchanged.

Reproduce from a repository or Git-free source archive (download the source under its upstream license):

```bash
python3 -B -m experiments.prepare_qa_remediation \
  --raw /path/to/squad-dev-v2.0.json \
  --output-dir runs/reselected-qa-remediation
python3 -B -m unittest discover -s tests -p 'test_qa_remediation_selection.py' -v
```

The output directory must not exist. This command imports no model framework, performs no model loading or inference, and writes a source/selector/spec/exclusions identity record in `command.json`.
