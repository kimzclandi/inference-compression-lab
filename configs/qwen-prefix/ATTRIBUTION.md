# QA fixture attribution

`qa-dev.jsonl` contains the unchanged 74-question local development subset of SQuAD 2.0 used in the author's Domain QA Lab. It is not a new blind test and is not the official SQuAD hidden test.

- Dataset: SQuAD 2.0; Pranav Rajpurkar, Robin Jia, Percy Liang; passages by Wikipedia contributors.
- Original file: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
- Project and license: https://rajpurkar.github.io/SQuAD-explorer/
- License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
- Previous subset: https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/tree/main/data/complexity-v1
- Modifications made upstream: selected Computational_complexity_theory, removed duplicates, added family/source/split fields and partitioned data. This work copies the dev subset unchanged. The upstream source hash and split method are retained in source-data-manifest.json.
- The fixture and reproductions of its passages/answers retain CC BY-SA 4.0; code licensing does not replace dataset licensing.
- source-qa-protocol.json records the prior prompt and its historical training/inference configuration. This new study uses MLX Q8, not that file's PyTorch FP32 device settings.

The fixed-token performance prompts are synthetic engineering workloads created for this study. They are not a task-quality benchmark or user traffic.
