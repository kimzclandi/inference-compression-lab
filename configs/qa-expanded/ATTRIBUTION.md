# SQuAD2 expanded correctness-ranking data

Source: [SQuAD2 official project](https://rajpurkar.github.io/SQuAD-explorer/), by Pranav Rajpurkar, Robin Jia and Percy Liang; passages by Wikipedia contributors. Data and reproduced passages retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en). The code license does not replace data or upstream model licenses.

Changes: deterministic title/question selection; retain IDs, passages, gold answers and answerability; add split/title/context-family fields; fixed class balance and per-context caps; exclude recorded historical exact IDs/normalized contexts/questions. Full source hashes and URLs are in `dataset/manifest.json`.

The 2,048 new training rows are from official train, on which the upstream model was trained. The 192 calibration and 192 evaluation rows are from public dev. This is not an official hidden test or proof of model-unseen data. Dev titles are disjoint from prior correctness-head cohorts, but some appeared in older Qwen studies; exact historical contexts/questions are excluded, semantic overlap is not ruled out. Train, calibration and evaluation titles are mutually disjoint in this study. Class balance is an experimental design, not an estimate of deployment prevalence.
