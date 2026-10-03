# Source and scope

The study reuses the unchanged 74-question SQuAD 2.0 local development subset and prompt already attributed in ../qwen-prefix/ATTRIBUTION.md. `bench-inputs.json` copies the first eight frozen TRAIN pilot prompts from the author's Domain QA Lab (`data/teacher-pilot-v1/request.json`); these are also SQuAD-derived and retain CC BY-SA 4.0. See https://rajpurkar.github.io/SQuAD-explorer/ and https://creativecommons.org/licenses/by-sa/4.0/legalcode.en . No held-out test was read or inferred.

`lab/qa_metrics.py` preserves the previous scoring functions verbatim; only unrelated CLI imports/entry point were removed. Original source and prediction hashes are in `results/qwen-quantization-history/origin.json`. No score adjustment is allowed.

MLX 0.29.3 and MLX-LM 0.26.3 provide affine grouped quantization, Metal kernels, Qwen2 layers, and KVCache. This project implements the intervention/export checks, diagnostic selection, matched benchmarks, and verification; it is not a new quantizer or a new inference kernel. The official API reference is https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html ; executable behavior is checked against the installed versions.

No school/company code, weights, robot commands, or hardware changes are included. New experiments belong to this personal project.
