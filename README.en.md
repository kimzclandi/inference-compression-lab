# Inference Optimization and Performance Analysis

[简体中文](README.md) | **English overview**

[![Offline checks](https://github.com/kimzclandi/inference-compression-lab/actions/workflows/tests.yml/badge.svg?branch=codex%2Fresearch-prerelease)](https://github.com/kimzclandi/inference-compression-lab/actions/workflows/tests.yml)

**Inference Compression Lab** is a personal research project developed and iterated during the maintainer’s time in a NUS lab. It studies repeated computation and latency in fixed inference workloads: reusing KV state across shared-prefix requests, removing redundant CPU QA work, and evaluating Attention/Metal changes against native-framework controls.

Each study fixes its inputs, timing scope and controls before checking output parity, latency and memory. Candidates that miss their gates remain disabled by default, with failures retained. Separate studies do not form a single end-to-end speedup claim.

Measurements cover CPU and Apple GPU. CUDA, Ascend and production serving are unverified. Implementation, execution and documentation are AI-assisted. MLX, PyTorch and ONNX Runtime supply frameworks and general-purpose kernels; the project work is in controlled experiments, diagnostics, runtime checks and evidence verification. The work is maintained as a personal project; team research, internships and historical Jetson deployment have separate contributions and evidence.

## Project progression

- **Avoid repeated prefix computation:** implement KV reuse with capacity limits and failure handling, compare a fixed request loop with recomputation, then separately test reservation against native MLX Cache. Reservation did not pass its speed gate.
- **Remove CPU work without changing decisions:** prune redundant normalization and candidate-feature calculations, replay 896 records, and measure complete warm computation separately from initialization.
- **Test operator changes against strong controls:** compare native and compiled implementations before deciding whether to adopt an Attention/Metal candidate. The newer [Q8 QKV projection experiment](docs/qkv-projection.md) passed numerical checks but missed the speed gates. Its implementation and records are included; the candidate remains disabled by default.

## Start with a question

| Question | Implementation and evidence | What the evidence supports |
|---|---|---|
| Can unnecessary CPU work be removed without changing decisions? | [Exact risk-feature pruning](lab/qa_risk_pruning.py), [hot-path report](docs/qa-risk-pruning.md), [initialization report](docs/qa-risk-startup.md) | Identical archived features/scores/decisions; warm computation and initialization have different timing scopes |
| Does KV allocation improve over an existing framework? | [Fixed-capacity append](docs/kv-append-mps.md), [native MLX Cache control](docs/qwen-cache-reservation.md) | The benefit over repeated `cat` does not transfer automatically to native MLX Cache |
| Are Attention outputs and masks equivalent? | [CPU semantics and backend protocol](docs/attention-backend-study.md), [MPS comparison](docs/attention-mps-study.md) | Fixed CPU/MPS checks, not CUDA execution or an original Attention kernel |
| Why can a custom Metal kernel fail? | [Residual Add + RMSNorm](docs/metal-residual-rmsnorm.md), [GQA decode](docs/gqa-shared-decode.md), [numerical diagnosis](docs/gqa-shared-diagnostic.md), [exp-family intervention](docs/gqa-exp-choice.md) | Slower fusion and failed model K/V checks remain visible; candidates were not adopted |
| Does request scheduling improve the whole workload? | [Native Qwen request queue](docs/qwen-request-scheduling.md) | Some mean TTFT improvements, but the complete adoption gate failed; no continuous batching claim |
| Do compression and acceptance preserve answer quality? | [Qwen confirmation failure](docs/qwen-confirmation-study.md), [selective QA study](docs/qa-risk-study.md), [evidence map](docs/EVIDENCE_MAP.md) | Limited historical QA results and later failed confirmation, not deployment-quality assurance |

Linked technical reports retain their original Chinese, English or bilingual text. This page is an English entry guide; the [Chinese README](README.md) retains the detailed project chronology.

## Selected results and their limits

| Study | Fixed-workload result | Limitation |
|---|---|---|
| CPU exact feature pruning | Complete warm INT8 computation: 68.681 → 55.342 ms (1.241×); all 896 archived record replays unchanged | No new QA-quality improvement; not concurrent service latency |
| Full `load_service` initialization | 17.919 → 5.139 s (3.487×) in one fixed six-process study | Includes verification, head reconstruction and model loading; excludes process startup/pre-call imports; ordinary OS caching |
| Native MLX Cache reservation | Native/reserved total-time ratios 0.994×, 1.025×, 0.992×: all failed the speed gate | Allocated KV payload decreased 17.5%–43.75%; active payload did not; not peak device-memory savings |
| MPS Attention | SDPA was 1.422–1.771× faster than the faster explicit FP16 control over eight fixed shapes | Framework synchronized API timings, not pure GPU or whole-model speed; later cache-derived inputs failed the original SDPA check |
| Metal Residual Add + RMSNorm | Primary latency 22.835 → 28.277 μs, about 23.8% slower than compiled native | Numerical/output checks passed, speed gates failed; remains opt-in |
| Shared GQA decode | 24 operator checks passed, but 10 of 48 final model K/V arrays failed the fixed gate | Stopped before performance timing; diagnosis did not establish a fix or precise floating-point root cause |
| Qwen block-10 quantization fallback | EM 29/128 versus Q4 26/128; confirmation failed | The fixed cohort does not establish a general quality gain |

Timing ratios from different studies must not be multiplied. Retained failures constrain the claims; passing CI verifies code and archived evidence, not fresh GPU performance, model quality or production readiness.

## Run offline checks

Use Python 3.11 or 3.12. The current default branch is `codex/research-prerelease`. Installation needs network access or local wheels; these checks do not download models or run training. For an existing checkout, begin at `python3 -m venv` from its root.

```bash
git clone --branch codex/research-prerelease https://github.com/kimzclandi/inference-compression-lab.git
cd inference-compression-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r configs/qa-nonlinear/requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m experiments.review_qa_evidence --output-dir runs/my-review
.venv/bin/python -m experiments.verify_qa_risk_pruning
.venv/bin/python -m experiments.verify_qa_risk_startup
.venv/bin/python -m experiments.verify_qwen_cache_reservation
.venv/bin/python -m experiments.verify_gqa_shared_decode
```

Use a new output directory; preserve `results/` and frozen protocols. Optional dependency paths may be skipped locally: inspect the reported reasons. The [CI workflow](.github/workflows/tests.yml) lists the complete offline suite and separate fixed-PyTorch CPU semantics checks. See [research-prerelease checks](docs/research-prerelease.md) and [runtime reproduction](docs/release-reproduction.md) for full acceptance, model requirements and opt-in execution. Archived-log replay cannot reproduce unarchived full logits/KV arrays or GPU timing.

## Inspect the code and provenance

- [Extractive QA and window decoding](lab/extractive_qa.py), [runtime identity checks and abstention](lab/qa_specialist_runtime.py), [serving CLI](experiments/serve_qa_specialist.py).
- [System overview](docs/qa-system-overview.md) and [claim → code → raw-record map](docs/EVIDENCE_MAP.md).
- [Research release and checksums](https://github.com/kimzclandi/inference-compression-lab/releases/tag/v0.1.0-research.4): an immutable release snapshot, distinct from later default-branch work.
- [MIT code license](LICENSE), [data license](DATA_LICENSE.md), [third-party attribution](THIRD_PARTY.md). Model weights, learned head/scaler parameters and environments are not distributed.
