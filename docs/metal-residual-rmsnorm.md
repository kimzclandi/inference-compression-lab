# Residual Add + RMSNorm Metal fusion / Metal 融合研究

已实现并在 Apple M4 Max GPU 实际执行自定义 Metal kernel；正确性检查通过，但唯一一次固定性能研究未通过加速门槛。候选仅可显式选择 `mode="metal"`，不替换默认入口。此节点作为 **v0.1.0-research.4 研究预发布**收录实现与负结果；research.3 保持原样。

A real custom Metal kernel is implemented and GPU-tested. Correctness passed, but the single fixed performance study failed its speed gates. The candidate remains explicitly opt-in in research.4; the previous research.3 remains unchanged. This is a reproducible kernel implementation and negative optimization result, not a successful acceleration claim.

## Implementation and attribution

- [Metal source](../lab/kernels/residual_rmsnorm.metal) fuses `h = x + residual` and weighted RMSNorm, returning both `h` and `y`. It uses one threadgroup per row, four consecutive elements per thread, FP32 accumulation and SIMD/threadgroup reduction. Width 896 uses 224 threads, selected by a fixed formula, with no configuration search.
- It preserves the input-type rounding after residual addition and after normalization before weight multiplication. A single final FP16 cast would change the pinned MLX semantics.
- [The Python entry](../lab/metal_residual_rmsnorm.py) rejects unsupported shapes/dtypes/modes/epsilon and CPU execution. Supported inputs share FP16 or FP32 dtype, matching nonempty shapes, last dimension 1–4096 and a one-dimensional weight. Inputs, rounded residuals and FP32 squared sums must remain finite; value scans are not hidden in the operator. No gradients are supplied. Noncontiguous inputs may require MLX copies.
- Qwen integration changes only each layer's attention-residual/post-attention norm pair. All 24 layers, attention, MLP, ordinary floating KV, whole-sequence final norm and last-position head remain identical across arms. It does not use the previous final-query pruning candidate or globally replace a production function.
- The reduction and rounding are adapted from [MLX v0.29.3 RMSNorm](https://github.com/ml-explore/mlx/blob/v0.29.3/mlx/backend/metal/kernels/rms_norm.metal), with Apple's copyright and [MIT license](../third_party/MLX-MIT.txt) retained. Project work is the residual fusion, guarded integration and evidence implementation, with AI-assisted implementation/execution. RMSNorm, SIMD reduction and fusion are established techniques; quantized matmul/attention kernels and Qwen weights remain upstream.

## Frozen design and correctness

[Protocol](../configs/metal-residual-rmsnorm/study.json) was committed as `1df4b4a` before implementation. Its SHA256 is `9e46fc69c65f631cc69919b61e820eaf191ddd4817a2aa4f58a8cbe5b42e304e`. Executed source was frozen in `d84a8a0`; raw receipts include full source snapshots, exact model/tokenizer identities, library versions and MLX binary hashes.

Three arms: uncompiled native Add + `mx.fast.rms_norm`; `mx.compile` of that native pair; `mx.compile` of the custom Metal pair. Both compiled paths cache by epsilon and use `shapeless=False`; every arm has the same public metadata checks. Compilation is excluded through fixed warmups. This is pair compilation, not compilation of the whole model.

- 300 primitive cases: FP16/FP32, 15 widths from 1 through 4096, rows 1/7, random/zero/cancellation/scaled/strided inputs. Residual and normalized outputs matched native byte identities in every case; observed maximum error and ULP difference were zero. This is finite tested coverage, not proof for every valid input/device.
- 32 model records: lengths 1/64/512/2048, prior cache offsets 0/17, initial call plus three teacher-forced continuations. All logits and effective KV digests matched native; the native integration also matched the pre-existing head-only path exactly.
- 74 already-consumed public QA examples: complete greedy sequences identical in all three arms. EM remains 18/74 and F1 0.296778; no new quality confirmation or improvement.
- All 54 timed model requests retained matching 32-token sequences, 18 requests per arm.
- Full utility suite: 449 total, 437 passed, 12 optional-dependency/GPU skips. Explicit Metal test invocation passed all 9 tests, including the three GPU tests skipped in ordinary CI. These counts overlap; do not add them as independent tests.

## One timing study — negative result

Apple M4 Max, Qwen2.5-0.5B Q8/group64, MLX 0.29.3, MLX-LM 0.26.3. Fixed three-round Latin order, nine new processes, one study, 29.9174 seconds total against a 900-second budget. No timing pre-runs, parameter search or retries.

Primitive latency is synchronized host wall time for a dependent chain of 50 pairs, divided by 50. Each next input is the previous normalized output, so all normalizations are needed. Inputs are fixed and pre-evaluated; each sample resets to the same input. Five warmup chains and 20 recorded samples per shape per process. Values are medians of the three within-process medians. They include host enqueue and final synchronization, not isolated GPU timestamp duration or measured DRAM bandwidth.

| FP16 rows × 896 | Native μs | Compiled native μs | Compiled Metal μs |
|---|---:|---:|---:|
| 1 | 18.955 | 18.116 | 19.235 |
| 64 | 10.083 | 10.536 | 11.248 |
| 512 | 13.282 | 13.867 | 19.975 |
| 2048 — primary | 23.806 | 22.835 | 28.277 |

Primary speedup was **0.841877× versus native and 0.807530× versus compiled native**, below both 1.10× requirements. The candidate was faster in only 1/3 and 0/3 rounds respectively, below the required 2/3. Against compiled native it took approximately **23.8% longer**.

Model timings use the same guarded head-only decoder in all arms, pre-tokenized inputs, fresh KV per request, two requests per length per round, and 32 synchronized generated tokens. TTFT is first-token model computation; decode covers the remaining 31 tokens. Loading, tokenization, compilation, warmups, network/queue/JSON are excluded. This does not measure stock `mlx_lm.generate`, a server or mobile deployment.

| Prompt length | Native TTFT ms | Compiled native TTFT ms | Metal TTFT ms | Native decode ms | Compiled native decode ms | Metal decode ms |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 8.775 | 8.824 | 8.819 | 99.091 | 98.826 | 99.381 |
| 512 | 32.587 | 32.592 | 32.554 | 103.032 | 102.915 | 103.854 |
| 2048 | 128.467 | 128.547 | 128.497 | 116.577 | 115.995 | 116.019 |

Neither primary TTFT nor decode achieved 1.02× against both controls. All model regression limits of 1.05 were respected, but passing a regression limit is not acceleration. The joint acceptance is **false**. No alternative candidate was selected from these results.

## Post-hoc logical-traffic diagnosis — not a new benchmark

[The deterministic cost model](../lab/kernel_cost_model.py) counts tensor bytes crossing framework primitive boundaries. The materialized native pair reads `x` and `residual`, writes `h`, then reads `h` and `weight` and writes `y`: six elements of logical traffic per output element. The fused contract must still return both `h` and `y`, so it reads `x`, `residual` and `weight` and writes both outputs: five elements. Under this deliberately optimistic model, fusion removes only **1/6 = 16.67%** of logical bytes and has a **1.20× traffic-only ceiling**. This is an upper bound under the stated assumptions, not a predicted speedup or measured Roofline.

| FP16 rows × 896 | Native logical bytes | Fused logical bytes | Traffic-only ceiling | Observed native/Metal | Observed compiled/Metal |
|---|---:|---:|---:|---:|---:|
| 1 | 10,752 | 8,960 | 1.20× | 0.985× | 0.942× |
| 64 | 688,128 | 573,440 | 1.20× | 0.896× | 0.937× |
| 512 | 5,505,024 | 4,587,520 | 1.20× | 0.665× | 0.694× |
| 2048 | 22,020,096 | 18,350,080 | 1.20× | 0.842× | 0.808× |

The low counted-operation-to-byte ratio makes the pair sensitive to bandwidth and dispatch overhead, but the model excludes caches, internal reduction passes, register/threadgroup traffic, occupancy and measured DRAM bandwidth. It therefore cannot identify the actual bottleneck. It does explain why this particular fusion has limited theoretical headroom: preserving the residual output prevents removal of its write, leaving only one logical reread to eliminate. The observed custom kernel did not realize even that idealized saving. The original failed gates and disabled default remain unchanged. [Derived JSON](../results/metal-residual-rmsnorm-v1/cost-model.json) binds this analysis to the frozen summary and analysis source.

For hardware diagnosis, [the profiler workload](../experiments/profile_metal_residual_rmsnorm.py) emits a deterministic native, compiled or Metal workload for an external Instruments trace. It deliberately records no latency and cannot alter the frozen benchmark. It supports a fixed startup delay and flushed progress records so `xctrace` can attach to a known PID; every batch is evaluated separately to avoid spending the capture window only constructing a long lazy graph. [The XML summarizer](../experiments/summarize_metal_trace_export.py) resolves Instruments' cross-row references and filters GPU intervals to the target process. Trace bundles and XML exports remain machine-local diagnostics rather than release evidence.

### Post-hoc Instruments interval trace — diagnostic only

After the Xcode license was accepted, one matched native/Metal diagnostic was captured with Xcode `xctrace 16.0 (17F113)`, the `Metal System Trace` template and PID attachment. Each arm completed 5 warmup calls and 5,000 profiled calls at the frozen primary shape `2048 × 896` FP16; both emitted checksum 1,835,008. The workload uses `chain=1`, while the frozen latency protocol uses dependent chains of 50. Instruments also perturbs scheduling. The following values therefore describe trace intervals, not replacement latency measurements:

| Process-attributed interval | Native | Custom Metal |
|---|---:|---:|
| Application command buffers | 5,008 | 5,008 |
| Application compute commands | 5,007 | 5,007 |
| GPU compute intervals | 5,007 | 5,007 |
| GPU compute median | 21.000 μs | 19.000 μs |
| GPU compute p95 | 81.875 μs | 82.834 μs |
| Sum of GPU compute intervals | 145.451 ms | 144.907 ms |
| Observed GPU interval span | 1,062.371 ms | 1,070.008 ms |

At this trace abstraction level the custom primitive did **not** reduce command-buffer, compute-command or GPU-compute-interval counts; aggregate GPU compute duration was nearly unchanged (`Metal/native = 0.9963`). An interval is not necessarily one kernel dispatch, so the count is not presented as a kernel-launch count. The template's exported counter metadata exposed only `RT Unit Active`; it did not expose DRAM bandwidth, occupancy, cache-hit or useful Roofline counters. Consequently this capture supports a narrower conclusion: the idealized 16.67% logical-traffic reduction did not translate into an observable interval-count reduction, while the actual low-level bottleneck remains unresolved.

Fewer graph operations did not produce lower frozen latency. `native.dot` and `compiled.dot` retain separate Add and RMSNorm primitives; `metal.dot` records the custom primitive. DOT is graph evidence, while [the compact trace receipt](../results/metal-residual-rmsnorm-v1/trace-diagnostic.json) records process-attributed interval evidence and hashes of machine-local XML exports. Neither establishes sustained bandwidth or occupancy. The original performance study remains authoritative: `accepted=false`, custom mode stays disabled by default, and there was no trace-guided retuning after observing results.

## Evidence and offline reproduction

[Raw audit](../results/metal-residual-rmsnorm-v1/audit/run.json), [nine-process timing receipt](../results/metal-residual-rmsnorm-v1/benchmark/run.json), [independent summary](../results/metal-residual-rmsnorm-v1/summary.json), [runner](../experiments/metal_residual_rmsnorm.py), [stdlib verifier](../experiments/verify_metal_residual_rmsnorm.py).

```bash
python -m experiments.verify_metal_residual_rmsnorm \
  --audit-root results/metal-residual-rmsnorm-v1/audit \
  --benchmark-root results/metal-residual-rmsnorm-v1/benchmark
python -m experiments.analyze_metal_kernel_cost \
  --output runs/my-metal-cost-model.json
python -m experiments.profile_metal_residual_rmsnorm \
  --mode metal --rows 2048 --width 896 --warmup-batches 5 \
  --batches 5000 --chain 1 --emit-progress --startup-delay 30
python -m experiments.summarize_metal_trace_export \
  --application runs/native-application.xml \
  --gpu runs/native-gpu.xml --output runs/native-trace-summary.json
python -m unittest tests.test_metal_residual_rmsnorm \
  tests.test_metal_residual_rmsnorm_study tests.test_verify_metal_residual_rmsnorm \
  tests.test_kernel_cost_model tests.test_profile_metal_residual_rmsnorm \
  tests.test_metal_trace_summary -v
# Only on a configured, supported Metal runtime:
RUN_METAL_TESTS=1 python -m unittest tests.test_metal_residual_rmsnorm -v
```

The offline verifier independently recomputes raw timing arithmetic, aggregation, token parity, recorded error consistency and all gates. It requires exact file coverage and binds archived/current source, protocol, model and runtime identities. Fault regressions cover tampering, missing samples, PID reuse, order, token timestamps, teacher forcing, false equality/scores, source drift and failed/over-budget workers. It does not independently regenerate GPU tensors; hashes are not proof against coordinated evidence fabrication. Linux CI checks evidence and tests, not Metal execution.

No model/head/scaler/adapter parameters or credentials are distributed. Prior startup 3.487×, warm-feature 13.070× and complete warm-request 1.241× belong to separate studies and are unchanged. Existing QA coverage/refusal limitations and all prior failed quality confirmations remain unchanged.
