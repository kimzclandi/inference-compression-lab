# Independent performance and preservation review / 独立性能与历史保全审计

结论：6 个独立子进程的 **240 条请求测量与 summary 完全一致**，未发现计时、RSS 单位或汇总计算错误。RC3 提交 `e3d311e` 下已跟踪的 **841 个 results 文件全部逐字节保留**；缺失 0、修改 0。审计只读原证据，没有重跑模型或修改历史材料。

| Primary statistic | FP32 | ORT dynamic INT8 |
|---|---:|---:|
| 3 个进程各自中位数的中位数 | 72.05194 ms | 54.18756 ms |
| 进程生命周期峰值 RSS 的中位数 | 1,405,943,808 bytes | 937,639,936 bytes |

按预设公式，FP32/INT8 延迟比为 **1.32967668×**，达到预设 1.05× 点估计门槛。INT8/FP32 峰值 RSS 中位数比为 **0.66691139**。这是该固定 benchmark 的测量结果，不应改写成业务 QA 已通过质量验收或任意设备均获得相同收益。

独立检查包括：原始文件 checksum 精确覆盖；固定顺序 `FP32, INT8, INT8, FP32, FP32, INT8`；每进程 8 个固定输入、5 次重复、40 条记录；以 `SHA256("2026100407" + id)` 排名选择输入；无缺失、重复或重排；逐进程中位数、总体中位数比、峰值 RSS 和加载耗时；240 个输出 hash/输入形状与相应 precision 的既有 calibration 预测一致。benchmark 源码快照、被冻结 runtime 源码的 Git 对象和当前文件 hash 一致。性能审计不使用项目 benchmark 汇总函数。

计时最外层 `perf_counter` 包围完整 `runtime.predict`，包含 tokenizer、所有窗口的同步 ORT CPU `Session.run`、logit 数据转换和穷举 span 解码。8 次 warmup、模型加载和 JSON 保存不在请求延迟内。各组件计时总和小于请求总耗时，剩余差额约 0.060–0.164 ms，与其间的 Python 工作范围一致；这不构成组件性能归因实验。CPU 调用同步，无异步 GPU 漏同步问题。这里没有生成 token，不能叫 TTFT 或 decode tok/s。

The measured eight inputs are **all single-window requests**, with token lengths 135–256. They do not test the maximum eight-window limit. The three processes per precision provide a small local repeated measurement, not a concurrency benchmark, tail-latency SLO, energy estimate or cross-device result. Warm filesystem/model caches may affect initialization; model initialization is separately reported.

RSS uses Darwin `ru_maxrss` byte units. The peak includes interpreter, tokenizer, ONNX Runtime/session creation, model and inference over the process lifetime. It is neither model-only memory nor steady-state RSS; it must not be equated with weight-file size or logical KV bytes. The new supervised risk head, threshold-serving gate, quality-artifact verification, HTTP/network and queueing are **outside this benchmark**. Do not transfer this exact number to a later risk-head service without measuring that path.

Provenance limitation: workers do not independently snapshot runtime source or record OS PIDs. The saved sequential `subprocess.run` calls and command list establish intended per-process isolation; benchmark-commit contents and clean tracked-source status support runtime identity. They are not an external process trace. This gap is disclosed rather than silently upgrading the evidence claim.

Recompute without model loading:

```bash
python3 -B results/qa-specialist-review-v1/performance/audit.py \
  --output /new/nonexistent/performance-audit.json
```

`audit.json` includes the complete 841-file baseline blob/current SHA256 ledger, all source identities, process-level recalculation, exact input IDs and scope findings. Output files are created exclusively and cannot overwrite an earlier audit.
