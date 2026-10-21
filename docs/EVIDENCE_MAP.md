# 核心表述与证据索引

| 表述 | 实现 | 协议、报告及原始记录 |
|---|---|---|
| 896条特征/决策一致；INT8完整热计算68.681→55.342 ms | [精确剪枝](../lab/qa_risk_pruning.py) | [固定研究](qa-risk-pruning.md) |
| 完整load_service 17.919→5.139 s | [服务初始化](../lab/qa_specialist_runtime.py) | [六进程原始记录入口](qa-risk-startup.md) |
| residual-add + RMSNorm；300算子/32模型检查；主形状比compiled native慢23.8% | [Metal源](../lab/kernels/residual_rmsnorm.metal)、[封装](../lab/metal_residual_rmsnorm.py) | [研究报告](metal-residual-rmsnorm.md)、[原始正确性](../results/metal-residual-rmsnorm-v1/audit/)、[固定计时](../results/metal-residual-rmsnorm-v1/benchmark/) |
| 理想边界减少16.67%逻辑字节；两路径各5,007 GPU compute intervals | [访存模型](../lab/kernel_cost_model.py)、[XML解析](../lab/metal_trace_summary.py) | [逻辑模型](../results/metal-residual-rmsnorm-v1/cost-model.json)、[旧回执](../results/metal-residual-rmsnorm-v1/trace-diagnostic.json)、[公开逐区间记录](../results/metal-trace-rows-v1/README.md) |
| Qwen量化回退确认失败；MiniLM同线程INT8负结果 | [诊断](../lab/quantization_diagnostics.py) | [研究发布与各轮证据](research-prerelease.md)、[结果树](../results/) |
| GQA共享Metal decode：算子8输入×3路径通过，模型48份K/V中10份失败；性能未运行 | [接入](../lab/gqa_shared_decode.py)、[kernel](../lab/kernels/gqa_shared_partial.metal) | [协议](../configs/gqa-shared-decode-v1.json)、[报告](gqa-shared-decode.md)、[冻结记录](../results/gqa-shared-decode-v1/run.json) |
| GQA数值诊断：原生重复/适配器控制一致；384同输入参考检查；旧10份K/V失败精确重现，性能0 | [捕获与控制](../experiments/gqa_shared_diagnostic.py)、[独立CPU重放](../experiments/verify_gqa_shared_diagnostic.py) | [冻结协议](../configs/gqa-shared-diagnostic-v1.json)、[诊断报告](gqa-shared-diagnostic.md)、[完整数组与记录](../results/gqa-shared-diagnostic-v1/) |
| exp单因素干预：48处旧差异消除、149处持续、37处新增；假设未通过，未运行模型/性能 | [执行](../experiments/gqa_exp_choice.py)、[独立CPU重放](../experiments/verify_gqa_exp_choice.py) | [协议](../configs/gqa-exp-choice-v1.json)、[报告](gqa-exp-choice.md)、[完整输出与回执](../results/gqa-exp-choice-v1/) |
| KV Cache容量/失败提交语义；固定1024-token前缀循环下降51.09%；480请求token一致 | [LRU](../lab/prefix_cache.py)、[Qwen接入](../lab/qwen_prefix.py) | [生命周期研究及原始trace](qwen-cache-lifecycle-study.md) |
| Attention prefill/decode/chunk CPU语义检查；CUDA性能尚未执行 | [独立参考](../lab/attention_reference.py)、[实验](../experiments/attention_backend_study.py) | [范围与协议](attention-backend-study.md)、[CPU原始记录](../results/attention-cpu-semantics-v1/semantics.json) |
| M4 Max MPS SDPA 8种形状同步API延迟相对更快显式对照1.422–1.771×；不外推整模型/CUDA | [实验与核验](../experiments/attention_mps_study.py)、[独立参考复核](../experiments/verify_attention_mps_reference.py) | [报告及告警边界](attention-mps-study.md)、[原始计时](../results/attention-mps-v1/timings.json) |
| 固定容量KV追加；4种64-step场景含显式Attention约1.48–1.57×；初版SDPA正确性失败 | [缓存](../lab/append_only_kv.py)、[实验](../experiments/kv_append_mps.py) | [失败边界与报告](kv-append-mps.md)、[v2计时](../results/kv-append-mps-v2/timings.json)、[v1失败](../results/kv-append-mps-v1/run.json) |

每项速度来自各自固定负载，不能拼接为统一端到端加速。启动不含进程启动/前置导入，不是冷磁盘测量。480请求为40条固定trace的请求记录。Metal失败结果与所有质量门槛保持原样；逻辑字节不是DRAM实测，Instruments interval不是kernel launch。

[完整CI检查](../.github/workflows/tests.yml)验证工程与冻结记录；Linux CI不执行Metal GPU，也不证明业务模型质量。实现与上游技术归属见[Metal说明](metal-residual-rmsnorm.md#implementation-and-attribution)及[第三方说明](../THIRD_PARTY.md)。未声称原创低比特kernel、CUDA/Ascend实现或生产部署。

请求队列单独研究：[原生生成器与有界策略](../lab/request_scheduling.py)、[固定协议](../configs/qwen-request-scheduling-v1.json)、[240 请求与负结果报告](qwen-request-scheduling.md)、[实际到达与逐 token 原始记录](../results/qwen-request-scheduling-v1/trials.json)。仅 burst 通过联合门槛，staggered 未通过平均 TTFT 门槛；不合并为通用加速结论。

## 原生框架强对照

[Qwen Cache 容量预留](qwen-cache-reservation.md)：[适配代码](../lab/mlx_cache_reservation.py)、[冻结协议](../configs/qwen-cache-reservation-v1.json)、[正确性记录](../results/qwen-cache-reservation-v1/correctness.json)、[60 次计时](../results/qwen-cache-reservation-v1/samples.json)、[CI 离线验收](../experiments/verify_qwen_cache_reservation.py)。3 个长度均未通过性能门槛；仅分配 KV payload 减少，无质量、峰值内存或生产能力结论。

[新增机制诊断与收益边界](mechanism-diagnostics.md)。与旧冻结实验分开保存，不替换历史结果。

[新增相同运算量的同步粒度与Metal trace诊断](metal-sync-granularity.md)：确认计时敏感性，不宣称kernel加速或根因已确定。

## QKV projection packing

See [fixed Q8 projection study](qkv-projection.md), [implementation](../lab/qkv_projection.py) and [public result](../results/qkv-projection-v1/summary.json). Full local array comparisons passed; both micro and model speed acceptance failed. Public CI verifies scalar consistency only; private arrays are required for numerical replay. Extra packed tensor storage is retained, and the candidate is opt-in.
