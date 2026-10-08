# 核心表述与证据索引

| 表述 | 实现 | 协议、报告及原始记录 |
|---|---|---|
| 896条特征/决策一致；INT8完整热计算68.681→55.342 ms | [精确剪枝](../lab/qa_risk_pruning.py) | [固定研究](qa-risk-pruning.md) |
| 完整load_service 17.919→5.139 s | [服务初始化](../lab/qa_specialist_runtime.py) | [六进程原始记录入口](qa-risk-startup.md) |
| residual-add + RMSNorm；300算子/32模型检查；主形状比compiled native慢23.8% | [Metal源](../lab/kernels/residual_rmsnorm.metal)、[封装](../lab/metal_residual_rmsnorm.py) | [研究报告](metal-residual-rmsnorm.md)、[原始正确性](../results/metal-residual-rmsnorm-v1/audit/)、[固定计时](../results/metal-residual-rmsnorm-v1/benchmark/) |
| 理想边界减少16.67%逻辑字节；两路径各5,007 GPU compute intervals | [访存模型](../lab/kernel_cost_model.py)、[XML解析](../lab/metal_trace_summary.py) | [逻辑模型](../results/metal-residual-rmsnorm-v1/cost-model.json)、[旧回执](../results/metal-residual-rmsnorm-v1/trace-diagnostic.json)、[公开逐区间记录](../results/metal-trace-rows-v1/README.md) |
| Qwen量化回退确认失败；MiniLM同线程INT8负结果 | [诊断](../lab/quantization_diagnostics.py) | [研究发布与各轮证据](research-prerelease.md)、[结果树](../results/) |
| KV Cache容量/失败提交语义；固定1024-token前缀循环下降51.09%；480请求token一致 | [LRU](../lab/prefix_cache.py)、[Qwen接入](../lab/qwen_prefix.py) | [生命周期研究及原始trace](qwen-cache-lifecycle-study.md) |
| Attention prefill/decode/chunk CPU语义检查；CUDA性能尚未执行 | [独立参考](../lab/attention_reference.py)、[实验](../experiments/attention_backend_study.py) | [范围与协议](attention-backend-study.md)、[CPU原始记录](../results/attention-cpu-semantics-v1/semantics.json) |

每项速度来自各自固定负载，不能拼接为统一端到端加速。启动不含进程启动/前置导入，不是冷磁盘测量。480请求为40条固定trace的请求记录。Metal失败结果与所有质量门槛保持原样；逻辑字节不是DRAM实测，Instruments interval不是kernel launch。

[完整CI检查](../.github/workflows/tests.yml)验证工程与冻结记录；Linux CI不执行Metal GPU，也不证明业务模型质量。实现与上游技术归属见[Metal说明](metal-residual-rmsnorm.md#implementation-and-attribution)及[第三方说明](../THIRD_PARTY.md)。未声称原创低比特kernel、CUDA/Ascend实现或生产部署。
