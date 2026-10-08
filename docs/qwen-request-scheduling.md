# Qwen 请求队列：平均首 token 与尾延迟的取舍

本轮为单 worker 的真实模型请求调度实验。相同 MLX-LM 原生逐请求生成器上的 FCFS 为控制组，候选按已知输入长度与生成上限排序，并限制老请求最多被越过三次。两个冻结到达 trace 的 240 次请求均保持 token 一致，但候选只在 burst 通过联合门槛，staggered 未达到预设平均 TTFT 改善门槛；整体 `accepted=false`，不作为普遍加速方案。

## 原生能力与贡献边界

固定 MLX 0.29.3 / MLX-LM 0.26.3。实读已安装源码并保存哈希：原生 `generate_step` 已有异步预取、分块 prefill、greedy 采样和 KVCache；本实验保留这些机制。该版本 `server.py` 使用 Python `HTTPServer`，没有本实验需要的 admission 策略接口。未升级框架，也未用朴素 Python 自写模型生成替代原生实现。

参考：[固定版本生成器](https://github.com/ml-explore/mlx-lm/blob/v0.26.3/mlx_lm/generate.py)、[固定版本 server](https://github.com/ml-explore/mlx-lm/blob/v0.26.3/mlx_lm/server.py)。只描述本次固定版本，不推断后续版本或其他框架能力。

本项目新增有界 admission policy、实际到达线程、排队与逐 token 计时、取消/错误后的资源释放和证据复核。每次生成独占一个 worker，使用私有原生 Cache；没有同时执行多个 GPU 请求。不是 continuous batching、vLLM 服务对照、生产服务、原创 kernel 或 CUDA/Ascend 实测。实现、实验与执行有 AI 辅助。

## 执行前冻结

协议、策略、runner 和单元测试在 `8789f74` 提交后开始唯一一次 GPU 运行。[完整协议](../configs/qwen-request-scheduling-v1.json)绑定模型全部文件、分词器及上游源码身份，没有下载/重新转换权重。

- Apple M4 Max、48 GiB；同一 Qwen2.5-0.5B-Instruct Q8 模型，KV 浮点。
- 12 个自编合成文本请求：prompt 128/1024，生成上限 8/32，各组合三次。强制生成规定数量，包括 EOS 之后；不是模型质量测试或真实业务流量。
- 种子 20261008 确定请求顺序；burst 全部立即提交，staggered 每 20 ms 提交、顺序反转。独立 CPU 线程按真实单调时钟提交，保存计划与实际时间；worker 只能选择已实际到达的请求。
- 候选分数 `prompt_tokens + 32 × output_tokens`；32 是事前固定启发式，未按本轮性能拟合。只使用已知的请求上限，不读取未来实际耗时。最多三次越过限制重排次数，不能保证等待时长或 SLA。
- 性能前 12 组完整词表 log-probability 重复对照（atol=rtol=0.001）；每请求预热一次。2 trace × 5 轮 × 2 策略 × 12 请求，共 240 条原始计时；每轮随机策略顺序，全部保留，无剔除或重跑。
- 每 trace 的轮均值 TTFT 中位数比至少 1.15，且至少 4/5 轮更快；候选 p95 完成延迟至多为基线 1.05 倍、吞吐至少 0.95 倍、MLX allocator peak 至多 1.10 倍。两 trace 全通过才接纳。
- 限制 900 秒、25 MiB 输出、4 GiB MLX peak；任何正确性或预算失败停止并保存部分记录。没有付费计算。

## 计时定义

每条请求保存实际 arrival、dispatch、生成开始、所有 token 交付时刻和最终同步完成时刻。Queue=dispatch−arrival；TTFT=first token−arrival；Total=end−arrival；TPOT=(last−first)/(N−1)。Service=end−dispatch。吞吐是全 trace 交付 token 数除以从生产线程启动到全部完成的实际 makespan。

计入排队、线程调度、Cache 创建、原生预取/采样、Python 开销及 GPU 同步；不含权重加载、分词、网络和完成后的 JSON 序列化。TPOT 使用 native generator 交付间隔，不能解释为纯 GPU kernel 时间。MLX `get_peak_memory` 是 allocator 统计，包含模型与临时张量，不能改写成进程 RSS、系统总内存或完整设备使用量。

## 结果与失败分析

以下是各策略五轮对应统计量的中位数；每轮 p95 使用 12 请求的线性插值，仅描述固定 trace，不估计线上总体尾延迟。

|指标|burst FCFS|burst 短预算|staggered FCFS|staggered 短预算|
|---|---:|---:|---:|---:|
|平均排队 ms|579.605|427.921|404.795|347.785|
|平均 TTFT ms|630.066|478.365|455.441|398.439|
|p50 TTFT ms|595.319|457.063|492.379|351.876|
|p95 TTFT ms|1163.626|1021.886|812.664|869.411|
|p50 完成 ms|643.126|505.260|540.780|432.789|
|p95 完成 ms|1180.406|1106.555|892.688|919.236|
|平均 TPOT ms|2.634|2.648|2.647|2.658|
|吞吐 tokens/s|200.208|200.116|199.680|198.956|
|MLX peak bytes|1789312944|1789312944|1789312944|1789312944|

- burst 平均 TTFT 比 1.317×，五轮都改善，联合门槛通过。
- staggered 平均 TTFT 比 1.143×，五轮都改善但未达到 1.15；p95 完成增加 2.97%，p95 TTFT 增加 6.98%。p95 TTFT 是预先保存的描述指标，并非执行后新加的否决条件。
- 总体不接纳。平均延迟改善来自请求重排，吞吐与每请求生成成本基本不变；没有证据表明 Attention、Cache kernel 或模型计算得到加速。平均指标改善不保证每条请求或尾部改善。

240 条计时请求 token 全部匹配独立 fresh-cache 基准；12 组完整词表检查通过。完整 log-probability 数组仅在内存比较，未公开存档，CPU 验收只核验记录、hash 和算术，不能独立重演该数值比较。

## 取消、失败与资源边界

在真实原生生成器交付第 3 个 token 后分别主动取消、注入 consumer 异常；记录准确交付时刻、Cache offset 和同步关闭时刻。原生预取可能已经执行下一步，因此 close 不声称立即取消 GPU 工作。两路径均先关闭 iterator、同步，然后释放私有 Cache；随后完整新请求 token 与基准一致。

这验证 consumer 生命周期及后续请求隔离，不是 backend 崩溃、GPU OOM、进程崩溃、异步设备取消或内存泄漏压力测试。队列内取消、重复取消、到达约束、越过上限另有 CPU 回归；性能 trace 不混入失败请求。

## 复核与入口

- [策略与流生命周期](../lab/request_scheduling.py) · [真实运行](../experiments/qwen_request_scheduling.py)
- [原始到达、token 时刻与计时](../results/qwen-request-scheduling-v1/trials.json)
- [完整词表核对记录](../results/qwen-request-scheduling-v1/correctness.json) · [取消/失败记录](../results/qwen-request-scheduling-v1/lifecycle.json)
- [固定源文件与身份](../results/qwen-request-scheduling-v1/run.json) · [独立汇总](../results/qwen-request-scheduling-v1/summary.json)

```bash
# Standard library only; no GPU, model or network needed.
python -m experiments.verify_qwen_request_scheduling
python -m unittest discover -s tests -p 'test_request_scheduling*.py' -v
# Optional new reproduction, never overwrite the frozen directory.
# Use an existing identity-matched model and the pinned MLX environment.
python -m experiments.qwen_request_scheduling --model /path/to/existing-q8 --output runs/my-request-study
```

CPU CI 绑定冻结 receipt/source/artifact 身份，重演到达条件下的选择顺序与越过次数，重算全部 Queue/TTFT/TPOT/Total、p50/p95、吞吐和门槛。它不重新执行 Metal 或评估模型质量。旧 Attention、Cache、Metal 和量化实验未修改；不同实验倍率不相乘。

## 最终审查发现的边界缺陷

计时完成后用反例发现：外部调用者若回填早于已有请求的到达时间，原策略只看最老请求是否达到越过上限，可能遮住排在其后的已达上限请求，让后者被第 4 次越过。[反例记录](../results/qwen-request-scheduling-review-v1/before.json)保留。

最终实现改为优先选择所有 ready 请求中最早达到上限者，新增回归。真实到达线程使用当前单调时刻，所以本轮没有回填；CPU 验收以最终实现逐条重演全部 240 次调度，顺序及越过计数完全一致。冻结计时源码与结果未改，性能数字仍属于 `8789f74`；最终边界修复没有重新运行 GPU 性能实验，也不声称已测得修复版本的精确耗时。CPU 验收分别绑定冻结源码与审查后的策略源码。
