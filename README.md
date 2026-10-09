# 大模型推理优化与性能分析

**简体中文** | [English overview](README.en.md)

[![Offline checks](https://github.com/kimzclandi/inference-compression-lab/actions/workflows/tests.yml/badge.svg?branch=codex%2Fresearch-prerelease)](https://github.com/kimzclandi/inference-compression-lab/actions/workflows/tests.yml)

**Inference Compression Lab** 是在 NUS 实验室期间持续迭代的个人研究项目，围绕固定模型负载中的重复计算和推理耗时展开。项目分别研究共享前缀请求的 KV 复用、CPU 问答流水线的冗余计算，以及 Attention／Metal 算子改动能否在原生框架对照下带来收益。

研究按问题逐项推进：先固定输入、计时范围和对照实现，再检查输出一致性、时延与内存；达到门槛的改动保留，未通过的候选保持默认关闭，并保存失败记录。各实验使用独立协议，性能数字不串接为一个端到端收益。

**Personal research developed during the maintainer’s time in a NUS lab.** The project evaluates repeated computation and latency in fixed inference workloads through KV reuse, CPU QA hot-path changes and controlled Attention/operator experiments.

已实测 CPU 与 Apple GPU；CUDA／Ascend 尚未实测，未验证生产服务。实现与执行使用 AI 辅助，MLX、PyTorch 和 ONNX Runtime 提供底层框架与通用 kernel；具体实现、归属和证据边界见各研究报告。

[研究发布说明 / Research prerelease](docs/research-prerelease.md) · [Release 与校验附件](https://github.com/kimzclandi/inference-compression-lab/releases/tag/v0.1.0-research.4) · [系统与代码导览](docs/qa-system-overview.md) · [MIT](LICENSE) · [数据许可](DATA_LICENSE.md) · [第三方归属](THIRD_PARTY.md)

## 项目主线 / Project questions

- **重复前缀是否需要重复计算？** 为 Qwen 请求实现 KV 复用、容量限制与失败状态保护，并比较固定请求循环和直接重算。后续单独对照 MLX 原生 Cache，预留策略未通过加速门槛。
- **CPU 流水线中哪些计算可以精确省去？** 对文本规范化与候选特征计算做精确剪枝，重放 896 条记录检查特征、分数与决策，再分别测量完整热计算和初始化。
- **KV Cache 淘汰能省多少、代价多大？**（草稿，进行中）[sink + 滑动窗口淘汰实验](docs/kv-eviction-streaming.md)目前只有淘汰策略与 CPU 单元测试，协议未冻结，尚无模型运行结果。
- **减少算子边界是否一定更快？** 以原生及编译实现为对照验证 Attention／Metal 候选。新的 [Q8 QKV 投影合并实验](docs/qkv-projection.md)数值检查通过，但未达加速门槛；实现与实验记录已收录，候选仍默认关闭。

## 从这里开始 / Start here

|方向 / Topic|实现与证据入口 / Evidence|当前结论与边界 / Scope|
|---|---|---|
|模型与请求耗时|[CPU 热路径](docs/qa-risk-pruning.md)、[初始化](docs/qa-risk-startup.md)、[真实 Qwen 请求队列](docs/qwen-request-scheduling.md)|热路径与初始化是不同计时范围；请求调度总体未通过采用门槛|
|KV Cache|[固定容量追加](docs/kv-append-mps.md)、[MLX 原生 Cache 强对照](docs/qwen-cache-reservation.md)|相对逐步 `cat` 的收益不能外推成熟框架；原生 Cache 预留未达到加速门槛|
|Attention|[掩码语义与后端协议](docs/attention-backend-study.md)、[M4 Max 框架对照](docs/attention-mps-study.md)|CPU 语义与固定 MPS 输入已验证；不是 CUDA 实测或整模型加速|
|Metal kernel 与数值分析|[Residual Add + RMSNorm](docs/metal-residual-rmsnorm.md)、[GQA decode](docs/gqa-shared-decode.md)、[逐层诊断](docs/gqa-shared-diagnostic.md)|保留 kernel 速度门槛失败和 GQA 模型 K/V 门槛失败；候选未获采用|
|质量与完整证据|[Qwen 质量确认失败](docs/qwen-confirmation-study.md)、[声明到源码与原始记录](docs/EVIDENCE_MAP.md)|历史有限 QA 样本通过点门槛，后续覆盖率优化未通过联合质量验收；不是生产 QA 服务|

## QKV 投影合并实验

已实现原 Q8 packed 参数按输出维合并，并完成两个提示长度的真实模型实验；完整 Q/K/V、logits、tokens 与最终 KV 逐位一致。但微测试未达到 1.05×门槛，整模型也未通过强对照速度门槛；额外常驻 packed 参数约25.15 MiB，候选保持显式启用。[实现、完整结果与复现边界](docs/qkv-projection.md)。

An opt-in packed QKV projection implementation preserves full numerical outputs on the fixed model workload, but fails the preregistered micro and end-to-end speed gates. It adds 25.15 MiB of packed tensors. This is an upstream-kernel scheduling experiment, not a new low-bit kernel or an accepted acceleration.


## 相同运算量的同步诊断

固定32次有依赖运算，改变同步粒度后调用级计时明显变化；新增三臂进程级Metal trace和区间并集核验。该比例不是kernel或模型加速，底层硬件瓶颈仍未闭环。[协议、原始记录与边界](docs/metal-sync-granularity.md)。

## 独立机制诊断与工作负载边界

新增一次未通过速度门槛的 Metal 归约干预，以及五类固定 Cache 访问负载。结果保留失败、零命中无收益和适用边界；[代码、协议与逐次记录](docs/mechanism-diagnostics.md)。

## 解决什么问题 / Problem and scope

模型量化以后，体积、速度和答案质量不一定同时改善。本项目固定模型 revision、数据、硬件与计时范围，研究真实 INT8/Q4/Q8 推理、错误定位、正确性排序和拒答之间的取舍，并用冻结协议、逐题原始记录与失败后关闭的入口约束结论。

The project separates model size, measured latency and answer quality. It combines fixed-revision CPU/MLX experiments with raw-record audits, correctness ranking, abstention and fail-closed runtime checks. MLX and ONNX Runtime supply quantizers and base inference kernels; the project contributes experiments, diagnostics, runtime controls, evidence verification and an opt-in Metal fusion experiment. Implementation and execution are AI-assisted.

## 结果与限制 / Results and limits

|研究|已测结果|边界|
|---|---|---|
|抽取 QA 基础管线，M4 Max CPU|FP32 72.052 ms；ORT 动态 INT8 54.188 ms，约 1.33×|固定单窗口、8 输入、6 进程轮换；不含后续排序决策，不外推并发服务|
|历史 INT8 正确性排序与拒答|128 题接受 27 题，27 题 EM 正确；可回答覆盖 27/64 = 42.19%；不可回答误接受 0/64|四篇公开 benchmark 文章；Wilson 95% 精度下界约 87.54%，不是总体 90% 保证；整篇文章可能全部拒答|
|包含排序和拒答的历史 INT8 管线|另测 69.600 ms；首次完整证据核验和模型加载约 18.04 s|与基础管线计时范围不同，不能套用 1.33×|
|扩大排序头训练 256→2,304 题；五/十六特征树|训练执行完成，但 192 题开发集全部预设阈值均未通过联合门槛|没有提高已验证的可用覆盖率；预留评估未运行|
|Qwen block-10 回退确认|EM 29/128 vs Q4 26/128；确认门槛失败|保留负结果；不能推断该方法普遍无效|

后续 [PR #14 语义排序实验](https://github.com/kimzclandi/inference-compression-lab/pull/14) 修复了 PCA 训练/推理投影不一致，但仍未通过质量门槛；其 Linux 分数重建也超过固定误差限制。该分支不属于本次技术验收通过的发布快照，失败结果保持公开可审阅。不能把它隐藏成一次成功优化。

The encoder is frozen. Learned correctness heads are fitted locally and rebuilt from verified archived training inputs; neither base weights nor learned head/scaler parameters are distributed. Public samples support reproduction and ablation, not future unseen confirmation. QA v1's original quality and quantization noninferiority failures remain unchanged.

`v0.1.0-research.2` 收录[正确性特征精确剪枝](docs/qa-risk-pruning.md)：同机固定负载下，特征提取 14.042→1.074 ms，包含排序与拒答的 INT8 热请求计算流水线 68.681→55.342 ms（1.241×）。896 条已公开记录逐项复算无差异；质量结论未改善。旧 `v0.1.0-research.1` 保留原样。

`v0.1.0-research.2` includes [exact risk-feature pruning](docs/qa-risk-pruning.md): 13.07× faster feature extraction and 1.241× faster complete warm INT8 computation on the fixed same-host workload. All 896 published-record replays retain identical features, scores and decisions. No new QA-quality claim; the original research prerelease remains unchanged.

同一版本还将精确剪枝用于[完整启动证据核验](docs/qa-risk-startup.md)，仍逐行重算 896 条特征。唯一一次固定六进程对照中，完整 `load_service` 初始化中位数 **17.919→5.139 秒（3.487×）**，三对均更快；完整核验结果与三个公开样例响应一致。包括证据核验、训练头重建、资产验证和模型加载；不含进程启动、前置导入或请求推理，使用正常 OS 文件缓存，不是冷磁盘启动或 QA 质量提升。

The same exact pruning now accelerates [complete startup evidence verification](docs/qa-risk-startup.md), recomputing all 896 feature rows. One fixed six-process study reduced median full `load_service` initialization from **17.919 to 5.139 seconds (3.487×)**, with all three pairs faster and identical verification and disclosed functional outputs. It includes evidence verification, head reconstruction, asset validation and model loading, excluding interpreter/pre-call imports and requests. Ordinary OS caching applies; no cold-disk, warm-request or quality improvement is inferred.

`v0.1.0-research.3` 另收录 [Qwen 最后位置 prefill 需求裁剪](docs/qwen-demand-prefill.md)，进入模型内部计算图。固定2048-token负载下，TTFT相对原路径1.407×，但相对强分段对照仅1.042×，未达到冻结的1.05×门槛。74条公开生成序列与完整KV检查一致，logits存在舍入差异；候选未默认启用；research.3 收录其可复现负结果，research.2 保持原样，不声称质量提升或新kernel。

`v0.1.0-research.3` adds [last-position Qwen prefill](docs/qwen-demand-prefill.md), a model-graph experiment with full, head-only and split-last controls. Its 1.042× TTFT speedup over the strongest control missed the preregistered 1.05× gate despite output parity. The candidate remains disabled by default. Research.3 publishes the reproducible negative result; research.2 remains unchanged. No new quality or kernel claim.

`v0.1.0-research.4` 收录 [Residual Add + RMSNorm Metal 融合 kernel](docs/metal-residual-rmsnorm.md)：已真实 GPU 执行，300 组算子与 32 组模型检查逐位一致，74 条公开生成序列保持一致；但固定主场景为 22.835→28.277 μs（相对编译原生对照慢约 23.8%），整模型无确认加速。候选仅显式启用；research.4 发布其实现与负结果，research.3 保持原样，没有调参重跑。归约与舍入参考 MLX MIT 实现，不声称原创 RMSNorm 或低比特 kernel。

`v0.1.0-research.4` includes a real [Residual Add + RMSNorm Metal fusion](docs/metal-residual-rmsnorm.md). Numerical/output checks passed, but primary pair latency was 23.8% slower than compiled native and no model speed gate passed. It remains opt-in; research.4 publishes its implementation and negative result, while research.3 remains unchanged. Reduction/rounding follows attributed MLX MIT source; this is not a new RMSNorm or low-bit kernel algorithm.

后续静态诊断在不重跑性能实验的前提下补充了[逻辑访存成本模型](docs/metal-residual-rmsnorm.md#post-hoc-logical-traffic-diagnosis--not-a-new-benchmark)：由于融合后仍需同时输出 residual 与 normalized tensor，理想模型只减少 1/6 的 primitive-boundary 逻辑字节，traffic-only 上限为 1.20×。该模型不是实测 Roofline，也不改变 `accepted=false`；它明确解释了该融合候选的理论空间和仍未确认的硬件瓶颈。

另用 Xcode Metal System Trace 对主形状 `2048×896` 做了 PID 归属的后验诊断：native 与自定义 Metal 各完整执行 5,000 次，均观察到 5,007 个 GPU compute intervals，累计 GPU compute duration 仅相差约 0.37%。这说明该融合在 Instruments interval 层面没有减少 command-buffer/compute-interval 数量；但模板只暴露 `RT Unit Active`，没有带宽、occupancy 或 cache counter，因此仍不能声称完成 Roofline 定位。该 trace 不是新 benchmark，不推翻固定研究中自定义 Metal 慢约 23.8% 与 `accepted=false` 的结论。

新增 [GQA 组内共享 K/V 的 Metal decode 特化](docs/gqa-shared-decode.md)：8 个固定算子输入 × 3 条实现路径共 24 项输出检查通过；真实 Qwen 128-token 提示的 token/logprobs 达到冻结门槛，但 48 份最终 K/V 中 10 份失败。因此长提示模型场景与全部性能计时停止；没有 Attention 或整模型加速结论，候选默认关闭。原理来自已有 online softmax/split-K，非原创 Attention 架构。

后续[固定输入数值诊断](docs/gqa-shared-diagnostic.md)用原生复跑、适配器原生控制和 384 次同输入 float64 对照，精确重现上述输出与最终 K/V。首个观察到的差异在第 1 个 decode 输入、第 5 层 Attention 输出的两个 FP16 值（各 1 ULP，层索引从 0 开始）；同输入算子检查均通过，但模型 K/V 的 10 份失败仍存在。已归档完整 K/V 与坐标供 CPU 重放；没有确定具体浮点机制、修复模型门槛或开始性能计时。

[exp 函数族单因素干预](docs/gqa-exp-choice.md)固定上述 384 组输入与归约顺序，仅替换三个 exp 调用：原 197 处原生/候选差异中消除 48 处、保留 149 处，同时新增 37 处，未达到事前冻结的全部消除门槛。全部 344,064 个输出参与 FP64 误差检查，85 个变化的 FP16 值中 47 个改善、38 个恶化。此结果仅说明函数族干预影响局部输出；未采纳为模型修复，未新跑模型或性能，旧 kernel 和失败证据保持不变。

## 运行与复现 / Run and reproduce

新增 [Attention 掩码与后端实验](docs/attention-backend-study.md)：CPU 上 36 组输入、两条实现路径通过独立 float64 参考检查，并验证带缓存 decode/chunk 与完整 prefill 对应位置一致。CUDA 后端对照协议与执行脚本已准备，**尚未在 NVIDIA GPU 执行，没有 CUDA 加速结论**。PyTorch 提供优化 kernel，本项目贡献实验、语义检查和证据审计。

另已完成 [M4 Max Attention 对照实验](docs/attention-mps-study.md)：8 个固定 prefill/decode 形状中，PyTorch MPS SDPA 相对更快的显式 FP16 路径，同步 API 延迟改善 **1.422–1.771×**；24 组完整输出通过 float64 参考检查。比较的是框架实现，非原创 kernel、纯 GPU 时间或整模型加速。NumPy 参考计算告警及独立非 BLAS 复核结果均保留；CUDA/Ascend 仍未实测。

[固定容量 KV 追加实验](docs/kv-append-mps.md)补充了缓存更新成本：4 个固定 64-step 场景，预分配相对逐步 `cat` 的仅追加循环约 **5.51–5.97×**，加入同一显式 FP32 Attention 后约 **1.48–1.57×**；512 次完整输出检查通过。初版在新增缓存输入上的 MPS SDPA 正确性检查失败，未计时；失败证据公开，修复版两组统一显式 Attention，旧输入、容差及速度门槛不变。这不是 SDPA、整模型或生产服务加速，也不能与上一实验倍率相乘。

Python 3.11/3.12。无需下载模型即可核验源码、证据和存档排序头；依赖安装需要网络或本地 wheel。输出目录必须全新。

```bash
git clone --branch codex/research-prerelease https://github.com/kimzclandi/inference-compression-lab.git
cd inference-compression-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r configs/qa-nonlinear/requirements.txt
.venv/bin/python -m experiments.verify_release --require-license
.venv/bin/python -m experiments.verify_qa_risk_pruning
.venv/bin/python -m experiments.verify_qa_risk_startup
.venv/bin/python -m experiments.verify_qwen_demand_prefill \
  --audit-root results/qwen-demand-prefill-v1/audit \
  --benchmark-root results/qwen-demand-prefill-v1/benchmark
.venv/bin/python -m experiments.review_qa_evidence --output-dir runs/my-review
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.qa_nonlinear verify \
  --folder results/qa-nonlinear-v2 --output runs/my-nonlinear-audit.json
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.qa_rich verify \
  --folder results/qa-rich-v1 --output runs/my-rich-audit.json
.venv/bin/python -m unittest discover -s tests -v
```

这些检查重算历史结果，不产生新的质量确认。完整新增特征核对、扩大训练重建、无 Git 流程与真实模型演示见 [发布说明](docs/research-prerelease.md) 和 [运行时复现](docs/release-reproduction.md)。可选 MLX/ORT 路径缺依赖时会跳过，查看 skip 原因；测试通过不是跨平台真实推理成功。

`verify_release`保留历史基线验收；当前分支的`review_qa_evidence`/`verify_release_rc4`以及原型启动还会校验剪枝源码与固定证据身份。只通过旧基线检查不能授权当前优化入口。历史计时记录保持原样，旧计时核验命令兼容转发到加强后的检查器。

Historical baseline checks alone do not authorize the optimized runtime. Current full acceptance and prototype startup also bind the pruning implementation to the fixed evidence manifests; modified or incomplete receipts fail closed.

新增 [真实 Qwen 原生 Cache 强对照](docs/qwen-cache-reservation.md)：MLX 原生已按 256 token 扩容；按请求预留容量在 3 个长度的总请求耗时比为 **0.994×/1.025×/0.992×，均未通过加速门槛**。完整生成 token、logits 和最终 KV 一致，最终分配 KV 张量减少 17.5%–43.75%，不代表峰值设备内存减少。此前相对逐步 `cat` 的微基准收益不能外推到成熟框架。

## 代码与证据 / Implementation and evidence

|模块|入口|
|---|---|
|抽取、token/span 与窗口解码|[lab/extractive_qa.py](lab/extractive_qa.py)|
|运行时身份校验与拒答入口|[lab/qa_specialist_runtime.py](lab/qa_specialist_runtime.py)、[serve CLI](experiments/serve_qa_specialist.py)|
|正确性头、校准与验收|[lab/qa_risk_calibration.py](lab/qa_risk_calibration.py)、[固定排序报告](docs/qa-risk-study.md)|
|第三特征与两窗反例|[固定消融](docs/qa-span-gap-ablation.md)、[不重叠竞争片段实验](docs/qa-coverage-gap-study.md)|
|扩大训练与非线性对照|[2,304 题研究](docs/qa-expanded-ranking-study.md)、[非线性研究](docs/qa-nonlinear-study.md)|
|Qwen 量化与缓存生命周期|[量化诊断](docs/qwen-quantization-study.md)、[确认失败](docs/qwen-confirmation-study.md)、[缓存研究](docs/qwen-cache-lifecycle-study.md)|
|MiniLM CPU 与负载实验|[Mac 复现](docs/mac-reproduction.md)、[长度分桶](docs/length-bucketing-study.md)|
|自定义 Metal kernel、成本模型与 trace 诊断|[Residual Add + RMSNorm 实现、固定负结果、逻辑访存分析及 Instruments interval 证据](docs/metal-residual-rmsnorm.md)|

这是个人研究项目；学校/团队、字节经历及历史 Jetson 工作不属于本仓库已验证成果。没有本项目 TensorRT/CUDA/昇腾部署或原创量化 kernel 声明。源码采用 MIT；公开数据和第三方内容保留原许可。

[核心表述、代码与原始证据索引](docs/EVIDENCE_MAP.md) · [公开 Instruments 区间记录与只读重算](results/metal-trace-rows-v1/README.md)。逻辑访存分析不是实测 Roofline，Metal 性能门槛仍失败。

[真实 Qwen 请求队列实验](docs/qwen-request-scheduling.md)使用相同 MLX-LM 原生逐请求生成器比较 FCFS 与有界短预算调度：240 次请求输出一致，burst 平均 TTFT 为 630.066→478.365 ms；staggered 为 455.441→398.439 ms，未达到冻结的 1.15× 门槛，且 p95 完成延迟增加 2.97%。总体不接纳，不宣称连续批处理、模型计算加速或生产尾延迟改善。
