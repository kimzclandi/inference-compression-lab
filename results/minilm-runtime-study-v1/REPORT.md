# 可交付成果：MiniLM 量化与 CPU 推理优化

2026-09-30；Apple M4 Max、48 GiB、macOS 27.0、Python 3.12.13、ORT 1.30.0 CPUExecutionProvider。源码与实验由 Codex 在用户授权下协助完成；量化核心使用 ORT 官方 API。不是 Jetson/TensorRT 或 NUSRI 历史项目实测。

## 交付结论

1. 在已量化模型不变时，4→8 intra-op 线程使 batch=1、padding64 的纯推理中位数降低 14.51%，分词+推理+池化路径降低 14.41%。这是运行配置优化，不是量化算法自身带来的加速。
2. STS-B test 全部1,379对首次在本项目冻结配置后评估，FP32 Spearman=0.820301395，INT8=0.818753223，下降 0.001548171；模型文件从 86.22 MiB 到 56.04 MiB，缩小 35.00%。
3. 公平比较调优后的 FP32 与 INT8，收益依赖形状：长度64及更大形状中 INT8仍较慢；独立请求确认的长度16负载则出现真实 INT8收益。不能把短序列收益外推到全部输入。
4. 已提供可复用 runtime、冻结配置、带哈希校验的本地句向量 CLI、完整原始证据与复跑命令。30项本地测试通过；标准库环境19项通过、11项按依赖/本地模型条件跳过。

## 从问题到干预

原先4线程基线中 INT8 文件更小但推理慢，初版只说明了现象。现在固定量化策略，比较1/2/4/8线程；每个候选用独立进程、串行运行，按固定随机种子打乱配置顺序。候选中8线程最好，1线程反而更慢；没有假定减线程一定降低开销，也不宣称8线程是所有候选之外的全局最优。

开发阶段比较 FP32、INT8按通道、INT8按通道+一个MatMul回退共12种配置，3轮、每轮50预热+200测量；按开发中位数选择每种精度的线程数。最终量化候选为全按通道，回退方案仍保留在历史误差消融中，没有强行推广为最佳。

配置在 [selection.json](selection.json) 中冻结。确认阶段改用 validation 第128—143行的请求，16请求循环，8轮，每轮50预热+200测量。两组请求的行范围与token输入哈希均保留。每个计时单元使用新的worker；一次只跑一个模型，没有并行计时。

## 主负载：B=1，padding S=64

| 配置 | session 中位数 ms | 每轮P95的中位数 ms | 分词+推理+池化中位数 ms |
|---|---:|---:|---:|
| FP32 / 4线程 | 1.966125 | 2.073462 | 2.048948 |
| INT8 按通道 / 4线程 | 2.260927 | 2.342008 | 2.332521 |
| FP32 / 8线程 | 1.730187 | 1.828876 | 1.811334 |
| INT8 按通道 / 8线程 | 1.932885 | 2.012323 | 1.996458 |

主指标为各轮中位数再取中位数。session 包括相同的Python请求轮转开销；pipeline包括tokenization、session.run、mask mean pooling与L2 normalization，不包括模型加载、网络、队列或机器人动作。

同一INT8模型的4→8线程加速比为 1.1697×，按轮配对bootstrap的95%区间为 [1.1658453177561867, 1.1727352947434024]。只表达本机轮次不确定性，不能把1,600个相关计时点当成独立机器样本。原始记录见 [confirm/raw](confirm/raw/)。

FP32也从4线程调优到8线程，防止拿未优化浮点对照夸大量化收益。INT8虽然略快于原始4线程FP32，但仍慢于8线程FP32，因此报告明确分解两种干预。

## 质量：首次 test split 验证

使用固定 revision `ab7a5ac0e35aa22088bdcf23e7fd99b220e53308` 的官方 STS-B test，1,379对，质量协议 batch=16、最长256、动态padding，与前轮质量定义一致。下载在配置冻结之后；此前选层/线程使用的validation结果不是新盲测集。

INT8相对FP32的Spearman差值95%探索性区间为 [-0.002642854470559683, -0.00046435055208269313]，提示小幅质量损失，不应写成“精度完全不变”。点估计损失0.001548低于预先设定的0.005工程预算；该预算不是行业标准。没有审计预训练数据重叠，公开test也不是新建盲测集；bootstrap未建模重复句子的依赖。

质量评测最长256与固定shape性能实验是不同负载，不能声称主表的时间就是整套test的单样本耗时。预测、gold标签、行号及配对bootstrap原始值见 [quality](quality/)，原始句子和parquet未提交。

## 多形状边界与一次追加确认

以下为固定开发选择之后的探索性形状测试，均为8线程；每形状3轮，每轮50预热+200测量。较大batch指标是每批延迟，不是每句延迟。

| Batch × padded sequence | FP32 ms | INT8 ms | INT8延迟相对FP32 |
|---|---:|---:|---:|
| 1 × 16 | 0.8204 | 0.7502 | -8.56% |
| 1 × 128 | 2.7586 | 3.2866 | +19.14% |
| 1 × 256 | 4.7970 | 6.0904 | +26.96% |
| 4 × 64 | 4.4051 | 5.6322 | +27.86% |
| 16 × 64 | 16.9551 | 20.1299 | +18.72% |

长度16的正向结果被单独确认：冻结模型和线程，不再选参；换用validation第512—527行的新请求，再测8轮，并补上batch=1、固定padding16、两句真实token长度均≤16的test子集质量。性能输入仍按最长16截断/padding，质量子集则不发生截断；这是匹配配置/形状的特定负载分析，不代表所有长句任务。

| 配置 | session ms | pipeline ms | 973对短句子集Spearman |
|---|---:|---:|---:|
| FP32 / 8线程 | 0.818375 | 0.870208 | 0.847850460 |
| INT8 按通道 / 8线程 | 0.744771 | 0.782365 | 0.845246610 |

短序列纯推理加速比 1.0988×，即延迟降低 8.99%；pipeline延迟降低 10.09%。对应Spearman下降 0.002603850，差值区间 [-0.004015681841313762, -0.001282365680792185]；存在可测的质量损失。较小权重访问与该形状内核路径可能影响收益，但没有硬件计数器，不能把可能原因写成已证明的访存瓶颈。

这是看到探索性形状结果后做的追加确认，使用过的test短句子集不再称为未见数据；没有在test上再调模型或线程。[短序列确认结果](../minilm-short-request-check-v2/)。

## 真实失败与修复：分词器隐藏状态

首次短句质量筛选错误地得到0对：上游tokenizer.json自带length=128的padding和max_length=128的truncation，直接数encode后的长度不能得到真实句长。程序未拒绝空子集，SciPy返回NaN；这批质量输出无效，不能支持任何质量结论。

修复是显式关闭padding/truncation后按真实token长度筛选，并在样本不足时终止、拒绝写出非有限指标。回归测试同时覆盖“padding导致误排除”与“truncation导致误纳入”。修复后得到973对。原失败目录完整保留并标记，新的v2质量结果写入独立目录；与质量筛选无关的有效计时文件逐字节复用，哈希见 [latency-reuse.json](../minilm-short-request-check-v2/latency-reuse.json)，未伪装为重新计时。

## 执行证据与瓶颈解释的边界

最终图仍为36个常量权重MatMul量化、12个浮点MatMul，Embedding/Softmax/归一化等没有宣称全量INT8。独立10请求profiler记录360个整数相关MatMul事件，均为CPUExecutionProvider。

同INT8模型中，profiler每请求累计DynamicQuantizeMatMul事件从约 1053.2 µs 变为 915.4 µs，MatMulIntegerToFloat从约 336.9 µs 变为 270.6 µs。它支持“线程干预改变算子耗时”的观察；profile包含插桩开销，只采样10次，不替代正式计时，也没有完全解释所有INT8与FP32差异。[profile原始节点](profile/)。

记录了每轮process CPU time/wall time和RSS/峰值RSS。CPU占用可能随着并行度上升，不能从延迟降低推导功耗降低。内存是整个隔离worker，包括Python、数据、tokenizer、模型/session，不是张量专属内存或显存；本报告只宣称模型文件缩小。

## 工程验收与模拟校招审查

预设目标：同精度主负载延迟改善≥10%；test Spearman点估计下降≤0.005；文件缩小≥25%；确认真实CPU整数执行。五项机器可核验条件均通过，见 [audit.json](audit.json)。这些阈值只用于本项目，不是招聘认证。

模拟审查结论：项目材料已形成“发现问题—受控改动—复验—质量/性能取舍—可运行交付”的完整链条，足以作为中等偏上工程型校招生的一个项目素材。复杂度中等、证据完整度较强；不是新量化算法、LLM服务框架或底层kernel研究。个人面试能力仍须用户复跑、解释并修改代码来验证，不因Codex跑通而自动通过。

[审查清单](../../docs/campus-project-review.md) 与 [完整复跑和贡献说明](../../docs/runtime-study-reproduction.md)。尚未完成Jetson真机、生成式模型prefill/decode/KV cache、多卡/生产并发、SmoothQuant或自研量化内核。

## 复用与发布范围

[deployment.json](deployment.json) 固定模型/分词器哈希、CPU provider和8线程；[demo.json](demo.json) 是真实本地运行的3句向量示例，仅证明功能运行，不是额外质量基准。模型、原始数据与本地会话文件不上传。旧Windows14文件与前轮Mac70文件均保持不变。

新增唯一原始耗时记录共44,800条（v1 study 32,000 + 短序列12,800），不重复计算v2复用的记录。全部是单机相关测量，不是44,800个独立实验。完整路径/哈希索引见 delivery-manifest.json；源文件哈希对应各运行记录里的Git提交。

方法依据：[ORT线程管理](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)、[ORT量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)、[固定revision的数据文件列表](https://huggingface.co/datasets/sentence-transformers/stsb/tree/ab7a5ac0e35aa22088bdcf23e7fd99b220e53308/data)。
