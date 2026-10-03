# 独立性能与招聘价值审查 / Independent performance and portfolio review

日期：2026-10-04（Asia/Singapore）。范围：MiniLM 动态 INT8、误差/回退、线程与分桶；Qwen 前缀缓存、公平基线、生命周期及量化性能。此审查未运行模型，避免干扰主审查的实际推理。所有耗时与资源数字均为历史原始记录的本轮离线重算，不是本轮重新测量。

结论：已审查部分足以支持有边界的、可复核的个人推理优化研究项目；部分固定负载确有性能收益，不能据此认定问答部署质量、原创量化算法、全部 AI Infra 岗位要求或用户个人熟练度。性能审查没有发现会推翻已报数字的计时或分母缺陷。根许可证、发布范围和整包验收由主审查另行决定。

English: The reviewed evidence supports a bounded, reproducible personal inference optimization study. Several fixed workloads show measured improvements, but this is not deployment-quality QA, a new quantization algorithm, proof of all AI Infra competencies, or proof of the owner's independent mastery. This review recomputes archived measurements; it does not rerun performance benchmarks.

## 独立核查方法 / Independent method

`recompute.py` 仅用 Python 标准库，不调用项目实现、评分器或 verifier。它从逐轮耗时、逐题 cosine/label、逐请求 token/计时、逐层误差充分统计量和 profiler 节点记录重算；采用自行实现的含并列平均排名的 Spearman、独立 LRU 列表模拟、`N-1` decode 间隔和逐请求阶段对账。`recomputed.json` 记录所有读取文件的 SHA256。它不证明历史机器确实执行过每一步，历史执行可信度还依赖源码快照、运行记录与主审查复跑。

```bash
python3 results/independent-review-v1/performance/recompute.py
```

本轮核对包括：960 行 MiniLM 误差统计/120 个分组；两个 MiniLM 确认目录的 128 份原始文件/25,600 条耗时（短序列 v2 的耗时是历史 v1 复用，不能算新增采样）；28 个分桶整语料耗时；前缀公平基线 800 条已存请求（400 主测量、400 插桩）；生命周期 480 条请求和 5 类注入故障；量化 25 个 PID/200 请求。数量仅说明检查范围，不能作为质量或算法创新指标。

## 核查结果 / Findings

|项目|从原始记录得到的结果|可以支持 / Cannot imply|
|---|---|---|
|MiniLM 动态 INT8|原始 profiler 节点记录含 36 个整数相关 MatMul 执行，单节点回退后 35 个；仍保留浮点 MatMul|真实 ORT 动态量化执行；不支持“全模型 INT8”或自研整数内核|
|局部误差|按存储 SSE/energy 重算，最大 local NMSE 为 layer.4/intermediate/dense/MatMul，0.00037537509037|可复核的同输入局部误差诊断；局部误差不是任务敏感性或唯一因果解释|
|线程|同 INT8、B=1/S=64，4→8 线程 session 2.260927→1.932885 ms，降低 14.5092%；含分词/池化路径 2.33252075→1.99645825 ms，降低 14.4077%|固定模型运行配置优化；调优 INT8 仍比调优 FP32 session 慢 11.7154%|
|分桶 FP32|2,758 句/345 批，1.252344875→1.044400417 s，降低 16.6044%；1,379 对 Spearman 均为 0.8203013947512559|相同精度、离线整语料路径收益；不能推导线上等待/P99|
|分桶 INT8|1.473085416→1.199769250 s，降低 18.5540%；Spearman 0.8188255026→0.8187232183|分组可改变动态量化数值，仍慢于 FP32；不能写“精度完全不变”|
|Padding|总槽位 54,416→42,072，下降 22.6845%；有效 token 39,028 不变|减少 padding 工作量；B×S² 仅是尺寸代理，不是硬件 FLOPs 实测|
|Qwen 公平缓存对照|2,048 前缀/4 请求/每次 32 token，含第一次 miss：直接分段 1.003265667→缓存 0.630391834 s，下降 37.1660%；另一固定合成负载 39.2383%|本机串行重复前缀工作流；不是生产命中率/服务吞吐|
|Decode|历史 2,048 前缀直接分段 268.7984 vs warm hit 268.0927 token/s|未见 decode 改善；TTFT 收益不能写成完整生成同幅加速|
|容量|1,024 前缀、[A,B,C]×4、每次 8 token：直接 1.121055917→三条目 0.548302708 s（51.0905%）；两条目/字节只容两条均 0 hit/12 miss/10 eviction|固定访问局部性的阈值与负对照，不是“大缓存总更快”|
|异常|5 类故障前后驻留顺序/字节/成功计数不变，failures+1，重放 token 与完整前缀一致|单调用者、受控注入错误路径；不是并发/崩溃/真实 OOM 保障|
|量化性能|FP16/Q4/Q8/回退10/回退22 的 TTFT、decode、总耗时及 5 轮资源中位数均与独立重算一致|探索阶段同机八固定请求；不证明确认集实际性能或大模型泛化|

MiniLM 线程质量独立重算：1379 对 FP32 Spearman 0.8203013947512559，INT8 0.8187532234983723，下降 0.0015481712528836；973 对短句 FP32 0.847850460366531，INT8 0.8452466102387562，下降 0.0026038501277748。短句是形状探索后的追加分析，不是重新发现的全新盲测。

Qwen 五变体量化性能的跨五进程中位数：

|变体|TTFT ms|decode token/s|mean generation s|采样 RSS bytes|MLX peak active bytes|
|---|---:|---:|---:|---:|---:|
|FP16|26.112537|270.310506|0.140795464|1301446656|1372993324|
|Q4|21.683281|395.091939|0.100012896|595968000|670450472|
|Q8|21.597489|318.247330|0.119001422|841924608|902788880|
|Q4 + block10 FP16|21.830198|387.687055|0.101590109|614318080|691243816|
|Q4 + block22 FP16|21.672875|387.298862|0.101714422|617267200|691735336|

这些计时从准备好的 token/input 和请求 KV 开始，模型加载、分词、输出文本解码、网络/排队不计入。Qwen 每个 token 有 `mx.eval` 和 `synchronize`；TTFT 结束于首 token 完成，32 输出 token 的 decode 只有 31 个间隔。前缀工作流外层 wall timer 含各请求和 Python 返回开销；插桩诊断的额外同步与主测量分开。MiniLM ORT CPU session.run 是同步路径。

RSS 每 10 ms 采样，含加载和预热，可能漏掉更短峰值；MLX peak active、结束时 allocator cache、RSS、逻辑前缀 KV 和权重文件大小不是同一指标，不能相加、互换或声称同幅下降。KV 公式为 `2 × 24 × 2 × L × 64 × 2 = 12288L` bytes，仅指驻留快照的有效 K/V 张量；临时请求副本及分配器容量不受这个预算限制。量化 RSS 采样线程的成本也在运行期间存在，因此小幅性能差异应保持描述性。未测功耗、温度、能耗或硬件性能计数器。

## 需要修订或降级之处 / Corrections and scope

1. 当前性能代码未发现需立即改动的实质错误。若无具体风险，不宜重新长跑追求更好结果。
2. `docs/qwen-quantization-study.md` 的“没有独立确认/下一步确认”属于探索轮历史叙事。单独阅读会与后续已完成且失败的确认相冲突，应在当前入口显著标注历史阶段并指向新确认结论；不要修改冻结原始结果。
3. `verify_qwen_prefix_fair.verify` 允许 complete 缺失，并遍历可为空的 checksum 字典，因为实验生成 complete 前也调用它。它检查结构，但不能独自证明清单完整性。本轮另强制完整覆盖全部顶层 JSON 并核对哈希：真实归档通过。正式发布应由包级完整性验证覆盖此 legacy 弱点。
4. 历史文档中的“任务待做”“未完成新环境”等应按日期理解。新发布入口应提供当前状态矩阵，保留历史报告但避免使用旧段落作为最新结论。
5. 不建议把旧第一轮“相对全 prompt 降低 47.5%”、fair 分段基线 37.2%、生命周期 51.1% 混成一个性能标题。它们的输入长度、输出长度、请求数和分母不同。

## 招聘判断 / Hiring assessment

最值得保留的两条证据是：

1. **低比特诊断到确认失败的完整决策链**：原始 FP16 张量回退、24 block 干预、预设等成本对照、探索与确认分离。真正价值在能识别“dev 改善不等于确认成功”、数值局部指标不等于完整答案、压缩/质量/性能之间的取舍。本性能审查只复核资源/耗时部分；确认评分结论采用主审查独立核验。负结果不能替代新方法、可泛化质量提升或论文研究贡献。
2. **生成式缓存的公平基线与失败处理**：直接分段重算避免以不必要快照作为唯一基线，拆分 TTFT/decode，记录首次 miss，容量预测与异常提交顺序有真实原始对照。它比“调用缓存 API”多了归因、状态约束和反例；仍是小型单调用者原型。

MiniLM 线程和分桶可合并为一个补充证据，不必另列第三个重复项目。它提供清楚的正向工程结果，但算法本身常规、模型小、负载离线。简历应删去测试总数、证据文件数、文档页数、多个 bootstrap 数字和多个相似速度百分比；这些是质量过程记录，不是招聘主贡献。

框架提供：ORT 动态量化与图融合、MLX 量化层/greedy 执行/浮点 KV/Metal 算子、Qwen/MiniLM 模型。项目贡献：受控 runtime、分桶/顺序恢复、缓存状态与错误路径、实验协议、模块回退与诊断、失败判据及证据核验。没有证据支持自研 CUDA/Triton kernel、FlashAttention 实现、分布式训练、多 GPU、token pruning、稀疏算法、蒸馏训练、QAT、量化 KV、云端服务、生产 P99、手机/Jetson/Ascend 实测或本人的独立实现能力。

定位：可作为 2027 校招 AI Infra/推理优化的**个人核心项目候选**，前提是用户能独立解释和修改至少一个关键模块；对强调原创算法/训练加速的研究岗，更适合作为补充工程证据。目前不能直接写成“熟练独立研发”或替代学校/字节实习成果。

## 简历映射 / Candidate wording

|原表述|候选改写|代码和原始证据|理由|本人待核实|
|---|---|---|---|---|
|无当前简历原文|在 AI 辅助下完成 Qwen2.5-0.5B/MLX 量化诊断与受控 FP16 模块回退，区分开发集选层与预先冻结的新文章确认；保留确认未达门槛的结果，并分析质量、权重大小与推理资源取舍。|qwen_quantization.py；qwen-quantization-v1；qwen-confirmation-v1；主审查张量/评分复验|强调可审计研究判断，避免写成量化算法成功或通用质量提升|能解释量化布局、FP16来源、选层偏差、bootstrap与门槛|
|无当前简历原文|实现精确 token 前缀 KV 复用、容量管理和失败恢复，对比无快照分段重算；M4 Max、2,048-token 前缀、每次生成32 token、4请求且计入首次miss，5轮中位耗时1.003→0.630秒（降低37.2%），保留decode无明显改善及容量不足零命中的负对照。|lab/prefix_cache.py；lab/qwen_prefix.py；qwen-prefix-fair-v1/timings.json；qwen-cache-lifecycle-gpu-v1|量化上下文与分母完整，能被原始token/时间复核|用户必须自己复跑并说明同步、缓存隔离、LRU/字节预算与边界|
|无当前简历原文|优化 MiniLM CPU 离线批处理，固定 ORT 8线程/batch8，以窗口内长度排序降低padding；2,758句7轮对照中，FP32总处理耗时降低16.6%，1,379对STS-B Spearman保持一致。|lab/batching.py；lab/minilm_runtime.py；minilm-bucketing-confirm-v1|作为补充，避免把INT8较慢掩盖为量化加速|能自行恢复输出顺序，解释动态量化批依赖与吞吐/延迟区别|

English project summary: Implemented and audited a small-model inference study with controlled quantization fallback, exact-prefix KV reuse, and CPU batching. Retained a failed preregistered confirmation, separated first-token from decode timing, and validated bounded workload gains against matched baselines. Framework kernels and quantization algorithms are upstream; contributions are experimental controls, runtime state handling, diagnostics, and reproducible evidence.

## 面试案例与亲自任务 / Interview case and owner task

最小现场案例：按 `[A,B,C]×4` 画出两条目 LRU。先预测 0 hit/12 miss/10 eviction；改为三条目且字节足够预测 9/3/0；再把字节预算限制为两个前缀，预测重新变为 0/12/10。展示原始 trace 中状态/token/耗时，并解释 51.1% 只对应 1,024 前缀/12请求/8输出 token。可离线先演示，模型复跑另开新目录。

用户必须亲自完成：在新协议副本中把 trace 改为 `[A,A,B,B,C,C]×2`，保持两条目和足够字节，**运行前写下预测**。随后本人修改配置、调用已有入口，保存新目录并逐条检查命中、淘汰、token一致性与完整耗时；对照直接分段路径，不按结果再调参。不要先复制本报告的结论作为自己的解释。该新 trace 的结果只算已知模型/方法的复现与理解训练，不是未见确认集。本人完成前，掌握状态仍为未证明。

最可能追问：为什么只恢复 FP16 block 而非反量化？为什么同样 18/74 不代表和 Q8 一样好？为什么缓存容量增加也可能没收益？为何 31 个 decode 间隔而非 32？逻辑 KV 64 MiB 为什么不保证 RSS 小于64 MiB？已存原始记录能回答前四类实验问题，生产并发/真实 OOM/跨硬件问题只能如实说明未覆盖。
