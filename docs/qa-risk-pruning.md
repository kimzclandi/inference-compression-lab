# Exact risk-feature pruning / 正确性特征精确剪枝

输入为已验证的原文、各窗口 start/end logits 和原解码记录；输出仍为同一五维特征及完整竞争答案诊断。本次减少 Python 文本规范化工作，未训练模型、修改量化方案或更换第三特征定义。实现与运行由 AI 辅助，个人独立掌握尚待验收。

Input: the passage, validated window logits and decoder record. Output: the same five features and competitor diagnostics. This change reduces Python normalization work while retaining all-window competition, the frozen head and threshold. It introduces no quantizer, kernel or model-quality improvement. Implementation and execution are AI-assisted.

## Hypothesis and decision / 假设与采用门槛

基线为已发布的 `65ed839d8f018bacc7fbdd1792782a9367bcb016`。剖析表明固定输入的规范化占特征提取剖析耗时约 81%；剖析带额外开销，不能把该比例当成完整流水线的性能占比。

[协议](../configs/qa-risk-pruning/study.json)在实现之前以 `27e6644` 固定，`f1d88be` 在任何候选实验之前补齐输入哈希。只允许一个剪枝算法、一次固定计时；不改样本、模型、线程、阈值或求解器追求正结果。预算为最多 900 秒免费本地计算，无下载或基础模型/LoRA训练，仅按原矩阵重建小排序头。

采用条件：768 条开发精度记录（384题×FP32/INT8）与128条已公开评估记录的完整特征、诊断、分数和决定零差异；边界检查通过；特征提取加速至少 1.25×、完整计算流水线至少 1.05×，各自至少两对进程更快。任何门槛失败均不得采用。

## Why pruning is exact / 为什么可剪枝

设当前最佳**不同规范化答案**的 margin 为 `b`，下一合法候选的 margin 为 `m`。两者都是该候选 span 的 start+end 分数减去其自身窗口的 CLS null 分数。

- 若 `m ≤ b`，无论候选规范化后是否与已选答案相同，它都无法替换当前竞争答案；平分时原实现保留先枚举的候选。因此可以不调用文本规范化。
- 尚无不同答案时不能剪枝；高分重复答案仍需排除，并继续遍历其后的候选。
- 溢出或不能表示为有限浮点的 margin 不能按低分剪掉，必须继续原规范化/错误路径。否则会把非法请求变成成功请求。
- 全窗口、所有合法 span、选中答案、五维输出、浮点计算顺序和首次平分规则保留。最坏枚举复杂度仍为 `O(W×L×A)`，`W` 为窗口数，`L` 为每窗 token 数，`A=30` 为最大答案 token 数；减少的是规范化调用与字符串处理，未声称渐近复杂度降低。

两窗反例仍得到 `gap=1`，保留另一窗重复 Alpha 后面的 Gamma，而非变成选中窗的 `gap=8`。这与此前“仅选中窗口”消融是不同问题。真实数据无变化不能证明任意算法普遍等价；这里还需上述剪枝条件论证和边界回归。

[新实现](../lab/qa_risk_pruning.py)保留了一个明确的小型特化函数。[原实现](../lab/qa_risk_calibration.py)的字节和旧证据绑定保持不变，供历史实验复现与差分对照。采用后仅本地原型入口使用特化函数；旧复现命令仍执行原路径，研究者必须区分入口。

## Retained failure / 保留的候选缺陷

第一版通过896条历史记录回放，但极大整数 logits 反例失败：decoder先将两个整数分别转float，原特征代码却可能在规范化排除重复答案后根本不计算整数和；提前计算会新增 `OverflowError`。这不是模型框架故障。

[失败日志](../results/qa-risk-pruning-v1/integer-overflow-before.log)和[第一版回放](../results/qa-risk-pruning-v1/audit)保留。`553e7ea` 在计时前修复：提前计算失败时回到原规范化顺序，只有原路径应拒绝时才拒绝。最终17项相关测试包含250组固定种子合成情形、跨窗第二候选、相同margin、重复答案、mask/offset、Unicode、零宽token和溢出。合成样例只验证机制，不参与QA质量指标。

## Measured result / 新测结果

Apple M4 Max、48 GB；Python 3.12.11；ORT 1.22.1 CPUExecutionProvider；NumPy 2.2.6；Transformers 4.56.2；tokenizers 0.22.0。模型和资产哈希沿用原INT8协议；ORT intra-op=4、inter-op=1。

六个独立进程顺序为 reference/pruned、pruned/reference、reference/pruned。每进程8条固定单窗口输入预热一次，随后每条测5次；共240条完整请求计时及240条特征计时。统计为各臂三个进程中位数的中位数。

| 本轮计时范围 | Reference | Pruned | 加速比 | 三对进程加速比 |
|---|---:|---:|---:|---|
| 五维特征提取，含重新解码与校验 | 14.042 ms | 1.074 ms | 13.070× | 12.959 / 12.853 / 13.131 |
| 分词、ORT、解码、特征、排序、parse及拒答 | 68.681 ms | 55.342 ms | 1.241× | 1.236 / 1.233 / 1.246 |

完整计算耗时下降19.42%。规范化调用在固定8条输入上由34,933减至121（-99.65%）；最终回放896条记录零差异。历史128条仍为接受27、正确27、可回答覆盖27/64、不可回答误接受0/64、system EM/F1=71.09%。这只是旧质量结果复现。

**边界：**计时包装器沿用旧benchmark的计算范围，并非CLI端到端计时；不含JSON I/O、完整证据核验、模型加载，也不包含服务入口附加的响应封装。六进程初始化1.580–1.666秒仅含存档头重建和资产核验/模型加载，不能与历史含完整证据核验的约18.04秒混用。进程峰值RSS 895.9–960.1 MiB，含初始化；未证明内存优化。8个固定单窗口输入和同机进程不是独立业务负载样本，无长文速度、并发、p95/SLO或跨设备推广结论。

The joint gates passed once. The result is a same-host CPU postprocessing optimization with exact replay on consumed public samples. It does not establish unseen QA quality, quantization noninferiority, GPU/NPU acceleration, mobile deployment or production readiness.

历史72.052/54.188ms的FP32/INT8基础管线、69.600ms的旧完整INT8管线及其各自范围仍保留；本轮不把这些历史数据拼成新的FP32/INT8比较。Wilson下界约87.54%、Civil_disobedience整篇拒答、37个被拒可回答问题中19个raw正确、Qwen回退失败及后续覆盖候选失败均未改变。

## Reproduce / 复核

使用既有固定依赖；所有运行写入新目录。无需基础模型即可回放输入并比较两种实现：

```bash
OMP_NUM_THREADS=1 python -m experiments.qa_risk_pruning audit --output-dir runs/my-pruning-replay
python -m experiments.verify_qa_risk_pruning
python -m unittest discover -s tests -p 'test_qa_risk_pruning.py' -v
```

第二个命令独立重算已保存的240条请求和240条特征计时，验证输入、源码、记录覆盖、输出摘要与门槛；不测当前设备速度。CI在Linux Python3.11/3.12执行差分回放和存档核验，不是Linux真实性能复现。

本地已有合法模型资产时可另行复跑固定计时，不覆盖本次结果：

```bash
OMP_NUM_THREADS=1 python -m experiments.qa_risk_pruning benchmark \
  --asset-root /absolute/path/to/existing-assets --output-dir runs/my-pruning-timing
```

[最终差分记录](../results/qa-risk-pruning-v1/audit-final) · [原始计时和摘要](../results/qa-risk-pruning-v1/benchmark) · [回归测试](../tests/test_qa_risk_pruning.py)

本分支接入原型入口；已有 `v0.1.0-research.1` tag和附件保持原样。新变更仅通过草稿PR交付，未经合并、未创建新Release。底层模型、量化和kernel来自上游，本项目贡献为瓶颈分析、精确剪枝、固定对照与验收。

## Delivery review / 交付完整性复核

继续审查发现第一版核验器只检查自洽哈希，未固定已验收清单身份：删空源码绑定、伪造质量摘要、替换回放ID或改写计时数量，再重算校验和，均可能被接受。原型启动也仅核验旧排序证据，没有绑定实际新剪枝实现。[修复前样例](../results/qa-risk-pruning-readiness-v1/before-probes.json)和[原核验器](../results/qa-risk-pruning-readiness-v1/reference/verify_measurements.py)保留。

当前[核验器](../experiments/verify_qa_risk_pruning.py)固定协议、已验收manifest、完整源码集合与对应字节；独立核对回放身份/角色、质量计数和计时记录。原型在重建头及模型加载前执行这一检查；完整RC4验收也纳入它。四类篡改现在均被拒绝，另有协议改门槛、源码漂移及符号链接回归测试。

原始 audit、benchmark、计时源码快照和优化算法字节未变。当前实验runner仅增加运行前协议哈希检查；测试核对去掉该检查后的AST与已计时runner一致，因此未重新测速度。旧`results/.../verify_measurements.py`入口转发到新核验器，原实现单独存档。

These are integrity checks under reviewed code, not signatures against a publisher able to rewrite both the verifier and its anchors. They preserve the accepted experiment and reject incomplete or altered evidence. They do not prove unseen model quality, arbitrary input equivalence by testing alone, or performance on another device.

复核日志、无Git新环境检查、七条真实路径摘要及源码漂移拒绝见 `results/qa-risk-pruning-readiness-v1/verification.json`。可从仓库根目录执行 `python results/qa-risk-pruning-readiness-v1/reproduce_reference_gaps.py`，在临时副本复现旧核验器的四次误接受；执行 `python -m unittest discover -s tests -p test_qa_risk_pruning_acceptance.py -v` 验证修复后的拒绝行为。无需模型，不改历史记录。
