# Complete startup verification / 完整启动核验优化

本项目将模型压缩质量与运行成本分开研究。本节点将已验证的精确风险特征剪枝用于完整证据核验，减少初始化期间的文本规范化开销。仍逐行重算全部特征，不读取缓存代替计算，不改变模型、head、阈值、解码或热请求实现。

This project separates model-compression quality from runtime cost. This change applies the already validated exact risk-feature pruning to complete startup verification. Every evidence feature is recomputed; no cached result replaces verification. Models, heads, thresholds, decoding and warm-request computation remain unchanged.

## Fixed study and implementation / 固定研究与实现

[协议](../configs/qa-risk-startup/study.json)在实现前由 `133e1e7dada9e5514b9ab4760e86353a8ada9fcb` 单独提交，SHA256 为 `b804da9d77dd4280cae6abdf28758a452abd190bbfe0e5fcbf8811bb8d99a0bf`。基线是 PR #16 先前的 `396d269d1c7ddb17451b32f692501c611172dfaa`；不是旧约 18.036 秒诊断记录与本轮新计时相除。

- `verify_qa_risk` 离线入口默认仍为 `reference`；新增 `pruned` 模式通过显式参数传入同一训练、校准和评估流程。
- `load_service` 默认用 `pruned` 完整核验；`verification_mode='reference'` 提供固定对照。两条路径使用相同计时与观察包装。
- 全部文件身份、父记录独立解码、768 条训练/校准与 128 条评估特征重算、存档训练矩阵核对与 head 重建、校准选择、评估指标和 gate 保留。
- `pruned` 先检查原剪枝源码与已接受证据身份；原全窗口不同规范化答案 margin、平分规则及异常处理保持不变。

独立预审发现 runner 未绑定四个实际导入的文件：`verify_qa_remediation.py`、`qa_specialist.py` 与两份包 `__init__.py`。计时前补齐为 21 个源码文件，在全新目录重做完整 audit；旧 audit 原目录保留，没有用旧 audit 授权新源码。协议、输入、计时器和性能门槛均未改。

新 audit 的调用计数为 reference 路径 `896/0`、pruned 路径 `0/896`；两份完整公开结果 canonical JSON SHA256 均为 `e02004ef46010646181e31aa113d5ef4fb279cabf1b052a520d353c4fc0b48b1`。

## One measured comparison / 唯一一次计时

Apple M4 Max、48 GB；Python 3.12.11；NumPy 2.2.6；ORT 1.22.1 CPUExecutionProvider、intra-op=4/inter-op=1；Transformers 4.56.2、tokenizers 0.22.0。使用原有固定合法本地资产，离线执行，无下载。

六个新进程按 `reference/pruned, pruned/reference, reference/pruned` 顺序执行，总计算预算 900 秒。没有 worker 预跑、输入筛选或第二次计时研究。主要门槛为完整初始化中位数加速至少 1.5×，且至少两对更快；结果与三例公开功能响应必须一致。

| 范围 | Reference 中位数 | Pruned 中位数 | 加速比 | 三对加速比 |
|---|---:|---:|---:|---|
| 完整 `load_service` 初始化 | 17.919445 s | 5.139233 s | **3.486793×** | 3.457030 / 3.494151 / 3.442823 |
| 其中：策略及完整证据核验 | 16.328515 s | 3.572126 s | 4.571091× | 4.533114 / 4.581732 / 4.510608 |

完整初始化耗时下降约 **71.32%**。Reference 三次总时长为 17.766484、17.919445、17.956015 秒；pruned 为 5.139233、5.128412、5.215491 秒。所有门槛一次通过，三对均更快。

计时从 `load_service` 函数入口开始，到策略/完整证据核验、256 行训练 head 重建、合法本地资产验证与模型加载完成为止。函数内部延迟导入计入相应阶段；不计 Python 进程启动、调用前导入、请求推理与 JSON I/O。正常 OS 文件缓存，不声称冷磁盘启动。三例请求仅做功能核验，未另报热请求速度。

The fixed startup gate passed once: median complete initialization fell from 17.919445 to 5.139233 seconds, a 3.486793× speedup. All six complete verification outputs and the three disclosed responses were identical. The result concerns fresh-process `load_service` initialization with ordinary filesystem caching; it does not measure cold-disk launch, request latency, Linux inference, concurrent serving or tail latency.

## Evidence and acceptance / 证据与验收

[完整 audit](../results/qa-risk-startup-v1/audit)、[六进程原始计时](../results/qa-risk-startup-v1/benchmark)、[预审记录](../results/qa-risk-startup-v1/review.json)保留原字节。各进程包含协议、运行状态、21 文件源码快照、完整核验结果、功能输出、依赖版本与校验和；不包含模型或 learned head/scaler 参数。

[独立核验器](../experiments/verify_qa_risk_startup.py)不调用 runner 的汇总函数、不运行模型：固定审查后的 manifest 与源码身份，从六份原始 measurement 独立重算组件和、各臂中位数、配对加速、输出摘要和门槛。当前源码、归档源码、实际响应 JSON 与历史功能分数也须一致。SHA 锚依赖审查代码，不能抵抗连同 verifier 一起修改的发布者。

默认入口的七条功能/错误路径以及实际 CLI 的正常回答、缺失模型、剪枝源码漂移、重新生成 checksum 的训练特征篡改均另做验收，见[验收记录](../results/qa-risk-startup-v1/validation.json)。这些运行的启动时间仅是诊断，未并入上述性能统计。

完整 RC4 验收仍实际运行原 reference 证据重算，随后检查本节点存档与当前源码；CI 另运行完整 pruned 重算和独立启动算术。Linux CI 通过只说明代码/证据合同在该环境通过，不建立 Linux 模型启动性能。

```bash
# Existing fixed offline-verification environment; no model/Git required.
python -m experiments.verify_qa_risk_startup
python -m experiments.verify_release_rc4 --require-license
python -m unittest discover -s tests -p 'test_qa_risk_startup*.py' -v

# Full pruned recomputation, not cached acceptance.
python -m experiments.verify_qa_risk --root results/qa-risk-v2 --feature-mode pruned
```

本节点的计时研究已完成，交付过程不再运行 benchmark。需要复核数值时运行上面的存档核验器。所有源码包和最终提交/CI 身份记入外部交付报告，以免收据自包含其自身哈希。

## Unchanged quality and publication limits / 保留边界

之前热请求特征 13.070×、完整热计算 1.241×来自另一固定研究，不能与本轮启动加速相乘或混用。固定 INT8 仍接受 27/128、27 个 EM 正确；可回答覆盖 27/64=42.19%，不可回答误接受 0/64，system EM/F1=71.09%；27/27 的 Wilson95% 下界约 87.54%，不是总体 90% 保证。

Civil_disobedience 整篇拒答；37 个被拒可回答问题中 19 个 raw 答案原本正确。Qwen block-10 回退确认失败、专用 QA v1 量化非劣失败、扩展训练/树模型未通过联合门槛、未发布语义/PCA 质量与 Linux 数值问题均保留。本次没有改善质量或覆盖。公开样本已经消耗，属于复现和机制/工程优化；存档矩阵重建不证明任意浮点重生成训练稳定。

上游提供模型、量化框架和 kernel，本项目贡献是实验、诊断、精确后处理及证据实现，实施与执行有 AI 辅助。无原创低比特 kernel、手机/GPU/NPU 或生产部署声明。个人掌握不属于本工程节点的前置条件或验收结论。

交付为现有草稿 PR #16；不合并、不改变可见性、不创建 tag/Release。默认分支及已有公开研究 prerelease `v0.1.0-research.1` 仍指向 `65ed839d8f018bacc7fbdd1792782a9367bcb016`，不含本优化。
