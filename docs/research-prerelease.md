# Research prerelease / 研究发布说明

当前版本 `v0.1.0-research.4` 收录 PR #18 的 Residual Add + RMSNorm Metal 融合 kernel 及完整负结果。300组算子、32组模型与缓存检查逐位一致，74条公开生成序列一致；但主算子场景相对编译原生对照耗时增加约23.8%，模型也未通过加速门槛。**正确性通过，性能验收失败，默认不启用。** 发布commit、CI和附件哈希以Release的 `acceptance.json` 为准。旧research.1/.2/.3 tag与附件保留原样。

This research prerelease publishes PR #18's real Metal residual-add/RMSNorm implementation and negative optimization result. Correctness checks passed, but primary pair latency was 23.8% slower than compiled native and the model speed gate failed. The kernel stays explicitly opt-in. Publishing inspectable research does not turn failed performance gates into an accepted optimization or establish production readiness. Earlier pruning/startup benefits and all prior failures remain unchanged.

## Included optimizations / 收录的优化

| 固定研究 | Reference | Pruned | 加速比与范围 |
|---|---:|---:|---|
| 完整 INT8 热计算 | 68.681 ms | 55.342 ms | 1.241×；分词、ORT、解码、特征、排序及拒答，不含启动和 JSON I/O |
| 其中：风险特征提取 | 14.042 ms | 1.074 ms | 13.070×；包含重新解码和校验 |
| 完整 `load_service` 初始化 | 17.919 s | 5.139 s | 3.487×；完整证据核验、head 重建、资产验证与模型加载 |

热请求与启动来自两项独立、各自只运行一次的固定六进程研究，不能混用计时范围。本次打包发布没有追加性能实验。Apple M4 Max CPU、固定已有本地资产；启动使用正常 OS 文件缓存，不包含进程启动、调用前导入或请求推理。896 条已公开记录重算一致；完整启动核验不使用缓存替代重算。详细范围和原始记录见[剪枝研究](qa-risk-pruning.md)与[启动研究](qa-risk-startup.md)。

## Earlier experimental result / 已收录实验

[最后位置 prefill 研究](qwen-demand-prefill.md)保留全部层与位置 KV，只裁剪最后层未被最终输出消费的计算。M4 Max Metal GPU、固定 Qwen2.5-0.5B Q8、2048-token 主负载中，模型计算 TTFT 175.501→124.757 ms，相对完整路径 1.407×；相对最强 split-last 对照仅 1.042259×，未过 1.05× 门槛。冻结的四臂、16进程研究只运行一次。74条公开生成序列与候选/full 的80项 KV 检查一致，但 logits 有舍入差异；不是全输入逐位等价、质量提升、原创量化或新 kernel。

The experimental candidate is distributed for inspection and reproduction, not enabled in the default inference path. The independent verifier requires valid evidence and the retained `accepted=false` outcome. The one completed timing study must not be rerun as a release acceptance step.

## New Metal experiment / 新收录 Metal 实验

[Residual Add + RMSNorm融合研究](metal-residual-rmsnorm.md)在M4 Max GPU实际执行自定义Metal源码，保留两处FP16舍入，并返回残差和归一化输出。固定三臂、九新进程研究只运行一次：2048×896 FP16主场景的原生/编译原生/编译Metal分别为23.806/22.835/28.277微秒；相对两个对照的速度为0.841877×/0.807530×，五项速度/轮次门槛失败。2048-token模型TTFT约128.5毫秒，基本持平。计时含主机提交与同步，编译和加载除外；不是硬件事件时间、stock生成器或新QA质量证明。

All 300 primitive cases, 32 model/cache records, 74 consumed public greedy sequences and 54 timed model request sequences retained equality. Native and compiled-native controls share the same guarded head-only graph; the previous final-query candidate is not stacked into this study. SIMD reduction/rounding follows attributed MLX v0.29.3 MIT code. The project contributes the fusion, integration and evidence, not a novel RMSNorm or low-bit matmul algorithm. [Raw evidence and full limits](metal-residual-rmsnorm.md) remain published with `accepted=false`.

## Acceptance scope / 验收范围

- 精确发布 commit 的 Python 3.11/3.12 CI：历史证据、RC4 运行时策略、第三特征与覆盖率实验、扩大训练、非线性与新增特征重算。
- Metal三臂实验的独立离线核验：源码/协议/运行库/原始时间绑定，300/32/74一致性和五项性能失败保留；Linux CI不执行Metal。
- Qwen 四臂实验的独立离线核验，重算指标、逐 token 时间与冻结门槛；`evidence_valid=true` 与 `accepted=false` 同时成立。
- 新增剪枝源码/证据身份、完整 pruned 证据重算与独立启动算术；存档 manifest、源码快照、实际功能输出和计时组件均须一致。
- 本地测试、无 Git ZIP 逐文件清单与存档 head 重建；research.2 已完成七条真实功能/故障路径，本版本的 QA runtime 源码未变。新输出单独保存，不覆盖历史证据。
- 质量结论保持独立：历史 27/27 接受正确的有限点门槛成立；后续开发候选仍失败。量化非劣和生产性能未经证明。
- PR #14 仍可审阅：PCA 路径修复后，0.8 点覆盖 44.79% / precision 84%；Linux 重建最大概率差 1.11712e-5 超过 1e-6。九阈值决策相同不等于分数复现通过。

## Obtain and verify / 获取与验证

从 [Release](https://github.com/kimzclandi/inference-compression-lab/releases/tag/v0.1.0-research.4) 下载自定义 `inference-compression-lab-v0.1.0-research.4.zip`、`SHA256SUMS.txt`、`acceptance.json`。GitHub 自动生成的 Source code ZIP 没有本项目逐文件清单，与自定义包不是同一个文件。

```bash
shasum -a 256 -c SHA256SUMS.txt  # Linux 可用 sha256sum -c
unzip inference-compression-lab-v0.1.0-research.4.zip -d research-source
cd research-source
python3 -m experiments.release_archive verify ../inference-compression-lab-v0.1.0-research.4.zip
python3 -m venv .venv
.venv/bin/python -m pip install -r configs/qa-nonlinear/requirements.txt
.venv/bin/python -m experiments.verify_release --require-license
.venv/bin/python -m experiments.verify_qa_risk_pruning
.venv/bin/python -m experiments.verify_qa_risk_startup
.venv/bin/python -m experiments.verify_qwen_demand_prefill \
  --audit-root results/qwen-demand-prefill-v1/audit \
  --benchmark-root results/qwen-demand-prefill-v1/benchmark
.venv/bin/python -m experiments.verify_metal_residual_rmsnorm \
  --audit-root results/metal-residual-rmsnorm-v1/audit \
  --benchmark-root results/metal-residual-rmsnorm-v1/benchmark
.venv/bin/python -m experiments.verify_qa_risk --root results/qa-risk-v2 --feature-mode pruned
.venv/bin/python -m experiments.review_qa_evidence --output-dir runs/review
.venv/bin/python -m experiments.verify_span_gap_ablation --output runs/span-gap-audit.json
.venv/bin/python -m experiments.verify_qa_coverage_gap --output runs/coverage-gap-audit.json
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.verify_qa_expanded --replay-fit --output runs/expanded-audit.json
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.qa_nonlinear verify --folder results/qa-nonlinear-v2 --output runs/nonlinear-audit.json
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.qa_rich verify --folder results/qa-rich-v1 --output runs/rich-audit.json
.venv/bin/python -m experiments.verify_qa_rich_inputs --output runs/rich-input-audit.json
.venv/bin/python -m unittest discover -s tests -v
```

依赖安装后，上述证据检查不需要模型、网络或 Git。若输出已存在，请换新目录，不要覆盖。完整真实模型环境及七路径命令见 [无 Git 运行指南](release-reproduction.md)。模型下载、导出和在新硬件推理是额外工作，不包含在离线证据检查通过这一结论内。

原五特征优化器的跨平台边界不变：独立核对 raw/features/labels 后，用已验证存档矩阵重建；不是任意重新生成浮点输入的独立训练稳定性保证。语义 PCA 版本未纳入此发布。所有公开样本以后只能用于复现/消融，不能再标记为未见确认。

## Licenses and access / 许可与访问

项目代码采用维护者选定的 [MIT](../LICENSE)，公开数据保留 [数据许可](../DATA_LICENSE.md)，模型和框架见 [第三方归属](../THIRD_PARTY.md)。包不含基础模型或训练得到的头参数，也不含公司、学校私有材料。新生成参数留在本地 `runs/`。

无需受邀即可访问公开 Release、源码和 PR；实际发布状态以 GitHub 页面为准。默认分支 `codex/research-prerelease` 是此版本入口，本次继承已验收的 PR #16，并保留 PR #17、PR #18 明确未通过性能门槛的实验；默认推理路径不启用这些候选。旧 main 和 PR #14 不作为发布源。CI 绿色仅代表本页限定的技术检查，不能写成所有模型质量/部署门槛通过。
