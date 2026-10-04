# Research prerelease / 研究发布说明

本版本公开的是可复现研究代码与证据，不是覆盖率优化成功或生产可部署模型。发布基线为 PR #13 commit `fdcee0523f5e973887e0501b618479cc3d11ca15`，追加 MIT 许可、双语入口、发布说明与强制许可证存在检查；不改历史实验、模型、默认策略或质量门槛。公开 tag 为 `v0.1.0-research.1`，最终 commit 和附件哈希以 Release 为准。

This prerelease packages the technically reproducible research baseline from PR #13, with explicit licensing and a bilingual entry point. Negative results are first-class evidence. It does not promote the unresolved semantic experiment in PR #14 or establish production readiness.

## Acceptance scope / 验收范围

- 精确发布 commit 的 Python 3.11/3.12 CI：历史证据、RC4 运行时策略、第三特征与覆盖率实验、扩大训练、非线性与新增特征重算。
- 本地测试、无 Git ZIP 逐文件清单与存档 head 重建；使用已有固定本地资产复跑七条真实功能/故障路径。新输出单独保存，不覆盖历史证据。
- 质量结论保持独立：历史 27/27 接受正确的有限点门槛成立；后续开发候选仍失败。量化非劣和生产性能未经证明。
- PR #14 仍可审阅：PCA 路径修复后，0.8 点覆盖 44.79% / precision 84%；Linux 重建最大概率差 1.11712e-5 超过 1e-6。九阈值决策相同不等于分数复现通过。

## Obtain and verify / 获取与验证

从 [Release](https://github.com/kimzclandi/inference-compression-lab/releases/tag/v0.1.0-research.1) 下载自定义 `inference-compression-lab-v0.1.0-research.1.zip`、`SHA256SUMS.txt`、`acceptance.json`。GitHub 自动生成的 Source code ZIP 没有本项目逐文件清单，与自定义包不是同一个文件。

```bash
shasum -a 256 -c SHA256SUMS.txt  # Linux 可用 sha256sum -c
unzip inference-compression-lab-v0.1.0-research.1.zip -d research-source
cd research-source
python3 -m experiments.release_archive verify ../inference-compression-lab-v0.1.0-research.1.zip
python3 -m venv .venv
.venv/bin/python -m pip install -r configs/qa-nonlinear/requirements.txt
.venv/bin/python -m experiments.verify_release --require-license
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

公开后无需受邀即可访问 Release、源码和 PR；实际发布状态须以 GitHub 页面为准。发布分支 `codex/research-prerelease` 是此版本入口，旧 main 未合并叠加 PR。设置发布入口不代表自动合并历史 PR。CI 绿色仅代表本页限定的技术检查，不能写成所有模型质量/部署门槛通过。
