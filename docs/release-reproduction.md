# 研究项目发布候选：复跑与验收

发布对象是可复现的个人研究代码和证据，不是已验证可部署的 QA 产品。确认实验未通过质量门槛；这个负结果不阻止研究材料交付，但必须保留在首页和报告。

仓库保持私有；main 尚未合并 PR #1–#6。独立审查发布候选在 `codex/independent-release-review`，不自动合并、不创建公开 Release、不改变可见性。根代码许可证等待维护者选择；第三方原有许可与数据声明已经整理。`--require-license` 会阻止在尚未选择根许可证时宣称开源授权已完备。

## 零模型、零网络验收

Python 3.11+；在解压后的仓库根目录运行。源码包含所有协议与冻结数据，不需要作者其他仓库。

```bash
python3 -B -m experiments.verify_release
python3 -B -m experiments.qwen_quantization_demo
python3 -B -m unittest discover -s tests -v
```

标准库模式会明确跳过需要 NumPy/ORT/原始本地模型的测试；这不意味着这些 GPU/ORT 路径在 CI 被执行。模型真实推理证据是 Mac 的原始记录。Linux CI 重算指标、验证合同，不宣称 Linux 执行过 Metal。

## 从本地原始模型独立重建

要求 Apple Silicon / 可用 Metal；本轮验证硬件 M4 Max 48 GiB。原始模型约 1 GB，五种导出模型合计约 2.4 GB，另预留环境/缓存空间。不要将历史环境中的 requirements 混装。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mlx-prefix.txt
```

取得 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775` 的原始 snapshot，保留原 LICENSE/README。已有文件可直接复用；需要下载时可显式执行下面命令，仅下载公开原始权重，不调用推理 API：

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", local_dir="runs/upstream-qwen"))'
```

准备输出目录的父目录，子目录必须全新；不会覆盖旧实验。下面代码全程离线读取本地模型：

```bash
.venv/bin/python -m experiments.prepare_qwen_quantization_assets \
  --source runs/upstream-qwen --output-dir runs/rebuilt-models \
  --evidence-dir runs/model-rebuild-check
.venv/bin/python -m experiments.qwen_confirmation \
  --model-root runs/rebuilt-models --output-dir runs/confirmation-replay
.venv/bin/python -m experiments.verify_qwen_confirmation runs/confirmation-replay --write-summary
```

这是对已公布数据的复跑，不是第二个独立确认集。不能看到结果后修改 blocks、prompt 或数据，再沿用“预注册确认”的名称。每次复跑保留新目录及版本，其他硬件/软件出现 token 差异必须记录。

## 重建数据与检查原始来源

只需 Python 标准库。获取官方原文件后，脚本先核对 SHA256；不需要访问作者旧项目。

```bash
curl -L --fail https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -o runs/squad-dev-v2.0.json
python3 -m experiments.prepare_qwen_confirmation \
  --raw runs/squad-dev-v2.0.json --output-dir runs/reselected-confirmation
cmp runs/reselected-confirmation/data.jsonl configs/qwen-confirmation/dataset/data.jsonl
```

旧数据排除项只包含 ID、标题与规范化哈希；并不上传学校/公司材料。源文件清单说明排除集的盘点边界，不能证明作者设备上不存在其他未盘点材料。

## 打包与转移后验收

全部改动提交且 Git 工作区干净后：

```bash
python3 -m experiments.release_archive build /tmp/inference-compression-lab-rc.zip
python3 -m experiments.release_archive verify /tmp/inference-compression-lab-rc.zip
```

包只来自确定的 Git commit；清单记录每个文件 SHA256，拒绝权重、Git 元数据、本地缓存及常见密钥特征。此检查不能替代人工审查敏感材料。解压到新目录后重跑本页第一组命令；清单本身不构成数字签名，外部发布的包 SHA256 应独立保存。

## 发布验收边界

必须满足：历史证据字节保留、原始记录重算一致、五模型独立重建身份一致、无 Git 真实确认执行完成、固定协议与负结果公开可见、源码包转移后可验、相关 CI 通过、许可和数据归属清晰。可复用代码许可证须维护者确认；公开可见性与 PR 合并不在自动执行范围。

不在本次验收内：机器人接入、CUDA/TensorRT/手机/昇腾实测、并发生产服务、GPU OOM 验证、功耗测量、独立多设备重复、质量达标、原创量化算法，以及用户本人已经掌握代码。

## 独立审查的小规模真实复现

独立 runner 不导入本项目的评分、生成或身份函数。它直接解析 safetensors，逐张量核对上游 BF16 到 FP16 转换及回退来源，然后复跑按文章和可回答性确定的 8 个已公布问题 × 5 变体。使用已安装的同版本环境与明确传入的权重路径；不声称本轮重新安装环境或进行了异机复现。

```bash
.venv/bin/python -B -m experiments.independent_qwen_replay \
  --upstream runs/upstream-qwen --model-root runs/rebuilt-models \
  --plan results/independent-review-v1/runtime/plan.json \
  --output-dir runs/independent-published-replay
```

预期为 40 条 prompt/token/top10 全部与冻结记录一致。差异会保存并报错，不能重试挑选一致结果。记录位于 `results/independent-review-v1/runtime/`；本轮在无 Git 源码目录完成运行。

ZIP 内部清单只检查自洽性，不是数字签名。转移时先校验交付的外部 SHA256，再检查包内 commit，解压后运行 `verify_release` 和测试。`verify_release` 的固定历史清单锚防止空清单缩小历史保护范围；单独的历史公平基线验证器可在 runner 完成清单前运行，因此发布必须用统一验收入口。
