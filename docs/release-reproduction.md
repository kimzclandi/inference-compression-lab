# 研究项目发布候选：复跑与验收

发布对象是可复现的个人研究代码和证据，不是已验证可部署的 QA 产品。确认实验未通过质量门槛；这个负结果不阻止研究材料交付，但必须保留在首页和报告。

当前 RC4 候选在 `codex/qa-extractive-specialist`（草稿 PR #9，基于 #8）；核验时仓库为 PRIVATE，PR #1–#9 未合并，默认 main 不含完整叠加成果。请使用明确 commit 的源码包或 PR 分支。根代码许可证和发布可见性仍待维护者决定，MIT 候选不代表已采用。第三方许可与数据声明单独保留。`--require-license` 只检查根 LICENSE 文件存在，不能替代维护者授权或法律审查。

## RC4：从无 Git 源码包开始 / Start from the source archive

先用发布者单独提供的 `SHA256SUMS.txt` 核对 ZIP（macOS 用 `shasum -a 256`，Linux 用 `sha256sum`），再解压到一个新目录。在解压根目录执行：

```bash
python3 -m experiments.release_archive verify /absolute/path/to/inference-compression-lab-rc4.zip
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.2.6
.venv/bin/python -B -m experiments.verify_release_rc4
.venv/bin/python -B -m unittest discover -s tests -v
```

需要可读的精简双语报告时，可用 `python -m experiments.review_qa_evidence --output-dir runs/my-evidence-review` 替代上面的 `verify_release_rc4` 命令；它调用同一完整验收，不必重复执行两遍。输出目录必须全新，失败记录会保留。系统与原始记录导航见 [技术导览](qa-system-overview.md)。

安装依赖需要网络或预先准备的 wheel；安装完成后，完整证据验收不需要 Git、模型或网络。核对 archive verifier 输出的 commit 与交付记录；包内清单只证明自洽，外部 SHA256 才是此次交付的独立比对锚。CI 的 Python 3.11/3.12 仅安装 NumPy；依赖 ORT/MLX 等的测试可能跳过，必须查看实际 skip 原因。不要将这种验收写成跨平台真实推理成功。

**数值重建边界：** 验证器先从 raw 独立核对全部特征和标签（既有数值容差 `1e-12`），再用已核验的原存档训练矩阵重放固定优化器，与运行时 `rebuild_head` 一致。Linux 的 `log/log1p` 重算曾产生约 `4.44e-16` 差异，让从新矩阵出发的训练在原 `1e-8` 梯度门槛附近停滞。没有放宽收敛或质量门槛。存档矩阵重建通过，不等于任意环境重新生成浮点输入并训练均稳定；`experiments.qa_risk train` 是后者，应保留失败记录，不能通过调参追求相同结果。

The verifier checks raw-derived features and labels before replaying the verified serialized training matrix. This establishes archived-input head reconstruction, not platform-independent convergence for freshly regenerated floating-point inputs. Published samples support reproduction and ablation only; they are not new confirmation data.

## RC4：七条真实功能与故障路径 / Local inference exercise

完整模型环境与本地资产准备见 [专用模型指南](qa-specialist-study.md#model-quantization-and-provenance--模型量化与来源) 及该页末尾的固定 revision 下载/导出命令。使用独立 Python 3.12 环境：

```bash
python3.12 -m venv .venv-qa
.venv-qa/bin/python -m pip install -r requirements-qa-specialist.lock.txt
.venv-qa/bin/python -m pip check
# 用 .venv-qa/bin/python 执行专用模型指南的下载和资产构建命令。
.venv-qa/bin/python -B -m experiments.reproduce_qa_prototype \
  --asset-root /absolute/path/to/local-qa-assets \
  --output-dir runs/my-rc4-reproduction
```

`--asset-root` 必须是显式本地普通文件目录，包含固定 FP32/INT8 ONNX、tokenizer、manifest 与上游模型卡；运行时会校验身份和固定依赖。发布包不包含这些资产，也不包含训练 head 的系数、截距或 scaler。离线演示使用已有本地资产；新的下载、导出与在不同硬件上的推理是额外工作，不能据本包离线验收推定已完成。锁文件记录原模型环境；全仓库可选 MiniLM/MLX 测试不全属于这份锁文件的安装范围。

输出应为 `status=complete`，涵盖：正确接受、不可回答拒答、可回答拒答、额外 gold 字段、超 question token 上限、空 context、缺失资产。无 Git 解压目录的 `git_head` 应为 null。已公布后选择的三个演示样本不代表新评估。输出目录必须不存在；不覆盖历史结果。

本地请求例子（恰好两个字段）：

```json
{"context":"Alpha is a city. Beta is a river.","question":"Which place is a city?"}
```

保存为 `request.json`，运行 `.venv-qa/bin/python -m experiments.serve_qa_specialist --asset-root /absolute/path/to/local-qa-assets --input-json request.json`。这只是输入格式示例，不保证被接受。CLI 每次启动均进行证据核验与加载；热请求约 69.600 ms 和历史首次初始化约 18.04 s 属于不同计时范围。

## RC4：发布和访问 / Release and access

发布包、tag 和 Release 必须绑定同一个验收 commit；不能从缺少叠加成果的 main 打包。任何许可证修改都先提交，再重建包、核对历史字节和对应 HEAD 的 CI。GitHub 自动生成的 source ZIP 不等同于本项目带逐文件清单的自定义 ZIP；交付使用附带外部 SHA256 的自定义附件。

保持 PRIVATE 时，Release、PR 和仓库链接仅供获准账户访问；读者需要由所有者授权访问，或由所有者单独分享经过审阅的无权重源码证据包及说明。当前流程不邀请他人、不主动发送材料，也不承诺私有链接公开可读。公开展示需另行明确授权改变可见性。根许可证的选择不改变第三方数据/模型的原有归属。

A private release is not a public portfolio. Recruiters need owner-approved repository access or an independently shared source/evidence package. No merge, visibility change, tag or release is implied by successful verification. Owner decisions on the root code license and distribution visibility remain required.

## 历史 RC2/RC3：零模型、零网络验收

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

不在本次验收内：机器人接入、CUDA/TensorRT/手机/昇腾实测、并发生产服务、GPU OOM 验证、功耗测量、独立多设备重复、质量达标、原创量化算法。

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

## RC3 QA 整改补充

后续候选在 `codex/qa-quality-remediation`，追加了[QA整改报告](qa-remediation-study.md)。统一 `verify_release` 已纳入新校准完整性和关闭策略检查；技术验收通过时仍明确 `qa_remediation_quality_passed=false`。发布包必须另外运行新 `verify_qa_remediation`、独立统计脚本与入口关闭演示。新校准使用现有环境1.5B原始模型；未重新安装环境、未运行新确认集。旧0.5B五变体和本轮1.5B数据/指标不得混合。
