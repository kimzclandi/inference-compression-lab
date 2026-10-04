# Qwen Q4 首 token 诊断与等成本 FP16 block 回退

本文记录 2026-10-04 早期探索阶段。后续固定 128 题确认已完成且未通过门槛，见 [确认报告](qwen-confirmation-study.md) 与 [独立审查](independent-release-review.md)。以下原始探索数字保持不变。

2026-10-04，个人项目，探索阶段实际执行；代码与实验由 Codex 辅助完成。

本轮完成了“历史失败 → 固定假设 → 逐块干预 → 原始 FP16 回退 → 质量/成本复验”闭环。0-based block 10 回退使重复使用的 dev 集归一化 EM 从 Q4 的 16/74 到 18/74，等成本 block 22 为 16/74。该探索阶段尚无独立确认结果；后续确认未通过，仍不声称找到唯一根因、通用敏感层或新量化算法。

## 当前状态与证据层级

- 远端现场核验：仓库 private，默认分支 main=`eefd28a`；PR #1–#5 均 OPEN/DRAFT，仍按原依赖链叠加，没有合并。原 checkout 为 `codex/qwen-cache-lifecycle` / `c018de8`，dirty state 为空。
- 新隔离 worktree / 分支 `codex/qwen-quantization-diagnostics` 基于 PR #5。协议与实现先提交于 `59a2c28`，模块遍历预检修正于 `c0453f2`，随后才执行模型实验。新证据运行源码为 `c0453f2`，另有逐文件源码快照和 SHA256。
- 本轮硬件实测为 Apple M4 Max / 48 GiB，Python 3.12.11、MLX 0.29.3、MLX-LM 0.26.3、Transformers 4.56.2。执行了 Metal smoke 和真实模型推理；没有 CUDA/TensorRT/昇腾/手机推理。
- 旧量化项目实际远端为 `kimzclandi/domain-qa-lab`；本轮新建来源元数据初填为更早的上游仓库名，核实 remote 后已更正并保留更正前记录。原项目文件未修改。执行原 `scripts.verify_quantization` 通过；本轮另逐题重算历史 EM、F1、format_valid，未改评分。MiniLM 分桶、Qwen 公平基线、生命周期结果只做归档重算和完整性检查，本轮不冒充重新运行其性能实验。514 个父分支结果文件保持逐字节一致。

## 发现的问题与预先固定的对照

旧材料能证明 Q4 退化，却没有同 prompt 的首 token logits、排名与干预证据，无法判断局部数值改善是否恢复完整答案。Q4 的 `format_valid=40/74` 也不能直接解释为“34 道格式错误”：该标志包含是否为原文精确子串，归一化 EM 可以正确而标志为 false。原评分逻辑保持不变。

假设：Q4 扰动 next-token 排名；恢复一个 decoder block 的原始 FP16 权重可能恢复部分首 token 和答案。假设不指定 Attention、MLP 或 outlier 为根因。

固定协议在 [study.json](../configs/qwen-quantization/study.json)：

1. 用同一原始 Qwen2.5-0.5B-Instruct，无 adapter；FP16、affine grouped Q4/Q8，group_size=64，浮点 KV。输入权重、分词器和配置全部匹配历史 SHA256；Q8 不是 FP8。
2. 先重跑完整 74 题 dev，要求 FP16/Q4/Q8 的 prompt 和生成 token 与历史完全一致，否则停止选层。三者实际全部一致。
3. 诊断集固定为历史 FP16 正确而 Q4 错误的 6 题；逐一将 24 个 decoder block 替换为原始 FP16 block，其他 block 及 tied embedding 保留 Q4。
4. 排序规则：恢复 FP16 top1 数最多优先，其次 FP16 top1/top2 这一对的 margin 偏差均值最小，再按 block index。对照固定为 `(selected+12)%24`，不按完整质量结果挑对照。
5. 选层一次后，导出 selected/control 两份本地混合模型并重新加载；每个张量与其 FP16/Q4 来源逐值一致。不能通过反量化 Q4 冒充恢复原始 FP16。
6. 五方案各一次完整 dev 质量评估；八个冻结 train prompt、强制生成 32 token、五轮 Latin rotation、每格独立进程，每进程两次 synthetic warmup。无重试挑最好结果。
7. 诊断 logits 单独执行。性能计时从已准备输入和独立 KV 之后开始，包含同步的 prefill/greedy 首 token；decode 仅计其后 31 个间隔。排除加载、分词、网络和排队。
8. RSS 每 10 ms 采样，含加载和预热；MLX peak active/cache 分开保存，统一内存计数不相加。没有功耗测量。

## 新结果

质量为原有 74 题 dev；性能为每个进程八请求统计量的跨五轮中位数。权重单位为 decimal MB；内存为 GiB。所有格点与原始记录见 [本轮证据](../results/qwen-quantization-v1)。

|方案|归一化 EM|token F1|format_valid|权重 MB|采样 RSS GiB|MLX peak active GiB|TTFT ms|decode token/s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|FP16|19/74|30.13%|71/74|988.10|1.212|1.279|26.11|270.31|
|Q4|16/74|24.44%|40/74|278.06|0.555|0.624|21.68|395.09|
|Q8|18/74|29.68%|70/74|525.05|0.784|0.841|21.60|318.25|
|Q4 + block 10 FP16|18/74|28.88%|50/74|299.50|0.572|0.644|21.83|387.69|
|Q4 + block 22 FP16|16/74|25.39%|39/74|299.50|0.575|0.644|21.67|387.30|

相对本轮 Q4，block 10 增加约 21.43 MB 文件、采样 RSS 中位数增加约 3.1%，decode 速度中位数降低约 1.9%，TTFT 中位数约增加 0.7%。这些是描述性统计；五轮区间有重叠，未证明小幅性能差异显著。FP16/Q4/Q8 的本轮 RSS 不可与旧报告跨时间相减归因，协议与运行时状态不同。

选定方案通过事先的 **dev 探索候选门槛**：相对 Q4 至少多对 2 题、相对对照至少多对 1 题，相对 FP16 少不超过 1 题、F1 下降不超过 2 个百分点，以及预定权重/RSS/TTFT/decode 约束。通过门槛不代表独立确认或可部署。

### 数值解释与反例

|方案|与 FP16 整条 token 序列不同|首 token 不同|相对 FP16 丢失/新增正确题|
|---|---:|---:|---|
|Q4|49/74|44/74|6 / 3|
|Q8|3/74|3/74|1 / 0|
|block 10 回退|44/74|35/74|4 / 3|
|block 22 回退|48/74|43/74|6 / 3|

选层诊断中 block 10 恢复 2/6 个 FP16 首 token，其余候选最高 1/6。本轮完整质量中相对 Q4 新增 2 个正确样本且未丢失原正确样本。block 22 则恢复 1 个原正确样本，同时丢失另一个，净改善为零。选层所用 6 题同时属于完整 dev，不能把这两个新增正确样本当成独立泛化证据。

Q4 首 token 翻转样本的 FP16 top1/top2 margin 从 0.015625 到 3.75，中位数 1.265625；Q8 三例的中位数为 0.03125。Q4 不能简单解释成“只在本来就很犹豫时发生小扰动”。44 个 Q4 首 token 翻转中只有 20 个满足原 FP16 top2 严格超过原 top1；其他 token 也可能成为新赢家。仅盯原 top1/top2 的相对顺序会漏掉失败。

最小案例 `56e1c0f6cd28a01900c67b2c`，输入 `[1,110]`：

|方案|首 token|本方案 top1/top2 margin|原 FP16 top1 的排名|完整答案|
|---|---|---:|---:|---|
|FP16|complex|2.609375|1|complexity classes|
|Q4|The|1.671875|9|The passage contains the answer.|
|Q8|complex|2.484375|1|complexity classes|
|block 10|complex|0.0625|1|complexity classes|
|block 22|The|1.5|7|The passage contains the answer.|

Q4 上原 FP16 top1 对 top2 的 margin 仍为正 0.03125，但另一个 token `The` 已经获胜；这是“原 top2 没超过 top1，所以输出应该不变”的反例。block 10 恢复正确输出后新 margin 仅 0.0625，也不能据此宣称鲁棒。

### 负结果与不支持的结论

- 恢复一个 block 后，仍有 35 个首 token 不同、4 个 FP16 原正确样本丢失。完整答案 EM 仅 18/74，始终拒答基线为 41/74。该任务质量仍不合格。
- 虽然 block 10 与 Q8 均为 18/74，Q8 的 F1、format_valid 和 token 稳定性更好；不能声称混合精度全面优于 Q8。
- 没有逐算子拆分、activation outlier 统计、Attention/MLP 消融，因此不定位到某个子模块或异常值原因。
- block 回退是标准干预/混合精度策略；量化层、Qwen 架构、Metal kernel 和浮点 KV 由 MLX/MLX-LM 提供。[官方 QuantizedLinear 文档](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html) 是 API 背景；本轮实现依据已安装 0.29.3/0.26.3 源码核验。
- 24 个候选筛选产生选择偏差；单一等成本对照只支持本轮局部干预效果，不证明 block 10 唯一重要或整体最优。
- 本页探索质量来自重复使用的同一个 dev；当时没有访问 test 或独立确认集，没有重新训练或修改评分。后续确认另见链接，不与本页探索合并计分。
- 所有性能是 batch=1、串行、固定八请求；不覆盖在线并发/P95、长上下文、手机、Jetson、CUDA、昇腾或功耗。

## 实现、失败记录与验收

新增 `experiments/qwen_quantization.py` 实现固定权重身份校验、真实模型诊断、24-block 干预、原始张量回退导出、独立进程性能对照；`lab/quantization_diagnostics.py` 实现选层与统计；`lab/qa_metrics.py` 保留历史评分函数；`experiments/verify_qwen_quantization.py` 不需要 MLX 即可重算；`experiments/qwen_quantization_demo.py` 提供离线展示与真实单题复跑。

首次离线校验错误地要求两份 safetensors **文件**长度严格相同，实际二者张量均为 299,428,608 bytes，容器头相差 10 bytes（67,569 / 67,559）。失败日志与初版 verifier 原文保存在 [review 目录](../results/qwen-quantization-review-v1)。修正后校验预先规定的等张量成本，仍使用各自真实文件大小参与资源门槛；未改模型、结果或门槛，也未为此重跑选择。

验收实际完成：历史三精度逐题重放；144 次干预；五种精度布局与导出张量来源；25 个完整性能格点和同方案五轮生成一致性；评分与计时独立重算；缺失干预、变更对照和缩短 benchmark 即使重新签名也被拒绝；父分支 514 个结果文件未变。最终标准库测试 64 项：52 通过、12 项旧可选数值依赖测试跳过。测试数量不代表模型质量或跨设备能力。

后续 Linux CI 揭示汇总比较的可移植性问题：原评分器的 family_macro_em 遍历 set 求和，Python 3.11 上出现 `0.23668430335096996`，归档为 `0.23668430335097002`，只差约 5.6e-17。修复 `aggregates_equal` 在汇总浮点比较中允许至多 1e-12 绝对尾差；逐题评分函数、整数计数、ID/覆盖、模型输出和冻结文件均未修改。新增反例测试拒绝一题 EM 变化、1e-9 差异、NaN 和覆盖变化。原始 CI 失败保存在 `results/qwen-quantization-ci-v1`；Python 3.12 三种 hash seed 的目标检查及本机 Python 3.14 全部通过。系统 `/usr/bin/python3` 受未接受的 Xcode license 阻止，未更改许可证或环境，以已存在的 3.12 环境完成不受阻的检查。

第二轮 CI 暴露父证据清单使用本机绝对路径，校验在原路径之外失败。已将清单键改为仓库相对路径，514 个 SHA256 保持不变；保留原清单与两次 Linux 失败日志，并在独立解压目录执行完整测试，防止本机路径掩盖交付问题。

## 复跑命令

在仓库根目录执行。只重算或演示归档数据无需模型、MLX 或网络：

```bash
python3 -m experiments.verify_qwen_quantization results/qwen-quantization-v1
python3 -m experiments.qwen_quantization_demo
python3 -m unittest discover -s tests -v
```

本机已验证的环境和模型位置：

```bash
qwen_python=/Users/yukaiyan/Documents/Codex/2026-09-18/github-gpu-api-cuda-api-1/outputs/domain-qa-lab/.venv-mlx/bin/python
qwen_models=/Users/yukaiyan/Documents/Codex/2026-09-18/github-gpu-api-cuda-api-1/outputs/domain-qa-lab/.cache/mlx

# 新目录，禁止覆盖已有 run；预计约两分钟，负载随机器状态改变。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$qwen_python" -m experiments.qwen_quantization study \
  --model-root "$qwen_models" --work-dir runs/my-qwen-quantization \
  --output-dir results/my-qwen-quantization
python3 -m experiments.verify_qwen_quantization results/my-qwen-quantization --write-summary

# 已保存本机混合模型时，只实际复跑一题。诊断演示不作为性能结果。
"$qwen_python" -m experiments.qwen_quantization_demo --live --model-root "$qwen_models" \
  --mixed-root runs/qwen-quantization-v1 --output-dir runs/my-quantization-demo
```

模型和混合权重只保存在本地 `runs/`，不上传。依赖沿用 `requirements-mlx-prefix.txt`；本轮没有重新安装依赖或从网络下载模型，不声称完成全新机器安装验收。

原计划的固定模块确认已完成并记录负结果，见 [确认报告](qwen-confirmation-study.md)。已公布样本今后只能称复现。
