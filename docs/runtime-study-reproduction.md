# MiniLM CPU 推理优化成果：复跑与贡献说明

这是一项当前个人项目的工程实验，不属于未经核实的 NUSRI 历史工作，也不是 Jetson 测试。研究问题是：在量化精度策略不变时，针对实际 CPU 负载调节 intra-op 并行度，能否降低推理与应用路径延迟，同时保留压缩后的文件大小和任务质量？

## 可运行交付物

- `lab/minilm_runtime.py`：可复用 CPU 句向量运行模块，支持指定线程数、长度和 padding 策略，可校验模型哈希；完成 tokenization、ORT 推理、masked mean pooling、L2 normalization。
- `experiments/minilm_runtime_study.py`：prepare/tune/confirm/quality/profile/shapes 六阶段，明确区分配置选择与复验。
- `configs/minilm-runtime-study.json`：运行前固定的候选配置、输入行范围、轮次和验收目标。
- `experiments/minilm_runtime_report.py`：从原始结果计算按轮 bootstrap 区间与工程验收，生成带模型/分词器哈希的部署配置。
- `experiments/minilm_encode.py`：加载通过本地工程验收的部署配置，执行真实句向量推理示例；未构建生产 HTTP 服务，不涉及机械臂控制。

模块输入为一批 Python 字符串，经 tokenizer 得到 int64 `[B,S]` 的三个模型输入；模型输出 float32 `[B,S,384]`，池化按历史协议保留 float64 `[B,384]`。一次实例供一个调用线程使用，没有实现多请求调度或线程安全服务封装。

## 完整复跑

从仓库根目录，在已有 Python 3.12 环境安装 `requirements-macos-py312.txt`。本轮环境已安装；新机器请先核对依赖和目标硬件，不把 M4 Max 的最佳配置直接迁移到其他 CPU。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-py312.txt
.venv/bin/python -m experiments.prepare_minilm
```

所有结果与模型中间目录必须是新目录，不能复用已有实验路径。下例只在本地 CPU 执行，不需要付费算力。quality 阶段联网下载固定 revision 的公开 STS-B test 文件；模型和原始数据被 Git 忽略。

```bash
set -e
for phase in prepare tune confirm quality profile shapes; do
  .venv/bin/python -m experiments.minilm_runtime_study "$phase" \
    --output-dir results/my-runtime-study-001 \
    --assets-dir runs/my-runtime-assets-001
done

.venv/bin/python -m experiments.minilm_runtime_report \
  --output-dir results/my-runtime-study-001 \
  --assets-dir runs/my-runtime-assets-001

.venv/bin/python -m experiments.minilm_encode \
  --deployment results/my-runtime-study-001/deployment.json \
  --text "A robot picks up a cup." \
  --text "A robotic arm lifts a cup." \
  --text "The weather is rainy today." \
  --output runs/my-runtime-demo-001.json

.venv/bin/python -m unittest discover -s tests -v
```

若新机器未通过既定工程目标，部署配置将标记为候选，demo 会拒绝把它当作已验收配置。这不是程序失败，也不能删除该负结果后只保留成功尝试。阈值是本项目事先设定的工程目标，不是通用行业标准。

全量 tests 验证仓库已归档结果及可用的本地模型合同；新命名结果目录的性能/质量须阅读其 audit.json，不能将测试通过当成新机器性能达标。不要同时运行其他推理、训练或 profiler 任务来干扰计时。

## 实验范围

原先实验有真实整数执行和误差分析，但没有改变实际运行配置并复验，没有独立的后续 test-split 结果，也没有可加载的部署配置。本轮通过先调优再冻结、独立请求复测、保持量化策略不变的线程干预，以及首次 test-split 评估补齐这些环节。

同时保留两个对照：同一 INT8 模型的4线程与调优线程数用于归因线程配置的收益；FP32也独立调优，防止用未优化浮点基线夸大量化优势。纯推理和包含分词/池化的路径分别计时，后者仍不包括队列、网络、模型加载和机器人动作。

## 工程验证链条

问题：INT8 文件缩小，但原有4线程条件下比 FP32 慢。假设：固定默认线程并不适合全部 workload，因此先在不改量化策略的条件下调节执行并行度。实施：可配置 runtime、候选搜索、配置冻结、隔离进程复测。验证：不同请求、多个形状、独立 test split、profiler 和按轮统计。结果与限制：按报告真实数值陈述，保留调优后 FP32 更快的事实，不把 CPU 配置优化说成新量化算法。

本项目代码和实验由 Codex 辅助实现与执行，底层量化、图优化和整数内核由 ORT 提供。

## 短序列追加确认（可选）

这是在多形状探索之后提出的追加实验，保持同一已冻结模型和线程，不重新选参。需要先完成上面的主实验。

```bash
.venv/bin/python -m experiments.minilm_short_request_check \
  --output-dir results/my-short-check-001 \
  --assets-dir runs/my-runtime-assets-001 \
  --prior results/my-runtime-study-001
```

旧版首次短句质量筛选错误地继承tokenizer的padding/truncation配置，产生0对样本和NaN；修复版显式关闭两者，按真实token长度选择。仓库保留失败v1与修复v2；v2复用了未受影响的有效计时，并单独重新评测质量。上述命令使用修复后的版本，全新执行计时和质量，不依赖失败记录。
