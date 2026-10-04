# Mac MiniLM 实验复跑与交接

本轮在 Apple M4 Max / 48 GiB / macOS 27.0 / arm64 上真实执行。实验代码由 Codex 协助实现与运行，量化核心调用 ORT 官方 API。用户是否已自行复跑、理解和参与修改尚未确认。

## 环境

- 2026-09-30 检出私有仓库，起始 HEAD `eefd28ac66d202477ad0e157b3d400c5a24952bd`；干净 main，无仓库或父目录 AGENTS.md。
- 默认 Python 3.14.6；依赖检查发现 PyArrow 18.1.0 对该解释器走源码包，因此改用独立 `.venv` 的 CPython 3.12.13。
- Windows 的七个直接依赖锁定均可用于此 Mac/Python 3.12 组合；`requirements-experiment.txt` 未修改。`requirements-macos-py312.txt` 保存实际完整环境版本，不是跨平台通用锁。
- ORT 1.30.0 安装报告 CoreML、Azure、CPU 三个可用 provider；所有实验显式只请求 CPUExecutionProvider，实际 session/provider 和 profiler 再次验证 CPU。没有使用 MPS、Apple GPU 或 Core ML。
- 初次沙箱内 GitHub/硬件检查失败；在允许网络/硬件查询的环境核验登录有效、仓库私有。沙箱内曾出现 PyArrow CPU 特征查询警告；正式基线、消融和轮换性能实验在可正常查询硬件的环境运行。

## 复跑命令

从仓库根目录执行。先安装 Python 3.12，再创建项目环境；无需 GPU、付费 API 或远程算力。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-py312.txt
.venv/bin/python -m experiments.prepare_minilm
```

下载脚本保留固定 revision；执行后比较 `models/minilm/manifest.json` 与历史 `results/minilm-cpu-dynamic-int8/provenance.json`。本轮四项 SHA256 完全一致。模型和原始数据留在 Git 忽略目录，不发布。

每次选择全新的结果目录和工作目录。已有目录（包括空目录）会报错退出；不要把历史目录删除后复用。

```bash
.venv/bin/python -m experiments.minilm_ptq \
  --output-dir results/my-mac-baseline \
  --work-dir runs/my-mac-baseline

.venv/bin/python -m experiments.minilm_layer_errors \
  --optimization all \
  --output-dir results/my-mac-layer-errors \
  --work-dir runs/my-mac-layer-errors \
  --baseline-work runs/my-mac-baseline

.venv/bin/python -m experiments.verify_local_weights \
  --quantized runs/my-mac-baseline/int8_per_channel.onnx \
  --probe-work runs/my-mac-layer-errors \
  --output results/my-mac-layer-errors/local-weight-verification.json

.venv/bin/python -m experiments.minilm_ptq \
  --output-dir results/my-mac-ablation \
  --work-dir runs/my-mac-ablation \
  --variants fp32 int8_per_channel int8_per_channel_excluded \
  --exclude-nodes-json results/my-mac-layer-errors/exclusion.json

.venv/bin/python -m experiments.minilm_latency_crossover \
  --output-dir results/my-mac-crossover \
  --baseline-work runs/my-mac-baseline \
  --ablation-work runs/my-mac-ablation

.venv/bin/python -m experiments.minilm_quality_audit \
  --output-dir results/my-mac-quality-audit \
  --ablation-results results/my-mac-ablation

.venv/bin/python -m unittest discover -s tests -v
```

不要让其他模型任务与计时实验并行运行。上述测试验证仓库已归档的 Mac 结果；新命名目录应按相同证据合同另行验收。`--worker` 是父进程使用的内部入口，不是完整实验入口。

每份环境/协议记录保存运行命令、Git HEAD 和源文件 SHA256。历史 Git 对象中的源码才是该次运行源码；最后一版程序增加选项后不应冒充原始运行源码。中间模型、句向量和完整 profiler 文件保留在 `runs/`；已提交的 execution JSON 含原始 profiler 节点事件，不含权重和句子。

## 指标与形状

输入为 tokenizer 产生的 `input_ids`、`attention_mask`、`token_type_ids`，均为 int64 `[B,S]`。模型输出 float32 `[B,S,384]`；先按 mask 对 token 求均值，再 L2 归一化得到 `[B,384]` 句向量。两句点积作为余弦相似度，和 STS-B 标签计算 Spearman。

逐层比较对象是 `[B,S,H]` 激活，仅统计非 padding token。MSE = `sum((reference-candidate)^2)/N`，NMSE = `sum((reference-candidate)^2)/sum(reference^2)`；N 为有效元素数。先累计误差平方和与能量再相除，避免不同长度 batch 的简单平均偏差。

局部实验固定 FP32 输入，只改变该 MatMul 的动态 U8 激活和 INT8 权重计算路径；累计实验比较完整图对应节点。两者都调用实际 ORT 算子。局部 NMSE 衡量数值误差，不能替代任务敏感性；选择后仍须在完整模型验证。

## 验证边界

匹配 revision/hash，使用独立结果目录，保持计时范围、形状和线程一致；核对 profiler/provider、插桩漂移及浮点回退。INT8 权重、文件缩小和局部误差下降分别不等于整数执行、运行内存降低或任务质量改善。

全量 tests 验证仓库已归档结果及可用的本地模型合同；新结果目录的性能与质量须阅读其 audit.json，不能将测试通过当成新机器性能达标。计时期间避免同时运行其他推理、训练或 profiler 任务。

量化数值模拟的输入、输出和 fake-quant 范围见 [数值演示](quantization-numerical-demo.md)。
