# Inference Compression Lab

面向模型量化、推理性能分析和端侧部署的可复现实验仓库。

## 当前状态

已完成一次真实预训练 MiniLM 的 ONNX Runtime CPU 动态 INT8 对照实验。报告与原始结果见 [实验报告](results/minilm-cpu-dynamic-int8/REPORT.md)。本仓库不包含历史 Jetson 项目代码、公司材料或模型权重。

| 工作包 | 状态 | 目标 |
|---|---|---|
| Jetson YOLOv8 / TensorRT | 待接入原项目与硬件 | 部署结果与分阶段性能对照 |
| Transformer PTQ | 已完成 MiniLM 动态量化粒度对照 | 尚未完成 SmoothQuant 或逐层误差研究 |
| 真实低精度后端 | 已完成 ORT CPU 整数算子执行验证 | 不代表 GPU、TensorRT 或 Jetson 结果 |

已提供：标准库计时器、数值模拟量化示例、测量配置模板、验收规范和测试。模拟示例不是 Transformer 实验，也不执行 INT8 内核。

## 本地运行

Python 3.10+；核心工具及测试不需要 GPU 或第三方库。

```bash
python -m unittest discover -s tests -v
python -m experiments.quantization_demo
python -m experiments.benchmark_smoke
```

两个示例仅写出明确标记为 synthetic 的本地输出，不能作为模型实验结果。`runs/` 默认不上传。

## 实际项目接入

真实 MiniLM 实验复跑（Python 3.12；联网下载公开模型与数据）：

```bash
python -m pip install -r requirements-experiment.txt
python -m experiments.prepare_minilm
python -m experiments.minilm_ptq
python -m unittest discover -s tests -v
```

复跑会覆盖同名结果；保留旧结果请先使用新 checkout 或复制结果目录。下载固定 revision 并校验 SHA256。量化使用 ORT 官方 API，不声称独立实现量化算法。

1. 按 `docs/acceptance.md` 确认数据、模型、环境和计时范围。
2. 将真实推理封装为无参数 callable；GPU 计时必须传入对应后端的设备同步函数，见 `lab/benchmark.py`。
3. 根据 `configs/run-template.json` 保存真实配置；不能用模板默认值替代实际值。
4. 保存原始样本、质量结果及执行证据，再填写 `docs/result-template.md`。

后续模型/后端依赖应按实际硬件版本锁定。仓库暂不自动下载模型、安装 CUDA 或启动收费算力。

## 方法来源与归属

- SmoothQuant: https://proceedings.mlr.press/v202/xiao23c.html
- 官方实现: https://github.com/mit-han-lab/smoothquant

未来复用外部代码时保留许可证、来源与版本。这里的标量量化示例是通用对称模拟量化，不是 SmoothQuant 复现。

## 成果边界

阅读、运行 demo、方法复现、个人实现与部署分别记录。fake quant 不等于真实 INT8；TensorRT 不自动代表量化；零权重不自动代表稀疏加速。README 的计划不得直接转写成已完成简历成果。
