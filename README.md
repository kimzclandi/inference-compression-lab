# Inference Compression Lab

面向模型量化、推理性能分析和端侧部署的可复现实验仓库。

## 当前状态

这是实验基础设施初版，不是完成的量化或加速成果。当前不含用户既有项目代码、模型权重、公司材料或真实硬件性能结果。

| 工作包 | 状态 | 目标 |
|---|---|---|
| Jetson YOLOv8 / TensorRT | 待接入原项目与硬件 | 部署结果与分阶段性能对照 |
| Transformer PTQ | 待选模型并实现 | 浮点、基础 PTQ、改进方法数值对照 |
| 真实低精度后端 | 待选择兼容后端 | 核验整数执行与性能收益 |

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
