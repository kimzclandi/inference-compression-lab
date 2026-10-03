# Inference Compression Lab

面向模型量化、推理性能分析和端侧部署的可复现实验仓库。

## 当前状态

当前包含两条可复跑主线：[MiniLM CPU量化与离线批处理](docs/length-bucketing-study.md)，以及[Qwen生成式推理与共享前缀KV缓存](docs/qwen-prefix-study.md)。前者验证编码器吞吐，后者分开首token延迟、解码速度、缓存构建成本与容量失效边界；均保留负结果。

已分别完成 Windows 和 Mac 的真实 MiniLM / ORT CPU 动态 INT8 实验。历史 [Windows 报告](results/minilm-cpu-dynamic-int8/REPORT.md) 保持不变；新增 [Mac 报告与原始证据](results/minilm-mac-m4max-baseline/REPORT.md)。Mac 上 INT8 文件缩小约 35%，但指定负载推理较 FP32 慢约 14%；逐层误差分析指导的单节点 FP32 回退让句向量 MSE 降低约 6.8%，没有显著任务质量提升证据。本仓库不包含历史 Jetson 项目代码、公司材料或模型权重。

| 工作包 | 状态 | 目标 |
|---|---|---|
| Jetson YOLOv8 / TensorRT | 待接入原项目与硬件 | 部署结果与分阶段性能对照 |
| Transformer PTQ | 完成粒度对照、逐层累计/局部误差、单节点回退消融 | SmoothQuant 未完成；验证集探索性结论 |
| 真实低精度后端 | 已完成 ORT CPU 整数算子执行验证 | 不代表 GPU、TensorRT 或 Jetson 结果 |

已提供：标准库计时器、数值模拟量化示例、测量配置模板、验收规范和测试。模拟示例不是 Transformer 实验，也不执行 INT8 内核。

## 新增可交付成果：CPU 运行配置优化

[完整成果报告](results/minilm-runtime-study-v1/REPORT.md) · [复跑说明](docs/runtime-study-reproduction.md) · [模拟校招审查](docs/campus-project-review.md) · [简历候选条目](docs/resume-runtime-project.md)

固定按通道INT8模型，4→8线程在B=1、S=64下将推理延迟2.261 ms降至1.933 ms（约14.5%）。冻结配置后首次评测STS-B test 1,379对，Spearman相对FP32下降0.00155。公平调优后，INT8在长度64及更大形状仍慢于FP32；长度16的追加确认中延迟降低约9.0%，匹配短句子集质量下降0.00260。文件压缩约35%不等于内存或功耗同幅下降。

提供可复用的CPU句向量模块、部署配置与CLI；学习和实验均保留AI辅助实现归属。短句筛选的分词器padding故障及修复记录完整保留，不把无效NaN质量输出当成果。

## 新增：离线长度分桶与批处理（2026-10-03）

固定8线程、batch=8、64请求窗口，在2758句完整语料的7轮对照中，FP32总耗时1.252→1.044秒（下降16.6%），INT8下降18.6%；FP32分桶仍更快。FP32 Spearman保持一致，INT8分桶相对同精度顺序批处理下降0.000102。只验证离线吞吐，不代表线上排队延迟。

[实现、复现、质量边界与简历候选](docs/length-bucketing-study.md) · [完整语料原始结果](results/minilm-bucketing-confirm-v1/summary.json) · [先导结果](results/minilm-bucketing-v2/summary.json)

## 新增：Qwen生成式推理与前缀KV缓存（2026-10-03）

新增真实Qwen2.5-0.5B/MLX Q8前缀缓存、LRU容量管理、请求隔离与完整成本对照。2048-token前缀下，计入首次miss的4请求总耗时较分段重算下降约38%；74题缓存前后token完全一致。零命中的淘汰trace未获收益，问答严格匹配仅18/74，不能把运行时正确性称为问答系统质量合格。

[主报告、复跑与简历候选](docs/qwen-prefix-study.md) · [主实验](results/qwen-prefix-v2/summary.json) · [失效场景](results/qwen-prefix-locality-v1/summary.json) · [数据归属](configs/qwen-prefix/ATTRIBUTION.md)

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
python -m experiments.minilm_ptq --output-dir results/my-run --work-dir runs/my-run
python -m unittest discover -s tests -v
```

结果和中间文件目录必须显式指定且尚不存在，已有目录会拒绝运行。下载固定 revision 并校验 SHA256。Mac 完整依赖、逐层诊断、消融和轮换计时命令见 [复跑与交接](docs/mac-reproduction.md)。量化使用 ORT 官方 API，不声称独立实现量化算法。

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
