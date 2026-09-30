# 从 Jetson 部署经历到可解释的量化与推理优化能力

状态：学习手册与练习代码，尚未确认用户掌握。当前历史事实由用户陈述：在 NUSRI 面向多套机器人样机，独立负责感知模型部署、模型导出、TensorRT engine 构建、推理程序与 ROS 控制相关工作。具体历史 INT8 配置、校准方法、性能和质量记录尚待确认。新练习不能自动成为 NUSRI 任职期间已完成的历史成果。

## 一、先看领域地图

问题：在准确性、延迟、吞吐、内存和能耗约束下，让模型完成真实任务。核心对象是计算图、张量、低精度表示、后端内核和应用数据流。

学习主线：浮点基线与测量边界 → 量化数值误差 → 数据/粒度/校准 → 实际低精度执行 → profiler 定位 → 改一个变量并复验 → ROS 感知控制链路验证。

| 模块 | 必须掌握 | 进阶 | 暂时可忽略 |
|---|---|---|---|
| 数值 | scale、zero point、舍入/裁剪、对称/非对称、按张量/通道 | 误差传播、异常值处理、混合精度选择 | 自行提出新量化算法 |
| 数据与质量 | 校准/选参/最终评测分离、mAP/召回及失败样本 | 分布偏移、稳定性和统计不确定性 | 以小演示集代替任务评测 |
| 后端 | FP32/FP16/INT8 区别、实际算子精度、回退 | Q/DQ 图改写、融合、kernel profiling | 一开始同时学所有推理框架 |
| 性能 | 同输入协议、GPU 同步、P50/P95、端到端 vs 纯推理 | copy/compute overlap、内存复用、并发 | 未定位瓶颈就重写 CUDA kernel |
| 集成 | ROS 队列/时间戳、帧龄、感知与控制接口 | 拥塞/丢帧策略与闭环失败归因 | 未确认安全边界就接在线机械臂实验 |

### 两个能力档位（学习目标，不是对招聘门槛的保证）

**扎实、接近“中等偏上”的校招目标：**能独立建立可信基线，解释并完成一条 PTQ 路径，检查实际整数执行，诊断一次质量/速度问题，完成一项有假设的对照，说明个人责任和限制。交付的是可复验的工程闭环，不是一次 export 命令。

**明显更深入的目标：**在上述基础上，能用 roofline/算术强度与时间线解释计算或访存瓶颈；针对真实原因修改图、融合或数据搬运；对 Transformer 解释并测试 prefill/decode、KV cache、并发/批处理；能区分服务指标与单 kernel 指标。若目标岗位要求底层实现，再增加一个经过正确性与性能验证的 CUDA/Triton kernel 或 TensorRT plugin。这里均是待学习内容，未声称已经完成。

没有具体 JD 和面试表现，不能认定达到 overqualified。能够解释一个失败案例并完成因果验证，比记住大量框架名更能体现深度。

## 二、五个核心思维模型

| 思维模型 | 是什么、为什么成立 | 适用问题与边界 | 面试/工作怎么用 |
|---|---|---|---|
| 量化是有限网格近似 | 实数映射到有限整数格点；范围越宽、位数固定时格距越大 | 定位舍入与裁剪；理想半步误差界只在无饱和且规则适用时成立，不能保证任务精度 | 解释为何 outlier 影响其他值，以及 per-channel 为什么有时有用 |
| 性能由瓶颈决定 | 总时间包含推理、搬运、预后处理、排队；优化非瓶颈收益有限 | 初步定位端到端问题；有重叠并发时不能简单把各阶段耗时相加 | 先画时间线，解释为什么 INT8 kernel 更快也可能不改善相机到控制的延迟 |
| 局部误差不等于任务敏感性 | 网络会传播、放大或抑制误差；输出还经过阈值、NMS、排序等 | 选择候选层；局部 NMSE 排名不是因果任务排名 | 同输入局部对照 → 单层回退 → 完整质量复验 |
| 软件格式不等于硬件执行 | 低比特权重可能解量化/回退；图融合、布局与内核路径改变成本 | 验证是否真正使用目标精度；算子 profiler 仍不等于硬件指令级证明 | 同时查看权重/图、engine 层信息、profiler；不凭文件后缀判断 INT8 |
| 优化是受约束的实验 | 一次变化的收益只有在其他条件一致且测量可信时才能解释 | 选择不同精度、粒度和校准；数据复用/测试集调参会削弱外推 | 先写假设和接受标准，再测量，并保留负结果 |

### 一个足够理解第一课的公式

`q = clip(round(x / s) + z, q_min, q_max)`，`x_hat = s * (q - z)`。

x 是原浮点值，q 是量化整数，s 是正的缩放系数，z 是表示实数零的整数位置，q_min/q_max 是整数取值界限，x_hat 是还原的近似值。round 表示舍入，clip 表示截断到合法范围。

例如 s=0.1、z=0，x=0.26 → q=3 → x_hat=0.3。量化误差为 0.04。无饱和条件下，最近舍入误差通常不超过半个步长；若 x 已超出校准范围被裁剪，该界限不再适用。

教学脚本用 signed [-127,127] 的对称权重量化和动态非对称 U8 激活。它是明确指定的一种数值模拟，不表示所有后端、量化算子均采用相同的舍入和范围规则。

## 三、三个常见取舍：不要把选型写成普遍结论

### 1. 静态还是动态激活量化？

静态方法提前使用校准数据确定范围，可能减少运行时范围计算；依赖校准分布能代表部署输入。动态方法运行时计算范围，通常更能适应变化，但增加范围计算/调度开销。强证据应是同后端、同任务的质量与性能实验，不能用不同设备的速度表替代。

共识：输入分布和内核支持很重要。无定论：对所有模型和形状谁一定更好。实践：MiniLM 现有动态路线可以先学习；Jetson YOLO 的 INT8 路线要匹配真实 TensorRT/导出器版本、校准数据和设备。[ORT 官方量化文档](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

### 2. 更细粒度/更多回退，还是更低部署成本？

按通道通常能避免不同通道共用过宽范围；保留敏感层浮点可能恢复质量。代价可能是更多 scale、布局/转换和浮点计算，后端支持也有限制。反方最有力的依据是实际 engine 可能无法从更细粒度获益，而非“数学误差一定更差”。

共识：比较质量、延迟、存储/内存等多个目标。无定论：局部误差最小配置是否任务最好或最快。实践：先用 per-tensor/per-channel 同源对照，再只改一个回退节点；本仓库回退让句向量 MSE 改善但没有显著 Spearman 提升证据。

### 3. 先 PTQ 还是直接 QAT？

PTQ 不需要重新训练整个模型，成本低，适合先验证部署可行性。QAT 在训练中模拟量化效应，可能改善某些模型的量化质量，但需要训练数据、计算资源和训练/评测设计；训练中的 fake quant 仍须落到真实后端验收。

假设差别：PTQ 假设现有模型与校准/粒度选择足以达标；QAT 假设存在可学习修复的量化误差且训练成本值得。共识：按任务质量和资源约束选择。无统一结论：QAT 不保证某个设备上更快。实践：先排除错误预处理、糟糕校准和不支持的算子，再考虑 QAT。[TensorRT 量化工作流](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)

## 四、第一课：现在在 Mac 上完成，不需要 Jetson

从仓库根目录执行（现有 `.venv` 已具备依赖）。每次选择一个从未用过的输出目录。

```bash
.venv/bin/python -m experiments.quantization_learning \
  --output-dir runs/my-learning-001
```

源码：`experiments/quantization_learning.py`。输入 X 的形状为 `[1,32,8]`，W 为 `[8,4]`，输出 Y=XW 为 `[1,32,4]`。B=1、S=32、输入通道8、输出通道4；权重按输出通道量化时，每个列通道有一个 scale。

运行逻辑：固定随机种子 → 给一个权重输出通道放大40倍 → 比较按张量/按通道，以及是否量化激活 → 再添加一个激活 outlier → 比较输出 NMSE。输出保存原始合成输入、scale、逐通道与整体误差，不计时、不报告加速。

**这里先量化再还原到浮点，矩阵乘仍是浮点运算，是 fake quant。** 它用来理解数值机制，不应写进简历为“实现真实 INT8 推理”。整体 NMSE 的分母也会随着 outlier 改变，因此不要只看整体误差而忽略每通道误差。

接着复跑真实 ORT 模型（三种配置）：

```bash
.venv/bin/python -m experiments.minilm_ptq \
  --output-dir results/my-understanding-baseline-001 \
  --work-dir runs/my-understanding-baseline-001
```

输入为 int64 `[B,S]` 的 token IDs、mask 和 segment IDs；输出 float32 `[B,S,384]`，池化后为 `[B,384]`。质量实验 B=16、最长256；性能实验 B=1、padding64。`*-execution.json` 验证 CPU 整数相关算子；该实验不会变成 TensorRT 或 Jetson 测试。

最小实践任务：运行第一条命令前，写下你的预测：哪种配置误差更小？添加激活 outlier 后，权重按通道是否能解决全部问题？运行后比较预测与真实 JSON，解释不一致，而不是只抄数字。

30 秒面试回答：量化用有限低精度值近似原始张量，需要权衡范围覆盖与舍入误差。我会先建立浮点基线，比较粒度与校准配置，再用真实后端检查精度与性能；发现退化时区分局部误差、传播误差和后端回退，通过单变量实验验证原因。

深入追问：为什么误差界小不代表检测结果不变？如何判断是量化误差还是预处理不一致？量化图有 MatMulInteger 是否已经证明端到端加速？

常见错误：把 FP16 称为 INT8；把 fake quant 当作低比特内核；认为按通道总更快；把模型文件缩小当成峰值内存同比降低。

工程检查：数据/权重相同；输入形状和预处理相同；关闭 profiler 再计时；GPU 工作正确同步；确认实际执行精度；保留未获收益的配置。

### 第一课主动回忆（先不看答案，共10题）

1. scale 和 zero point 分别解决什么问题？零一定能表示吗？
2. 为什么一个大权重通道会伤害其他通道的按张量量化精度？
3. 若权重按通道后误差仍大，你会检查激活的什么性质？
4. 若超出校准范围，半个步长误差界为什么失效？
5. 为什么教学脚本的 INT8 数组不能证明执行了加速 INT8 kernel？
6. 降低 MSE 却没有提高 mAP，有哪些可能机制？
7. 如何设计一个实验区分局部误差和上游误差传播？
8. 为什么动态量化可能比 FP32 慢？怎样把猜测变成可验证假设？
9. 改变 batch/序列长度后，哪些性能结论需要重新验证？
10. 面试官问“哪些代码是你写的，哪些来自 ORT，哪些由 Codex 协助”，你如何准确回答？

答题后再针对错误做最小补课、反例和变式；当前不把任何题目登记为已掌握。

## 五、Jetson：先读取版本，再给设备匹配的导出命令

可单独复制 `experiments/jetson_inventory.py` 到设备，使用现有 Python 执行；只读、不安装、不改变功耗、不发机械臂命令。

```bash
python3 jetson_inventory.py --output jetson-inventory-001.json
```

在完整仓库内也可运行：

```bash
python3 -m experiments.jetson_inventory --output runs/jetson-inventory-001.json
```

它记录 Jetson release、Python/TensorRT/PyTorch/Ultralytics 版本、CUDA 是否可用、trtexec 路径和帮助，以及当前功耗模式（若有权限）。缺少信息会保留错误，不自动升级。Mac 试运行只验证可运行和正确识别为非 Jetson；尚未在用户设备执行。

### 历史 Ultralytics `int8`/`half` 接口的示例（不是设备已验证命令）

下面展示常见导出思路，必须先用 inventory 核对实际版本。当前官方文档已采用不同量化参数，并区分 TensorRT 10 与11；不要为了使示例通过而升级 JetPack。示例只适用于确实提供 `int8` 和 `half` 配置项、且该组合支持当前 YOLOv8 checkpoint 的环境。

```python
from pathlib import Path
import shutil
import tensorrt as trt
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

assert int(trt.__version__.split('.')[0]) in (8, 10), '先核对 JetPack/TensorRT 兼容性'
assert {'int8', 'half'} <= set(DEFAULT_CFG_DICT), '接口已变化；不要直接套用此示例'
weights = Path('/替换为你有权使用的本地模型/best.pt')
calibration = Path('/替换为独立校准集配置/calibration.yaml')
assert weights.is_file() and calibration.is_file()
run = Path('runs/jetson-int8-learning-001')
run.mkdir(parents=True, exist_ok=False)
local_weights = run / 'model.pt'
shutil.copy2(weights, local_weights)  # 导出器通常在权重旁写文件，保护原目录。
model = YOLO(str(local_weights))
model.export(format='engine', imgsz=640, batch=1, device=0,
             int8=True, half=False, data=str(calibration))
```

这个示例尚未在 Jetson 执行，不算完成量化。INT8 export 的 data 可能使用 YAML 中的 val 路径做校准，因此 calibration.yaml 的 val 应明确指向校准子集；最终评测另用独立配置。不要使用默认小演示集替代任务校准，也不要把校准集当最终评测集。[Ultralytics 官方导出说明](https://docs.ultralytics.com/integrations/tensorrt/)

导出后要独立评测质量、核查 engine 实际精度并计时。先做 FP32/FP16 基线，再做 INT8；每个配置独立目录。FP16 是低精度浮点优化，不能写成已经完成 INT8 量化。部分 Ultralytics engine 带元数据包装，不能未经检查就当作原始 TensorRT plan 交给 trtexec。

不要把裸 `trtexec --int8` 当作准确性已验证的量化；旧版隐式量化若没有正确校准/范围，可能使用不合适的动态范围。[TensorRT 10 官方说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)

进入 ROS 集成前先离线验证，区分 GPU 推理延迟、相机帧到感知输出的延迟、控制命令延迟和完整机械动作时间。ROS 队列积压可能让输出帧率正常而结果陈旧；优化的是帧龄/尾延迟还是吞吐必须说清。不能从“推理更快”直接推出“抓取成功率提高”。

## 六、简历草稿：当前版与验证后版分开

### 当前可用版（基于用户新确认的个人责任；未加入虚构指标）

- 独立负责多套机器人样机的感知模型部署与 ROS 控制模块开发，在 Jetson Orin Nano 上部署 YOLOv8 等感知模型，完成模型导出、TensorRT engine 构建、推理程序开发及机械臂控制流程集成。
- 面向机器人端侧运行需求，开展感知模型推理加速与量化优化工作，承担模型部署适配及感知—控制链路调试。

ROS1/ROS2、具体其他感知模型名称、历史量化方式和优化收益仍需核对，不自动填写。团队项目的其他环节不包含在“独立负责”的范围内。

### 验证后候选版（不得未经核实直接投递）

- 对 YOLOv8 建立 TensorRT FP32/FP16/INT8 对照，使用与最终评测分离的代表性数据完成 INT8 校准，结合 engine 层信息与 profiler 检查低精度执行和回退，评估检测质量、推理延迟及端到端尾延迟。
- 针对【实测瓶颈】实施【具体改动】，在固定输入、功耗与软件环境下，将【指标】从【真实基线】改善至【真实结果】，并验证【质量/功能约束】。

以上是能力目标和待核对槽位，不是已发生事实；没有结果数字时可以不写数字。若现在的新工作补齐了这些内容，应写入当前个人项目，或者标明后续复现，不能自动归入当年 NUSRI 任职时期。

本轮教学样例由 Codex 协助实现，核心 ORT/TensorRT 量化能力来自官方实现。本人实际复跑、定位、修改和验证后，按真实贡献更新表述；“独立负责部署”与“独立提出量化算法”是不同事实。

## 七、后续学习节奏

先交第一课预测和10道题的回答，再针对漏洞补课。下一单元是“从量化图到真实执行”，之后是“Jetson 计时与瓶颈定位”，最后才是“生成式模型与底层优化”。每个单元沿用30秒回答、深入追问、错误诊断、工程检查、最小实践与主动回忆。只把确认掌握的内容更新到掌握清单。

当前掌握清单：待用户答题与实际复跑后确认；当前环境与已有项目结果不自动证明个人已掌握。
