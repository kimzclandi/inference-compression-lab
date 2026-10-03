# MiniLM 离线长度分桶：实现与真实模型确认

日期：2026-10-03。由 Codex 辅助实现与执行；用户本人独立复跑、解释和修改能力尚待确认。

## 为什么强化旧项目

现有 MiniLM 已覆盖量化粒度、误差分析、浮点回退及线程优化。本轮补充同一模型上的输入组织与批处理优化，连接有效 token、padding、算子工作量与端到端耗时。已有 Qwen 项目提供教师答案监督与权重量化证据，暂不另建功能重复的仓库。

## 协议与结果

固定 ONNX Runtime CPU、8线程、batch=8、最大长度256、每64条输入为一个排序窗口。顺序批处理与长度分桶均使用动态 padding、相同模型、同一批句子、相同批次数。计时包含 tokenizer 克隆、分词、排序、张量构建、推理、池化、归一化和顺序还原；排除模型加载、磁盘读取、网络、线上排队等待。每种配置预热2遍，7轮中按固定种子随机安排4种配置顺序，保存全部28个整语料耗时。

本轮运行环境为 macOS arm64，详见 manifest 的系统与包版本。沙箱拒绝 CPU 型号及部分缓存信息探测，CPU 型号字段保留 null，不用旧实验的硬件字段替换。本轮没有功耗、热状态或CPU亲和性控制。两种模型 session 均驻留，不作峰值内存比较。

|精度|输入组织|2758句总耗时中位数|句/秒|1379对 Spearman|
|---|---|---:|---:|---:|
|fp32|顺序批处理|1.252345 s|2202.3|0.82030139|
|fp32|长度分桶|1.044400 s|2640.7|0.82030139|
|int8_per_channel|顺序批处理|1.473085 s|1872.3|0.81882550|
|int8_per_channel|长度分桶|1.199769 s|2298.8|0.81872322|

完整测试语料上，FP32 总耗时下降16.6%，INT8下降18.6%；这两个百分比均相对于同精度顺序批处理，不是INT8相对于FP32的加速。FP32分桶仍是本轮最快配置。padding后总token槽位54416→42072（减少22.7%），有效token仍为39028；345个批次和输入输出条数均不变。B×S²之和1284216→812016只是注意力尺寸代理量，不是实测FLOPs。

FP32 Spearman不变；INT8分桶相对自身顺序批处理的Spearman下降0.0001023，通过预设0.005门槛。INT8分桶相对FP32分桶下降0.0015782。INT8逐条向量最大余弦距离约0.01264，不能称为无损；不同分组会改变动态量化激活尺度，任务指标必须重新检查。

先导实验使用validation前128对（256句），FP32/INT8总耗时下降9.9%/12.8%；随后保持配置不变，以全部1379对test（2758句）确认。该test在此前项目中已经使用，不能称为全新盲测，也未审计预训练重叠。本轮没有据test搜索批大小、窗口或线程参数。先导语料偏短，主要结论采用完整语料确认。

7轮配对耗时减少比例的均值bootstrap 95%区间：FP32约16.38%–16.78%，INT8约18.48%–18.75%。这是同机短时重复的不确定性，不代表跨设备或生产流量泛化；它与“两个中位数的比值”不是同一统计量。

## 代码与正确性

- `lab/batching.py`：按截断后长度稳定排序，只在固定窗口内重排；保留尾批。
- `MiniLMRuntime.encode_batched`：每窗口分词一次，按当前批次最长输入补齐，执行推理，恢复原始请求顺序；返回[N,H]句向量和工作量计数。输出向量仍占O(NH)内存，分词缓存限于窗口。
- `experiments/minilm_bucketing.py`：预热、随机轮次顺序、原始计时、完整质量预测、模型/数据/源码哈希；输出目录必须不存在。
- 真实模型测试覆盖空字符串、重复句子、长文本截断、尾批、FP32数值对照及 tokenizer 状态隔离；证据测试从逐条预测重算Spearman，并重算耗时和分母。

## 复现

干净源码目录建议使用Python 3.12；安装依赖及下载模型需要网络，不需要GPU或付费API。下面的准备脚本校验FP32模型、tokenizer及数据哈希，在新目录重新生成INT8资产；无需先运行旧线程调优实验。`prepare_minilm`只在尚未准备模型的新目录运行，已有模型可直接进入第二条准备命令。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-py312.txt
.venv/bin/python -m experiments.prepare_minilm
.venv/bin/python -m experiments.prepare_bucketing_assets --assets-dir runs/my-bucketing-assets
.venv/bin/python -m experiments.minilm_bucketing --assets-dir runs/my-bucketing-assets --output-dir results/my-bucketing-pilot
.venv/bin/python -m experiments.minilm_bucketing --assets-dir runs/my-bucketing-assets --spec configs/minilm-bucketing-confirm.json --output-dir results/my-bucketing-confirm
.venv/bin/python -m unittest discover -s tests -v
```

无模型、无第三方依赖时也能运行最后的标准库检查，模型与数值计算相关检查会跳过。CI不重跑性能实验，不能把绿色CI理解为云端复现了16.6%的收益。

API示例（离线文本列表，保持输出顺序）：

```python
from lab.minilm_runtime import MiniLMRuntime
rt = MiniLMRuntime("models/minilm/onnx/model.onnx", threads=8, max_length=256)
vectors, stats = rt.encode_batched(["A robot.", "The robot picks up a red cup."],
                                  batch_size=8, window_size=64, bucket=True)
# vectors: [2,384]，每行对应同位置输入；stats记录有效/补齐token和批次数。
```

## 简历候选条目（未改动原简历）

> 在 AI 辅助下实现 MiniLM 离线长度分桶与批处理，支持窗口内排序、动态 padding 和输出顺序恢复；固定 ORT CPU 8线程、batch=8，在2758句混合长度输入上以7轮对照确认，FP32总处理耗时由1.252降至1.044秒（约16.6%），1379对STS-B的Spearman保持一致；同时保留INT8仍慢于FP32及分组导致数值变化的结果。

建议与原线程调优条目合并成“线程与批处理优化”，替换部分重复数字，不增加第三个相似量化项目。只有本人能复跑并解释实验后，才把它作为熟练掌握的面试经历；本轮实绩不倒填到旧实验日期。

## 面试前最小实践

自己复跑默认协议，再只改变window_size=8（等于batch_size），推导为什么排序基本不能降低每批最大长度，并用token计数与耗时验证。另解释为什么动态INT8会受同批其他句子影响、为什么吞吐提高不能推出线上P99延迟下降。

## 保留的限制与故障

v1启动时因sysctl CPU信息查询被沙箱拒绝而在任何模型实验前退出；增加“探测失败记录为未知”的处理后，新目录v2成功。旧结果没有覆盖。两次有效实验的complete.json记录原始JSON哈希。没有新增Jetson、CUDA、Ascend或线上服务实验，也没有测排队延迟、功耗和内存收益。

## 发布状态

本轮基于既有PR分支的c6012d6，独立分支codex/minilm-length-bucketing；代码和结果已获用户授权推送，草稿PR待审查；未合并、未改变私有仓库可见性。

## 本轮验证

`python -m unittest discover -s tests -v`：35项通过，含真实模型与历史证据检查；`git diff --check`通过。日志保存在`results/minilm-bucketing-confirm-v1/tests.log`。测试通过证明代码合约与保存证据一致，不代替模型质量与实际工作负载测量。

审查准备补充：证据检查以430个历史文件的SHA256清单替代Git历史依赖，支持浅克隆和源码下载包；新增可指定资产目录的准备/运行入口。本机重新量化得到的INT8文件与历史文件哈希一致；没有宣称已完成另一台机器上的性能复现。
