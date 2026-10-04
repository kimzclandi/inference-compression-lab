# Qwen 共享前缀 KV 缓存：实现、归因对照与失效边界

日期：2026-10-03。个人工程项目，Codex辅助实现与执行。

## 为什么继续扩展现有仓库

MiniLM项目已有CPU量化、线程和离线批处理实验，但它是编码器。本轮在同一inference-compression-lab仓库新增真实Qwen生成式推理实验：围绕同一文档的多次请求，减少重复prefill，分开TTFT和decode，并加入缓存隔离、容量限制和淘汰对照。复用原Qwen项目的固定Q8权重，保持原训练/量化报告不变。

## 方法与贡献

输入是完整prompt token IDs及其精确前缀，输出是greedy生成token、首token耗时、decode耗时、缓存状态和逻辑KV字节数。缓存键绑定模型/分词器/配置文件指纹与完整前缀token；不使用字符串近似匹配，不接受与完整输入不一致的前缀，必须保留至少一个suffix token。

前缀第一次计算后保存各层K/V快照；每次请求复制成独立KVCache，再处理suffix和逐token生成，防止一个请求的续写污染后续请求。实现两条目LRU、64MiB逻辑张量预算、超预算绕过缓存、清空与失配拒绝。只支持单调用线程和普通浮点KVCache；模型权重和tokenizer在runtime生命周期内必须保持不变，更换模型必须重建runtime。

MLX/MLX-LM已有prompt caching能力，本项目没有发明该算法，也没有实现GPU内核。贡献是受控的缓存管理实现、请求隔离与错误路径、完整收益归因、真实模型输出复验及可复跑交付。[MLX-LM官方](https://github.com/ml-explore/mlx-lm) · [vLLM前缀缓存设计参考](https://docs.vllm.ai/en/latest/design/prefix_caching/)。没有使用vLLM，也不把本项目称为其复现。

## 固定协议

- Qwen2.5-0.5B-Instruct，源revision `7ae557604adf67be50417f59c2c2f167def9a775`；MLX affine Q8权重量化/group64，KV保持浮点。权重量化与KV量化分开。
- 本机Apple M4 Max，51,539,607,552字节统一内存；MLX0.29.3、MLX-LM0.26.3。固定串行batch1，不使用网络服务、动态批处理或付费算力。
- 性能前缀长度64/256/1024/2048，各4个suffix请求、每个强制生成32token、每种路径2次warmup，5轮随机排列路径次序。合成token流只用于固定负载性能测量，不是业务请求或任务质量。
- 计时从分词之后开始，含缓存查找、命中快照复制、miss构建、prefill、token采样和GPU同步；不含模型加载、分词、网络和排队。所谓cold是没有可复用前缀的warm-model路径，不是进程冷启动。
- 四条路径：全prompt直接计算；同样分段/快照路径但每次重算；第一次miss之后3次hit的真实四请求工作流；已构建缓存的4次hit。分别报告构建成本，避免只展示热缓存。

## 主结果（v2，5轮）

|前缀token|无前缀复用TTFT|缓存命中TTFT|命中TTFT降低|四请求总耗时降低：相对分段重算|前缀KV逻辑大小|
|---:|---:|---:|---:|---:|---:|
|64|12.26ms|7.26ms|40.8%|5.7%|0.75MiB|
|256|25.18ms|7.52ms|70.1%|10.4%|3.00MiB|
|1024|85.59ms|8.72ms|89.8%|25.7%|12.00MiB|
|2048|182.97ms|10.24ms|94.4%|38.0%|24.00MiB|

2048-token前缀的四请求工作流包含首次miss，耗时从分段重算的1.014秒降至0.629秒（约38.0%）；相对全prompt直接计算的1.198秒下降47.5%。两种对照分母不同，不能混用。缓存命中TTFT约10.24ms，相对全prompt约182.97ms下降94.4%；这不是完整生成快了94.4%。该长度decode约269→270token/s，基本没有改善。

64-token短前缀首次miss的TTFT约15.21ms，高于全prompt的12.26ms；缓存不是每次都更快。前缀越长、重复利用越多，收益在本轮负载下越明显；不推断到其他模型、GPU或线上流量。

逻辑KV占用公式：`2 × 层数24 × KV头数2 × 前缀长度L × head_dim64 × FP16字节2 = 12288L字节`，2表示K与V。2048-token前缀约24MiB。预算只限制保存快照的张量有效字节，不是GPU进程、分配器缓存、RSS或峰值内存上限；请求副本与模型另占空间。

v1是最初三路径探索，发现分段计算本身影响首token耗时；因此追加v2的分段重算对照，所有旧结果保留。v2是本轮最终主报告。没有做热状态、功率或跨机器控制；随机顺序与重复测量不能代替独立硬件验证。

## 真实任务输出一致性与负结果

74题、9个共享文段，来源于原Domain QA Lab已使用的本地SQuAD2公开dev子集；不是新的盲测，也不是SQuAD官方成绩。按文段顺序构造9次miss和65次hit是受控复用场景，不能当作生产命中率。

- 74/74题冷路径与缓存路径生成token序列完全一致；2题还与当前安装版本的MLX官方greedy解码器一致。固定token性能请求80/80组冷/热输出也一致。有限样本一致不构成对所有prompt无损的证明。
- 严格答案匹配两条路径均为18/74；可回答33题中18题匹配，无答案41题中0题输出规范的NO_ANSWER。始终拒答基线可得41/74，所以不能据此宣称问答系统可用于业务。该严格指标不同于忽略冠词/大小写的SQuAD归一化EM。
- 缓存隔离、错误前缀拒绝、交错请求后输出复验、淘汰后重算、超预算不入缓存均有真实模型记录。
- 交互demo中，前两题分别正确输出room17和packages；第三题应为charging station，却输出room17。保留该错误，说明缓存不会修复模型的任务错误。

## 失效场景：缓存反复淘汰

固定两条目缓存、每条trace12个请求、每次生成8token、5轮。hot连续访问同一前缀；cyclic_three循环访问3个不同前缀。对照均为相同分段但不复用的路径。

|前缀token|访问模式|每轮命中|每轮淘汰|总耗时变化|
|---:|---|---:|---:|---:|
|64|hot|11/12|0|降低21.15%|
|64|cyclic_three|0/12|10|增加0.01%|
|1024|hot|11/12|0|降低61.83%|
|1024|cyclic_three|0/12|10|增加0.59%|

1024-token循环访问下零命中、每轮淘汰10次，耗时约增加0.59%；不把这个很小的差值写成显著性能退化，但它明确未展示缓存收益。hot场景的61.8%与前述四请求38.0%属于不同生成长度和复用次数，不能混作同一成绩。

## 从干净环境复跑

需要Apple Silicon及可访问的Metal。沙箱首次导入MLX时可能因GPU不可见而失败，本轮经允许访问GPU后完成；没有悄悄切换CPU。Linux CI只验证标准库逻辑和归档证据。

```bash
python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx-prefix.txt
# 首次下载固定版本公开模型，约1GB；以下命令在新源码目录运行
.venv-mlx/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_dir='models/qwen-source')"
.venv-mlx/bin/python -m experiments.prepare_qwen_prefix_model --source models/qwen-source --output-dir runs/qwen-q8
.venv-mlx/bin/python -m experiments.qwen_prefix_study --model runs/qwen-q8 --output-dir results/my-prefix-study
.venv-mlx/bin/python -m experiments.qwen_prefix_trace --model runs/qwen-q8 --output-dir results/my-prefix-trace
python3 -m unittest discover -s tests -v
```

本轮从已有固定版本本地源重新构建Q8，权重SHA256与历史权重完全一致；没有重新从网络下载或在第二台机器复现性能。模型权重和缓存不提交。新实验目录必须不存在；已有目录报错。模型/分词器/协议/代码哈希、逐次token和耗时随结果保存。

可演示入口（输入自备文档与问题）：

```bash
.venv-mlx/bin/python -m experiments.qwen_prefix_demo --model runs/qwen-q8 \
  --document configs/qwen-prefix/demo-document.txt \
  --question "Which room does the red robot work in?" \
  --question "What does the blue robot inspect?" \
  --output-dir runs/my-prefix-demo
```

## 验证与代码导航

- `lab/prefix_cache.py`：无后端依赖的模型绑定、精确token缓存、LRU、字节预算和隔离接口。
- `lab/qwen_prefix.py`：真实MLX前缀构建、快照复制、cold/hit/miss/分段重算及同步生成。
- `experiments/qwen_prefix_study.py`：主实验、真实问答、官方解码对照、缓存错误路径。
- `experiments/qwen_prefix_trace.py`：复用率与容量淘汰对照。
- `experiments/prepare_qwen_prefix_model.py`、`qwen_prefix_demo.py`：独立模型准备和演示。
- `tests/test_prefix_cache.py`、`test_qwen_prefix_evidence.py`：缓存逻辑与离线证据检查，不伪装成CI重新执行模型。

本地含原ORT环境的43项测试通过；标准库环境31项通过、12项可选模型/数值检查跳过。实际Qwen GPU执行由主实验、trace及demo记录证明。保存两轮主实验及原始预测、时间记录、清单；没有改写旧MiniLM或Qwen训练实验。

## 发布与范围

使用原私有仓库的独立分支codex/qwen-prefix-cache，叠加于长度分桶分支；已创建[草稿PR #3](https://github.com/kimzclandi/inference-compression-lab/pull/3)，不自动合并前序PR，不改变仓库可见性。所有代码和实验均保留AI辅助归属。
