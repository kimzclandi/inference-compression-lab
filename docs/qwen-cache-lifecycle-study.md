# KV 缓存生命周期：异常提交修复与容量阈值验证

2026-10-04；Codex 辅助审查、实现、执行。用户独立解释和复跑待检验。继续现有项目的任务 2，不新增项目；具体 JD 仍未提供，本报告不声称完成岗位对齐。

## 审计与问题复现

起点为远端/本地一致的 `c661a1ead09598c318b1fce66583a0dfdd95c8ed`，工作区干净，PR #4 仍为 OPEN/DRAFT。新分支 `codex/qwen-cache-lifecycle` 叠加于 #4。继续适用本任务用户提供的 AGENTS.md 教学及事实边界要求。

`results/qwen-cache-lifecycle-v1/before-fix.json` 保存修复前故障：驻留 A/B、上限两条，C 的 clone 抛异常后，旧实现已经把 A 淘汰、留下 B/C，并将 misses 从 2 加到 3、evictions 从 0 加到 1。请求失败却消耗了可用工作集。命中分支也在 clone 之前改变 LRU/hits。

## 修复与语义

`lab/prefix_cache.py`：先完成 builder、字节校验与 clone，再提交状态。miss 在临时 OrderedDict 中计算淘汰，只复制元数据引用，不批量复制 KV；hit 也在 clone 成功后才更新 LRU。异常不改变驻留内容、顺序、逻辑字节或成功计数，单独累加 failures。

- hits/misses 表示成功获得请求缓存的分类；成功 bypass 同时计一次 miss 和一次 bypass。
- failures 记录 acquire 抛出的 Exception，不计成功 hit/miss。非法 token 等也会进入此计数；runtime 在进入 acquire 前拒绝的 prefix mismatch 不改变 store 计数。
- clear 只清驻留条目和字节，保留累计计数。评估每轮用 before/after 差值，不能直接读历史累计 hits 充当本轮命中。
- model_id、max_entries、max_bytes 为只读属性；需要改变它们时新建 store/runtime，防止驻留状态与配置脱节。拒绝 bool、float、非正数容量。身份字符串必须绑定模型/分词器/配置。
- 权重或 tokenizer 在 runtime 内被原地修改仍不会自动侦测。调用者必须保持不变，更换时重建。精确 token key 防字符串边界碰撞；不能保证同文本在不同 tokenizer/template 下有同 token。

契约限制：clone 不得修改输入（包括抛异常前），builder 必须交出自有不可变快照，回调不得重入，单调用者；不是恶意回调、并发、进程崩溃或真实 OOM 下的通用事务系统。复制前保留旧条目会增加临时共存内存，这是可靠性与峰值压力的取舍，不宣称降低内存。

## 先预测，再执行

提交 `0454e1d` 先保存 `configs/qwen-prefix/lifecycle.json` 预测及实现，再启动真实 GPU 实验：三前缀循环 `[A,B,C]×4`，冷缓存共 12 请求。

|配置|事先预测：hit / miss / eviction|实际每轮|驻留逻辑 KV：1024-token|
|---|---|---|---:|
|两条目，预算容三条|0 / 12 / 10|全部符合|24 MiB|
|三条目，预算容三条|9 / 3 / 0|全部符合|36 MiB|
|三条目，预算只容两条|0 / 12 / 10|全部符合|24 MiB|

字节限制与条目限制共同约束容量。逻辑每前缀字节为 `2×24×2×P×64×2=12288P`；包含 K/V、层、KV 头、前缀长度 P、head dimension、FP16 字节。KV 仍为浮点，Q8 是权重量化。

## 真实模型验证

固定历史同一 Qwen2.5-0.5B-Instruct MLX Q8 模型指纹；M4 Max，MLX 0.29.3、MLX-LM 0.26.3。batch=1 串行、预分词、每请求强制 greedy 生成 8 token；前缀 64/1024；四路径各预热每前缀两次，每长度 5 轮随机路径次序，计入冷缓存首次 miss。对照为上轮新增的无快照直接分段重算。

|前缀|直接重算：12 请求中位数|两条目|三条目且预算足够|三条目但字节仅容两条|
|---:|---:|---:|---:|---:|
|64|0.421241s|0.421833s|0.348464s（下降17.28%）|0.423735s|
|1024|1.121056s|1.117065s|0.548303s（下降51.09%）|1.121376s|

两条目/受字节限制路径相对直接重算只有约 −0.59% 到 +0.36% 的小变化，没有明确性能收益；不包装为显著退化或优化。三条目结果支持这个固定访问序列的容量收益，不意味着容量增加总能加速。51.09% 的分母是 **12 请求、8 生成 token、1024 前缀**，不能与上轮四请求、32 生成 token、2048 前缀的 37.17% 混用。

40 条 trace，共 480 请求记录；按长度、轮次、请求位置对照直接分段路径，生成 token 全部一致。不是 480 个独立业务样本，没有在线排队、P95/P99、功耗或真实业务命中率结论。

五类真实模型故障注入：hit clone、miss clone、超预算 bypass clone、真实 prefill 后 builder 异常、非法 snapshot size。均保持原 A/B 顺序及字节，成功计数不变、failures 增一，重放 A/B 与完整 prefill token 一致。这里的 clone 异常是主动注入，不是声称触发了真实 GPU OOM。另验证重复超预算请求不入缓存、错误 prefix 拒绝不改状态、clear 后首次请求为 miss 且输出一致。

性能记录对应 `0454e1d`。之后增加只读容量属性和 contracts-only 入口，在 `dff490c` 对最终实现重新执行全部真实模型合约 smoke，保留到独立目录；未把最终 smoke 说成第二次完整性能复跑。

## 证据与复跑

- 修复前：`results/qwen-cache-lifecycle-v1/before-fix.json`。
- 主实验：`results/qwen-cache-lifecycle-gpu-v1`，含 manifest/spec、token 工作负载、逐请求计时、注入异常前后状态、重放输出、汇总及校验和。
- 最终合约：`results/qwen-cache-contracts-final-v1`。
- 复核：`results/qwen-cache-lifecycle-v1` 中的日志和 preservation/source 检查。
- 实现：`lab/prefix_cache.py`；运行/离线重算：`experiments/qwen_cache_lifecycle.py`、`experiments/verify_qwen_cache_lifecycle.py`。

最短检查不需要 MLX 或模型：

```bash
python3 -m experiments.verify_qwen_cache_lifecycle results/qwen-cache-lifecycle-gpu-v1
python3 -m unittest discover -s tests -v
```

GPU 最小演示需要已有固定 Q8 模型与 MLX 环境，模型准备沿用旧报告：

```bash
.venv-mlx/bin/python -m experiments.qwen_cache_lifecycle --model runs/qwen-q8 \
  --contracts-only --output-dir runs/my-lifecycle-contracts
```

完整容量对照：

```bash
.venv-mlx/bin/python -m experiments.qwen_cache_lifecycle --model runs/qwen-q8 \
  --output-dir results/my-lifecycle
python3 -m experiments.verify_qwen_cache_lifecycle results/my-lifecycle
```

所有新目录拒绝覆盖。模型不上传，无付费算力。55 项标准库检查中 43 项通过、12 项可选项跳过；模型执行由独立 GPU 原始记录证明，不由测试数量代替。历史结果不覆盖。没有新环境依赖安装或跨设备复现结论。

## 能力事实更新与下一主线

|候选岗位能力（非具体 JD）|已实现|已执行验证|未覆盖／本人掌握|
|---|---|---|---|
|KV 生命周期与失效|异常提交顺序、配置不可变、独立失败统计|五类故障、clear/bypass/mismatch、重放 token|并发、真实 OOM、恶意回调不覆盖；本人待验证|
|容量与性能分析|两/三条目、字节上限双重约束|先预测后验证，5轮/40 traces/480请求|非业务流量，不推导线上命中率|
|可复核实验|数据/模型/代码身份、独立原始记录与重算|格点覆盖、LRU独立重算、计数/字节/输出检查|完整新机器交付仍待任务5|
|模型压缩/硬件后端|沿用原 MLX Q8 权重与浮点 KV|本轮只验证缓存工程|无新增蒸馏/量化算法、CUDA/昇腾/TensorRT实绩|

下一主线是任务 3：固定更有代表性的低命中/混合访问负载，区分预分词与包含分词的应用耗时。任务 4 的质量失败细分与任务 5 的全新依赖安装仍待处理。收到 JD 后再调整优先级。

简历候选（不直接修改原文件）：

> 在 AI 辅助下完善 Qwen 前缀 KV 缓存的异常提交与容量管理，通过真实模型故障注入验证失败后条目、LRU、逻辑字节及请求输出保持一致；预先预测并验证三前缀循环访问的容量阈值，保留零命中及字节预算限制下无收益的对照。

现场演示优先展示“失败前后状态不变 + 重放 token 一致”，避免只展示快了多少。个人掌握与训练任务见 `qwen-cache-lifecycle-learning.md`；本轮仍不能标记用户独立掌握。
