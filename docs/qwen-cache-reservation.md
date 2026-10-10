# Qwen 原生 Cache 强对照：容量减少，速度门槛失败

本研究将逐步 `cat` 微基准推进到真实 Qwen2.5-0.5B Q8 推理。**三个长度均未通过预先冻结的加速门槛**；不替换默认推理入口。MLX 的原生 KVCache 已按 256 token 扩容，不能用相对每步完整拼接的收益代表相对成熟框架的收益。

## 方法和贡献

M4 Max、MLX 0.29.3、mlx-lm 0.26.3，使用与历史 Cache 工作负载实验逐文件哈希匹配的现有 Q8 模型。输入为固定维护记录文本的前 128/256/1024 tokens，batch=1，每次强制贪心生成 32 tokens（不因 EOS 提前退出）。不是模型质量评测。

对照沿用 `mlx_lm.models.cache.KVCache` 的原生 `step=256`；候选仅将每层新建 Cache 的 `step` 设置为 `prompt_length+32`。这是框架配置适配，**没有将 PyTorch AppendOnlyKV 移植到 MLX，也没有实现新 kernel、PagedAttention 或 KV 量化**。适配器在修改任何层之前验证所有缓存均为空、类型精确匹配且使用原生步长；不支持复用非空 Cache，也不是硬容量上限。

[协议](../configs/qwen-cache-reservation-v1.json)、[运行脚本](../experiments/qwen_cache_reservation.py)及[适配器](../lab/mlx_cache_reservation.py)在第一次执行前提交于 `af6f8796ad9a8e615676bdf19d0cae116c80bdb7`。采用 5 轮，每轮每组 2 次、每组 1 次预热，以固定随机种子交替组顺序；保留全部 60 次计时。主门槛为总请求耗时中位数改善至少 1.05×，且至少 4/5 轮更快，不事后调参或更换输入。

总请求时间包括新建 Cache、prefill、每步模型执行、采样、GPU 同步和 host token 提取，直到第 32 个生成 token；不包括权重加载和 tokenization。第 32 个 token 不再回填，所以最终可见长度为 P+31。首 token 时间在第一次同步后、host 提取前记录。该逐 token 同步循环不是异步生产服务吞吐基准。

## 冻结结果

|Prompt tokens|原生总耗时 ms|预留总耗时 ms|原生/预留|更快轮数|原生/预留 TTFT ms|最终分配 KV MiB（原生→预留）|
|---|---:|---:|---:|---:|---|---|
|128|114.175|114.827|0.994×|3/5|15.465→15.408|3.000→1.875|
|256|124.311|121.334|1.025×|3/5|22.811→22.692|6.000→3.375|
|1024|191.471|192.974|0.992×|2/5|84.100→83.861|15.000→12.375|

三个长度的 32 个生成 token、所有生成位置的完整词表 logits、24 层最终可见 K/V 均一致；logits 最大绝对差为 0。最终 KV 张量分配量分别减少 37.5%、43.75%、17.5%，但这**不是峰值 GPU 内存、RSS 或整模型内存下降比例**。按 24 层、2 个 KV heads、head_dim=64、K/V fp16 计算，每个容量 token 共 12,288 bytes；活跃 KV payload 不变。

零差异检查是本次 GPU 运行的记录，不代表所有输入上的等价证明。公开记录包括输入 token IDs、完整生成 token IDs、各步各层容量、最终 payload、全输出差异统计和源文件身份；未发布完整 logits/KV 数组，CPU CI 只能校验记录完整性、容量公式、计时算术和门槛，不能独立重演数值比较或 GPU 性能。复现数值检查需本地模型及匹配的 MLX/Mac 环境。

## 证据与复验

- [运行环境、模型文件及源码哈希](../results/qwen-cache-reservation-v1/run.json)
- [全部正确性检查](../results/qwen-cache-reservation-v1/correctness.json)
- [全部 60 次原始计时和生成序列](../results/qwen-cache-reservation-v1/samples.json)
- [CPU 离线验收](../experiments/verify_qwen_cache_reservation.py)
- [历史 KV 追加微基准及 SDPA 失败边界](kv-append-mps.md)

```bash
python -m experiments.verify_qwen_cache_reservation
python -m unittest discover -s tests -p test_mlx_cache_reservation.py -v
# 仅在本地 Mac 匹配环境，指定全新目录；不要覆盖冻结结果：
python -m experiments.qwen_cache_reservation --model /path/to/verified-q8 --output runs/my-new-cache-study
```

实验只改变分配粒度，模型、Attention、量化权重均由 MLX/MLX-LM 提供；个人研究工程采用 AI 辅助实现和核验。没有 CUDA/Ascend 运行、生产部署或质量提升结论；不能把本结果与历史 prefill/缓存复用/Attention 倍率相乘。
