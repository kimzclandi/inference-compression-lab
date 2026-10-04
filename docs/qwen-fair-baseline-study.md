# Qwen 分段基线公平性：有收益，但收益不来自多余快照

2026-10-03。本轮由 Codex 辅助审计、编码、执行与分析。

## 具体问题、假设和实现

旧 `reuse_prefix=False` 仍调用 `build` 保存 snapshot，再调用 `clone` 恢复请求 KV。不复用的请求只需自己的 KV，不需要这两个快照步骤。保留旧路径默认语义；新增 `segmented_snapshot=False`，前缀得到请求私有 KV 后直接处理 suffix。显式 `mx.eval` 保证惰性前缀计算完成，而不是把未执行计算移出计时。

在提交 `15c0c49` 中先冻结 `configs/qwen-prefix/fair-baseline.json` 的假设、负载、轮数和验收条件，之后才执行。假设是新基线可能更快并缩小缓存收益；不要求正向结果。验收要求五路径同输入/greedy/输出长度、所有请求 token 一致、计时阶段可对账、无复用路径不访问共享 store。

五路径：full 完整 prefill；legacy_segmented 旧快照分段重算；direct_segmented 本请求 KV 直接续算；cached_workload 首 miss 后三 hit；warm_hit 预先准备后四 hit。热准备时间独立保存，不计入其四请求计时。

诊断开启 `profile=True` 时记录 lookup（含 token 校验、哈希、字典成员查询）、prefix prefill、snapshot copy、request clone、suffix prefill 加首 token 采样、decode；管理与计时杂项记入 unattributed。每条记录各阶段之和等于 total。decode 从首 token 完成到第 32 token 完成，因此只有 31 个间隔。四请求外层 wall timer 还含函数返回和 Python 调用开销。

**额外同步会改变执行图、开销与设备状态，所以诊断轮次不能当主性能结果，不能把阶段中位数相加冒充总耗时中位数。** 两次 `deepcopy` 是 API 层快照操作，不是已验证的物理显存拷贝/带宽测量。新旧路径还有预分配容量、惰性图差异，不能把全部差值归因为复制。

## 固定协议与结果

Qwen2.5-0.5B-Instruct，MLX Q8/group64、浮点 KV，模型文件指纹与历史 v2 一致。Apple M4 Max，MLX 0.29.3、MLX-LM 0.26.3。串行 batch=1，前缀 64/2048，四请求，每请求强制 greedy 生成 32 token；每模式预热 2 次，每组 5 轮，固定种子随机模式顺序。

先跑历史 token 流，再按相同协议跑另一组天文记录 token 流，不按结果选参。两组都是合成形状负载，不是生产流量或新 QA 盲测。分词在计时外，排除模型加载、网络、排队。无温度、功率和跨设备控制。分别执行不插桩主测量与插桩诊断，共 200 组、800 请求记录；400 条来自主测量，400 条来自诊断，不是 800 条独立业务样本。

下表为**不插桩**四请求总耗时中位数，缓存列计入首次 miss：

|负载|前缀|旧分段 s|新直接分段 s|缓存 s|缓存相对新基线下降|缓存相对旧基线下降|
|---|---:|---:|---:|---:|---:|---:|
|历史|64|0.456291|0.453701|0.432148|4.75%|5.29%|
|历史|2048|1.005062|1.003266|0.630392|37.17%|37.28%|
|确认|64|0.468765|0.467844|0.444188|5.06%|5.24%|
|确认|2048|1.051270|1.054601|0.640794|39.24%|39.05%|

历史 2048 新基线比旧基线仅快约 0.18%，确认负载反而慢约 0.32%。没有稳定显著的“去掉快照使基线加速”证据。历史报告约 38.0% 保持原样；新实验不是对旧数据改分母，而是在新时间段同时复跑所有路径，得到新基线 37.17%。二者不能相减解释为纯复制开销。

历史 2048 诊断轮次中，cached miss 的 prefix prefill 中位数约 136.57ms，snapshot 操作约 0.093ms，request clone 约 0.072ms；命中 lookup 约 0.144ms，request clone 约 0.082ms，suffix 加采样约 10.23ms，decode 约 118.12ms。命中省去前缀计算是主要收益来源；这些是本次诊断计时，不能推导物理拷贝带宽。完整分阶段数值见 summary，逐请求见 timings。

不插桩历史 2048 direct/hot decode 约 268.80/268.09 token/s；确认组约 269.28/270.56。没有 decode 明显加速证据。热命中 TTFT 约 10.32ms 与 10.38ms，不能写成完整生成同幅提速。

所有记录都按 dataset/长度/轮次/请求与不插桩 full 的生成 token 对照通过，其中包含插桩与不插桩的一致性。仅有限合成样本验证。本轮没有重跑 74 题问答，历史 74/74 与 18/74、41/74 仅重新读取和离线验证，不能声称本轮新质量成绩。

## 实际执行与最短检查路径

无需模型、MLX、Git 历史、网络即可重算归档数据：

```bash
python3 -m experiments.verify_qwen_prefix_fair results/qwen-prefix-fair-v1
python3 -m unittest discover -s tests -v
```

verifier 检查校验和、完整模式格点、重复/缺失记录、输入与输出长度、token parity、命中状态、逻辑 KV 字节、阶段对账、分母和汇总重算；故意删记录、改 token、把复制成本塞进直接路径的测试必须被拒绝。标准库检查不是 GPU 性能复跑。

Apple Silicon 的真实模型最小演示（模型必须已准备）：

```bash
python -m experiments.qwen_direct_contract --model runs/qwen-q8 --output-dir runs/my-direct-contract
python -m experiments.qwen_prefix_fair --model runs/qwen-q8 \
  --spec configs/qwen-prefix/fair-smoke.json --output-dir runs/my-fair-smoke
python -m experiments.verify_qwen_prefix_fair runs/my-fair-smoke
```

直接路径合约探针把 `build`、`clone`、store lookup 替换成“一调用就报错”，同时记录模型输入形状和每层 offset；已实测通过。64-token 前缀从 offset=0 开始；suffix 从 offset=64 继续；逐 token decode 输入 `[1,1]`，位置递增。交错另一个前缀后重放仍与 full token 一致。新源码归档目录中也执行五路径 smoke，复用当前已经安装的 MLX 环境，不冒充全新安装。

完整复跑（需要约 1GB 模型与 Apple Metal，可按 [旧报告](qwen-prefix-study.md) 固定 revision 准备模型）：

```bash
python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx-prefix.txt
.venv-mlx/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_dir='models/qwen-source')"
.venv-mlx/bin/python -m experiments.prepare_qwen_prefix_model --source models/qwen-source --output-dir runs/qwen-q8
.venv-mlx/bin/python -m experiments.qwen_prefix_fair --model runs/qwen-q8 --output-dir results/my-fair-study
python3 -m experiments.verify_qwen_prefix_fair results/my-fair-study
```

每个输出目录必须是新目录。完整性能实验本轮使用已有环境和已有固定权重，未重新从网络安装或下载；上面全新准备命令在本轮没有完整执行。源码下载包的离线检查已支持；真实实验的现有 `source_record` 仍依赖 Git，此限制留给任务 5，归档 smoke 使用重新初始化的独立 Git 仓库，不依赖完整旧历史。

## 剩余风险与下一步

- store 是单调用者接口；模型/tokenizer/config 生命周期不变是调用者约定，不是自动侦测可变权重的保证。
- `PrefixCache.acquire` 当前 miss 入库后才 clone，clone 异常可能留下条目并先淘汰旧条目。旧测试只证明 builder 异常不入库，不应扩大到所有异常原子性。任务 2 应先复现再修复。
- 两条目循环三个前缀的零命中是旧实测；三条目对照本轮尚未执行。
- 逻辑 KV 是 `2×24×2×L×64×2 = 12288L` 字节，只算保存状态的有效张量；预分配、请求副本、分配器和 RSS 不由这个预算约束。没有本轮运行内存/功耗改善证据。
- 没有 CUDA、昇腾、TensorRT、内核开发、多卡、服务并发或 P95/P99 结论。问答质量诊断仍未完成。
