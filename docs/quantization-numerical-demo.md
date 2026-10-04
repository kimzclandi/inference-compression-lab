# Quantization numerical demo / 量化数值演示

`experiments.quantization_learning` 是固定随机种子的数值模拟，用来比较权重量化粒度和激活 outlier 的误差；不计时、不报告加速。

```bash
python -m experiments.quantization_learning --output-dir runs/quantization-demo-001
```

输出目录必须为新目录。输入 X 为 `[1,32,8]`、W 为 `[8,4]`，输出 Y=XW 为 `[1,32,4]`。脚本将一个权重输出通道放大 40 倍，比较按张量/按通道权重量化与动态激活量化，再添加激活 outlier，保存原始合成输入、scale 和逐通道/整体误差。

映射为 `q = clip(round(x/s) + z, q_min, q_max)`，重建为 `x_hat = s * (q-z)`。其中 s 为正 scale、z 为零点，q_min/q_max 为整数范围。本脚本使用 signed [-127,127] 对称权重与非对称 U8 激活；不代表所有后端的舍入及范围约定。

先量化再还原到浮点后执行矩阵乘，因此属于 fake quant，不能证明真实 INT8 kernel 执行。整体 NMSE 的分母会随 outlier 改变，须同时检查逐通道误差。真实 ORT 模型的执行证据与性能测量见 [Mac 复现说明](mac-reproduction.md)。
