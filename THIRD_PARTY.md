# 第三方来源与实现归属

|对象|用途与归属|原始入口|
|---|---|---|
|Domain QA Lab|`lab/qa_metrics.py` 复用评分函数；移除原 CLI。保留原 MIT 声明于 NOTICE-Domain-QA-Lab.txt。旧预测只读复用|https://github.com/kimzclandi/domain-qa-lab |
|Qwen2.5-0.5B-Instruct|原始预训练/指令模型，Apache-2.0，revision `7ae557604adf67be50417f59c2c2f167def9a775`；无训练 adapter|https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775 |
|MLX / MLX-LM|框架提供 affine group weight-only Q4/Q8、Metal kernels、Qwen 层和 KVCache，MIT|https://github.com/ml-explore/mlx / https://github.com/ml-explore/mlx-lm |
|all-MiniLM-L6-v2|编码器模型，Apache-2.0，固定 revision 见原报告|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md |
|ONNX Runtime|CPU 推理与动态 INT8，MIT|https://github.com/microsoft/onnxruntime |
|SQuAD / STS-B|任务材料；许可与变更说明独立于代码|[DATA_LICENSE.md](DATA_LICENSE.md)|

量化不是本项目原创。新增工作是诊断、受控模块回退、固定对照实验、失败处理和证据重算。Q8 不是 FP8，KV 保持浮点。下载模型随附 LICENSE/README 在本地重建时保留；权重不随仓库或发布包分发。

除上表明确继承许可的内容外，仓库当前尚未选择根代码许可证。维护者确认后才能加入新的复用授权；准备发布包本身不代表已经授权公开开源。个人实验、学校/团队工作、字节模型评测和 Jetson 经历分开归属。
