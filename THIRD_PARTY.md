# 第三方来源与实现归属

|对象|用途与归属|原始入口|
|---|---|---|
|Domain QA Lab|`lab/qa_metrics.py` 复用评分函数；移除原 CLI。保留原 MIT 声明于 NOTICE-Domain-QA-Lab.txt。旧预测只读复用|https://github.com/kimzclandi/domain-qa-lab |
|Qwen2.5-0.5B-Instruct|原始预训练/指令模型，Apache-2.0，revision `7ae557604adf67be50417f59c2c2f167def9a775`；无训练 adapter|https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775 |
|Qwen2.5-1.5B-Instruct|QA 整改本地 FP16/Q8 对照，Apache-2.0，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；无训练 adapter|https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/README.md |
|Qwen2.5-3B-Instruct|一次本地非商业研究容量对照，Qwen Research License，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`；不是默认部署候选，不随包分发模型|https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE |
|MLX / MLX-LM|框架提供 affine group weight-only Q4/Q8、Metal kernels、Qwen 层和 KVCache，MIT|https://github.com/ml-explore/mlx / https://github.com/ml-explore/mlx-lm |
|all-MiniLM-L6-v2|编码器模型，Apache-2.0，固定 revision 见原报告|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md |
|deepset/roberta-base-squad2|已由上游在 SQuAD2 训练的英文抽取式 QA；CC BY 4.0，revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`。本项目导出 ORT FP32/动态 INT8，保留 QA projection 浮点；不分发任何权重|https://huggingface.co/deepset/roberta-base-squad2/tree/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4 |
|ONNX Runtime|CPU 推理与动态 INT8，MIT|https://github.com/microsoft/onnxruntime |
|SQuAD / STS-B|任务材料；许可与变更说明独立于代码|[DATA_LICENSE.md](DATA_LICENSE.md)|

量化不是本项目原创。新增工作是诊断、受控模块回退、固定对照实验、失败处理和证据重算。Q8 不是 FP8，KV 保持浮点。下载模型随附 LICENSE/README 在本地重建时保留；权重不随仓库或发布包分发。

除上表明确继承许可的内容外，仓库当前尚未选择根代码许可证。维护者确认后才能加入新的复用授权；准备发布包本身不代表已经授权公开开源。个人实验、学校/团队工作、字节模型评测和 Jetson 经历分开归属。

新增正确性排序头使用本项目固定五特征训练代码；系数及标准化参数仅保存本地 `runs/`，发布材料只含公开数据派生特征、目标、训练过程与重建命令。上游模型能力、ORT 量化 kernel 与本项目实验/门控贡献分开归属。
