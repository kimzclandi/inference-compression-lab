# Independent decoder review / 独立解码审查

This audit enumerates every legal start/end pair (up to 30 tokens), checks every token inside the pair against the context mask, reconstructs the exact Unicode character span, trims its boundary whitespace, and recomputes each window's best span, same-window CLS-null difference, winner, and sigmoid ordering score. It imports neither the runtime decoder nor the project's scoring/gate/verification modules. It scores only the protocol's fixed grid on calibration; evaluation uses the frozen selected threshold.

本项目定义窗口 `w` 的最高合法 span logit 和为 `S_w`，该窗口 CLS start/end logit 和为 `N_w`，选择 `argmax_w(S_w−N_w)`。窗口、start、end 的平局按较小索引处理。`sigmoid(S_w−N_w)` 只是固定排序分数，不是校准过的正确率。tokenizer 产生的零长度 context offset 不可作端点，可在 span 内部占一个 token；把它改成 mask=false 会错误截断跨连续空格的 span。

Transformers v4.56.2 的官方 QA **example postprocessor** 使用跨窗最小 null 分数，与候选中最高 raw span 分数作差；还使用 top-n start/end 候选。因此这不是该官方 postprocessor 或通用 `pipeline` 的逐值复现。[官方源代码](https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/examples/pytorch/question-answering/utils_qa.py)

一个无需模型输出的反例：窗口 A 的最好答案 `cat` 得分 18、null 20；窗口 B 的 `dog` 得分 10、null 0。本项目选 B（margin 10 大于 −2），跨窗 highest-span/min-null 规则会取 A（18−0）。这解释了算法差异，没有证明哪个更准确。若给同一窗口全部 start/end logits 加常数，同窗差值不变，跨窗不同的原始 logit 偏移可能改变另一规则；这也是本项目固定同窗比较的机制动机。多个窗口取最大值仍有极值选择偏差，可能提高长文本误接受率，因此不能把单窗口或 SQuAD 上的阈值视为任意长度业务输入的风险保证。

The fixed mechanism is an engineering policy evaluated on an exposed public benchmark. A positive result would demonstrate this bounded system's measured behavior, not a new trained answerability model, independent confirmation of the old quantization fallback, or guaranteed deployment risk. Raw logits do not prove model execution by themselves; model/asset identity and execution provenance require the separate runtime evidence. The audit checks complete run/source/data checksum coverage, reconstructs every stored prediction from its logits, and independently computes all five empirical gates.

The synthetic self-tests cover the counterexample, masked interiors, Unicode trimming, zero-length internal tokens, the inclusive token limit, and rejection of tampered prediction/confidence/offset/window fields. No real evaluation output was read to design these checks.
