# SFT implementation / SFT 实现

正式入口是 [sft.py](../src/grpo_posttraining/sft.py)，CLI 为 `sft-train`。
原始训练器实际运行过；整理后的入口只做了 CPU 测试，本次没有执行 GPU 训练。
[源码映射](../provenance/sft_source_map.json)关联原始行号、正式函数和工程改动。

## Data and teacher forcing

每条样本包含完整 `prompt` 与 `response`。`load_records` 校验非空字符串；
`collate_records` 使用共享的 `run_tokenize_prompt_and_output`，分别编码两段文本，
`add_special_tokens=False`，拼接后按 next-token prediction 对齐：

$$
x=(p_1,\ldots,p_m,r_1,\ldots,r_{n-1}),\qquad
y=(p_2,\ldots,p_m,r_1,\ldots,r_n).
$$

`response_mask` 对齐 **labels**；第一枚 response token 由最后一枚 prompt token 位置预测。
Prompt 与 padding 不参与直接 loss，prompt 仍通过上下文影响 response 的梯度。
Padding 按位置排除；历史 microbatch 为一条样本，因此没有跨样本 padding 的注意力问题。
超长样本报错，不截断；保持历史数据正文与 EOS 行为，不新增 EOS。
训练数据统计显示 128 条均有完整标签，EOS 监督目标共 0 个。

## Objective and normalization

共享 `run_get_response_log_probs` 使用完整真实前缀做 teacher forcing，再以 labels gather：

$$
\ell_{b,t}=\log p_\theta(y_{b,t}\mid x_{b,\leq t}),\qquad
L_{\mathrm{micro}}=-\frac{1}{KB}\sum_b\sum_t m_{b,t}\ell_{b,t}.
$$

历史参数 `normalize_constant=1.0`，microbatch \(B=1\)，累积 \(K=4\)。
四次 backward 后执行一次 optimizer update，相当于对四条 response 的 **NLL 总和取均值**，
不是逐条按长度归一化。长 response 的权重更大。日志另算
`token_nll = summed_response_nll / response_tokens`；这是诊断指标，不能偷换成实际优化目标。
SFT 与 RL 共享 tokenization、log probabilities、masking 和 backward primitives；
RL 的实际 loss 使用每条 response 的有效长度均值，两种 normalization 不相同。

## Precision and optimization

| Item | Historical BF16 arm | Historical FP32 arm |
|---|---|---|
| Trainable parameters | BF16 | FP32 |
| Forward | BF16 model, no autocast context | BF16 autocast |
| Gathered log probabilities | Kept at forward dtype | Cast to FP32 after gather |
| Optimizer | AdamW default betas `(0.9,0.999)`, weight decay 0 | Same |
| LR / clip | `1e-5` / global norm 1.0 | Same |
| Microbatch / accumulation | 1 / 4 | Same |
| Epochs / updates / seed | 1 / 32 / 42 | Same |

FP32 cast 在 log-softmax/gather **之后**，不等于 FP32 logits/log-softmax。
BF16 参数方案的优化器状态随 PyTorch 的参数 dtype 分配；不是 FP32 master-weight 训练。
两组都使用 SDPA、non-reentrant gradient checkpointing 和 `use_cache=False`。
历史数据长度、顺序相关 token 计数吻合，但单次对照不足以把效果差异归因到权重精度。

正式 `train_loop` 保留这些语义，分离可 CPU 测试的 microbatch/update 流程。
新增本地加载限制、新目录输出、模型/数据/源码哈希、输入检查和非有限值检查。
这些安全检查不更改历史结果。保存 HF 模型与 tokenizer；**SFT 没有 optimizer/RNG resume**。
完整状态恢复是 RL trainer 的能力，不能挪用到 SFT 上。

## Evaluation boundary

[历史 SFT evaluator](../original_implementation/sft/eval_sft_math.py)对三个模型使用相同
Base tokenizer、R1-Zero prompt、50 题与每题 seed，FP32 参数加载加 BF16 autocast，
temperature 1.0、top-p 1、top-k 0、1024 token 预算、`</answer>` stop string。
默认 generation config 中 EOS 为 151643；原始记录只存生成长度和 decode 后文本，
不能还原 EOS/stop string/预算停止的精确原因。达到上限仅按 `generated_tokens >= 1024` 统计。

新 `evaluate` 是 vLLM 固定协议入口，**不是历史 HF SFT evaluator 的数值等价复现**。
若将来比较新 SFT 与新 RL，必须在共同、独立的确认集及统一解码协议下重新评测，
输出归档 hashes、raw token IDs、停止原因，再做逐题配对。此次仅重聚合历史分数，未生成答案。
