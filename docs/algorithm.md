# 中文算法说明：组相对优势的 REINFORCE

## 1. 数学目标

SFT 通过参考响应最大化条件概率，通常写作

$$L_{SFT}=-\sum_t m_t\log\pi_\theta(y_t^{ref}\mid x,y_{<t}^{ref}).$$

RL 则由当前策略采样响应，用终局奖励来更新策略。基础 REINFORCE 的方向为

$$\nabla J=\mathbb E[R\nabla_\theta\log\pi_\theta(y\mid x)].$$

本实验采用组内中心化、标准化及响应长度归一化，因此实际优化的是下述 surrogate，
不能把它说成未经任何变换的原始期望奖励无偏估计。

同一问题采样 G=8 条响应，严格格式门控奖励为 r：

$$\bar r_g=\frac1G\sum_i r_{g,i},\quad A_{g,i}=\frac{r_{g,i}-\bar r_g}{s_g+10^{-6}}.$$

源码使用 `torch.std` 的默认估计。高于组平均的响应得到正 Advantage，低于平均则为负；
全组奖励相同，分子为零，此组无优势梯度。每条响应的同一个 Advantage 广播到所有有效响应 Token。

$$L=-\frac1B\sum_i A_i\frac{\sum_t m_{i,t}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}{\sum_t m_{i,t}}.$$

`m` 与右移后的 labels 对齐：prompt 和 padding 不参与 loss。这里分母是每条响应的有效长度，
不是全 batch Token 平均，也不是 Dr.GRPO 的固定长度归一化。

## 2. 实际调用与维度

log_probs/mask: `[microbatch, sequence_length]`；advantage: `[microbatch, 1]`。
32 prompts × 8 = 256 responses；128 microbatches × 2 = 256。
每个 microbatch 先对响应求均值再除128，梯度累计后等价于这256条响应的平均。
所有 backward 后 clip norm=1，随后仅一次 AdamW step；lr=1e-5、betas=(0.9,0.95)、decay=0。

## 3. On-policy 和停止边界

每批由更新前的当前策略生成，只更新一次，下一批重新导出权重再生成。
但 BF16 导出/FP32 主参数、推理后端差异、min_tokens=4、stop string 和文本重新编码意味着
严格动作分布一致性尚未量化。旧数据只存文本，没有采样log-prob与原始Token IDs。
不能盲目把 SFT 的“额外追加EOS监督”搬进RL：应先检查真实采样动作与训练序列是否一致。

## 4. 与 PPO / GRPO-Clip 的区别

实际 trainer 使用 `reinforce_with_baseline`，old_log_probs 与 cliprange 都为None，未启用比值clipping，
没有KL penalty。core中保留并测试了clip helper，但它不是该成绩对应的训练目标。

## 5. 为什么训练Reward不是泛化

每步题目和响应变化，格式与长度也变化；训练Reward上升只能说明采样训练批次得分上升。
需要固定的评测协议、独立确认题集和多seed，才能进一步判断泛化与算法差异。

## 6. 源码映射

以下函数位于 `src/grpo_posttraining/core.py`；原文件为代码归档的 `tests/adapters.py`。

| 函数 | 原始 adapters 行号 | 新文件行号 |
|---|---:|---:|
| `run_tokenize_prompt_and_output` | 12 | 14 |
| `run_compute_group_normalized_rewards` | 49 | 47 |
| `run_compute_entropy` | 88 | 85 |
| `run_get_response_log_probs` | 96 | 91 |
| `run_compute_naive_policy_gradient_loss` | 135 | 129 |
| `run_compute_grpo_clip_loss` | 153 | 147 |
| `run_compute_policy_gradient_loss` | 187 | 180 |
| `run_masked_mean` | 226 | 218 |
| `run_sft_microbatch_train_step` | 251 | 242 |
| `run_grpo_microbatch_train_step` | 266 | 256 |
| `run_masked_normalize` | 319 | 308 |

原trainer的关键位置：303–344行计算优势和loss，318–364行累计与更新，383–386行导出及checkpoint。
