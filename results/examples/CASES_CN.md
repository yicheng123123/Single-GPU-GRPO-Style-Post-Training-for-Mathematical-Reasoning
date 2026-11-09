# 五个真实案例

仅展示最终答案区域；长区域明确截短，不据此认定整段CoT正确。

## dataset_index=177

题目：The shortest distance from the circle $x^2 + y^2 = 4x + 8y$ to the point $(5,-2)$ can be written in the form $\sqrt{m}$, where $m$ is an integer. Find $m$.

Ground Truth：`5`

- base 最终区域：$25$
- base 原始评分：`{"reward": 0.0, "answer_reward": 0.0, "format_reward": 1.0}`
- grpo 最终区域：5
- grpo 原始评分：`{"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0}`

严格Reward由0变为1；Base原本格式有效，仍需区分数学错误与评分器表示差异。 来源：results/records/base.jsonl:35、grpo.jsonl:35。

## dataset_index=193

题目：A rectangular patio has an area of $180$ square feet and a perimeter of $54$ feet. What is the length of the diagonal (in feet) squared?

Ground Truth：`369`

- base 最终区域：361
- base 原始评分：`{"reward": 0.0, "answer_reward": 0.0, "format_reward": 1.0}`
- grpo 最终区域：369
- grpo 原始评分：`{"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0}`

严格Reward由0变为1；Base原本格式有效，仍需区分数学错误与评分器表示差异。 来源：results/records/base.jsonl:40、grpo.jsonl:40。

## dataset_index=77

题目：For what value of $x$ will $\frac{2x-1}{2x+2}$ and $\frac{x-3}{x-1}$ be equal?

Ground Truth：`-7`

- base 最终区域：$x = -7$
- base 原始评分：`{"reward": 0.0, "answer_reward": 0.0, "format_reward": 0.0}`
- grpo 最终区域：\(-7\)
- grpo 原始评分：`{"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0}`

Base有明确且匹配标准答案的最终答案，但严格格式失败；属于经过选择的诊断案例，不增加正式准确率。 来源：results/records/base.jsonl:14、grpo.jsonl:14。

## dataset_index=523

题目：Evaluate $\log_{\sqrt{5}} 125\sqrt{5}$.

Ground Truth：`7`

- base 最终区域：7
- base 原始评分：`{"reward": 0.0, "answer_reward": 0.0, "format_reward": 0.0}`
- grpo 最终区域：7
- grpo 原始评分：`{"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0}`

Base有明确且匹配标准答案的最终答案，但严格格式失败；属于经过选择的诊断案例，不增加正式准确率。 来源：results/records/base.jsonl:101、grpo.jsonl:101。

## dataset_index=3155

题目：Find the sum of the smallest and largest prime factors of $10101$.

Ground Truth：`40`

- base 最终区域：$40$
- base 原始评分：`{"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0}`
- grpo 最终区域：26
- grpo 原始评分：`{"reward": 0.0, "answer_reward": 0.0, "format_reward": 1.0}`

GRPO格式有效仍答错，说明格式学习没有消除数学错误。 来源：results/records/base.jsonl:622、grpo.jsonl:622。

