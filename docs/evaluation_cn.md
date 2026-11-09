# 中文评测与格式偏差审计

## 三种数值不能混用

| Metric on the same 1024 questions | Base | GRPO-style Step 105 |
|---|---:|---:|
| Strict R1-Zero reward (format gated) | 148/1024 (14.45%) | 743/1024 (72.56%) |
| Format compliance | 39.94% | 90.82% |
| Supplementary automatic extraction estimate — **not official accuracy** | 42.68% | 72.66% |

严格Reward要求字面出现 `</think> <answer>` 以及 `</answer>`，再用课程评分器判断答案。
format失败时answer_reward也为0，因此answer_reward不是独立的纯数学准确率。
源码：`src/grpo_posttraining/grader.py:1008–1046`，与原归档字节相同。

同题配对：双方都对142、Base对GRPO错6、Base错GRPO对601、双方都错275。
601个收益案例中427个Base原本格式无效；不能把所有收益归为数学能力，也不能说这427题都只是格式问题。

## 补充提取的边界

恢复候选来自历史自动脚本，不是人工裁决。算法会尝试answer标签、boxed、显式final-answer cue等，
多个候选之一匹配就可能计入估计，因此有候选择优和歧义风险。不能随意取最后一个数字。
文件 `automatic_recovery_candidates.jsonl` 的名称刻意不再暗示已完成人工复核。

原审计还发现未约分分数的数值等价与严格规范差异：Base题113的703/1110与19/30相等，
GRPO题1447的902/1617与82/147相等；评分器961–969行明确对分数形式严格比较。
没有因此修改任何历史Reward或增加总体准确率。这些个案不证明提取估计是数学正确率的上下界。
本次整理缺少本地math-verify/latex依赖，未重跑完整符号评分；依赖安装被审批阻止，未绕过。

## 数据与确认集

训练7500条，排除4条无boxed标准答案后7496；测试5000条；1024题题面/GT与Parquet均匹配。
逐字全量Train/Test重复为0，忽略空白后Train2948/Test4656重复；当前评测不含此题。
当前1024题有6道历史开发题：297、716、1383、1778、3883、4516，因此不是完全盲确认集。
来源及按题型/难度统计见 [data_isolation.json](../results/metrics/data_isolation.json)。

## 错误案例与长度

见 [五个案例](../results/examples/CASES_CN.md)：两道Base数值错误到GRPO正确、两道Base正确但格式失败、一道GRPO仍错。
只展示最终答案区域，不以推理开头认定完整CoT正确。Base/GRPO中位字符数分别564和1168.5；
字符数不等于Token数。缺少结束标签也不能直接算1024-token截断。

## 独立新评测

新的评测入口禁止复用输出目录，显式固定Tokenizer、模型文件hash、Prompt/题集/评分器hash与采样配置。
正式严格Reward仍用原评分器，新增stop元数据只作观测。不对旧记录反填未知EOS状态。
详见 [英文协议](evaluation.md) 与 [运行命令](reproducibility.md)。
