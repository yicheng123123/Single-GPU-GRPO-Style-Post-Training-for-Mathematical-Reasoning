# Selected original implementations / 精选原始实现

These files expose the code used during the project, alongside the maintained package.
All four Python files are **byte-identical** to their archived members. Full SHA256,
archive names and the A/B/C/D selection for all 29 archived Python entries are in
[original_selection.json](../provenance/original_selection.json).

| Published file | Source and purpose | Maintained counterpart |
|---|---|---|
| [adapters.py](adapters.py) | `a5_code_results_backup.tar.gz:tests/adapters.py`; filled SFT and RL primitives under course-provided interfaces | [core.py](../src/grpo_posttraining/core.py); 11 preserved bodies |
| [FP32 SFT trainer](sft/train_sft_a5_fp32.py) | `a5_experiment_records_20261106.tar.gz:train_sft_a5_fp32.py`; complete teacher-forcing training, accumulation, logging and model export | [sft.py](../src/grpo_posttraining/sft.py) |
| [SFT evaluator](sft/eval_sft_math.py) | Same archive, `eval_sft_math.py`; fixed-question sampling, strict reward and generation lengths | Historical protocol evidence; new evaluations use the guarded [evaluation.py](../src/grpo_posttraining/evaluation.py) with their own protocol |
| [Answer audit](grpo/audit_grpo_answers.py) | `a5_code_results_backup.tar.gz:audit_grpo_answers.py`; candidate extraction and mathematical grading | Preserved diagnostic evidence and [analysis.py](../src/grpo_posttraining/analysis.py) |

The [BF16–FP32 diff](sft/bf16_vs_fp32.diff) shows the precision change without duplicating
an almost identical 500-line trainer. Both original versions are retained in the private backup.
The GRPO trainer is already represented by the maintained trainer and its
[source diff](../provenance/trainer_changes.diff); intermediate broken/storage/debug versions
add little review value. Rollout smoke scripts and early incomplete GRPO drafts stay private.
Course tests are inventoried rather than copied into a second test suite requiring model downloads.

原始文件是研究过程的证据，**不是当前推荐的运行入口**：它们依赖旧目录布局，
部分会复用或覆盖输出目录。原始 `adapters.py` 保留五个可选课程函数的
`NotImplementedError`（packed dataset、batch iterator、MMLU/GSM8K parser、DPO）；
不把这些接口计入已完成的核心实现。文件中的部分类型标注、注释也按原样保留。

课程函数接口来自 Stanford CS336 A5，填充实现及单卡实验流程来自本项目工作；
数学评分器、预训练模型与课程模板保留各自归属。参见
[NOTICE](../NOTICE.md)、[许可证说明](../LICENSE_NOTES.md)和[来源说明](../docs/provenance.md)。
原始归档的 Git 元数据为空，文件哈希证明来源字节一致性，不独立认证个人作者身份。
