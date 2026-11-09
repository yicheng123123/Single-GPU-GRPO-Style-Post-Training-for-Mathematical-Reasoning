# Evaluation protocol and interpretation

## Preserved historical evidence

| Metric on the same 1024 questions | Base | GRPO-style Step 105 |
|---|---:|---:|
| Strict R1-Zero reward (format gated) | 148/1024 (14.45%) | 743/1024 (72.56%) |
| Format compliance | 39.94% | 90.82% |
| Supplementary automatic extraction estimate — **not official accuracy** | 42.68% | 72.66% |

Results are recomputed from stored scores in `results/records/base.jsonl` and `grpo.jsonl`.
All case IDs, question text, gold answers, and literal format gates are checked.
This reaggregation is not a fresh execution of the full symbolic grader.
Automatic recovery candidates are preserved, not relabeled as human-reviewed decisions.

The archived `eval_grpo_a5.py:61–90` uses the same prompt, grader, engine and sampling
implementation for both models. Defaults at lines137–138 are temperature0 and1024 tokens;
min_tokens4, n1, top_p1, stop `</answer>` with the stop included. Actual historical CLI,
model hashes and tokenizer revision were not saved, so operational identity is unverified.

## Dataset isolation

See [computed data checks](../results/metrics/data_isolation.json): 7500 Train / 5000 Test;
4 unparseable training golds excluded. No exact or whitespace-normalized Train overlap in
the evaluated subset. Full Test has one whitespace-normalized duplicate: Train2948/Test4656
(training text has a trailing newline). It is absent from this subset.
Six evaluated questions were previously used in the historical SFT development evaluation:
297,716,1383,1778,3883,4516. No claim of an entirely unseen final confirmation set is made.
Checks cover text equality, not semantic duplicates or pretraining decontamination.

## New safe evaluation

The maintained entry is `scripts/project.py evaluate`. It takes an explicit local model,
shared local tokenizer, explicit label and unique run-name. It refuses existing output;
it never reads latest_checkpoint to relabel old JSONL. A manifest hashes model files,
tokenizer, cases, prompt and grader, and records sampling and package versions.
Raw continuation IDs, token counts, finish_reason/stop_reason are saved when vLLM exposes them.
Model bytes are rechecked after generation. `compare-runs` rejects protocol/environment mismatches.
These new GPU paths were not executed during local preparation.

## Caveats

Strict +58.11pp includes format learning. Supplementary +29.98pp is post-hoc,
not an isolated reasoning improvement or a fully human-reviewed accuracy. The extraction
audit tries multiple candidates; a gold-matching candidate does not prove the complete
response is unambiguous or its reasoning correct. The 94 missing GRPO answer closings are
not a measured truncation count because old token/stop metadata is unavailable.
See [Chinese examples](../results/examples/CASES_CN.md) and [limitations](limitations.md).
