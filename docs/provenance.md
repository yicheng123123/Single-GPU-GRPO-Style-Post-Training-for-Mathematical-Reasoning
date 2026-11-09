# Provenance and preservation

Original source archives remain unchanged outside the repository. Only basenames and hashes
are public in provenance/source_archives.json; private user/cloud paths remain ignored.
Original code, logs, datasets and intermediate revisions are preserved in .local/originals.
No model weight tensor was extracted into the new repository or loaded into a framework.

## Classification

- Course interfaces/templates/prompts: Stanford CS336 Spring2025 A5. Exact historical upstream commit unknown.
- Project additions: SFT/GRPO training and evaluation scripts and filled core adapters from the experimental archive.
  Archive git metadata are empty; personal authorship is not independently authenticated.
- Third-party grader: src/grpo_posttraining/grader.py retains its original Apache2 header and source attribution.
- Pretrained model/tokenizer: Qwen; no claim of having implemented or pretrained the underlying LM.
- New packaging work: safe evaluation entry, source manifests, CPU tests, analysis/plots and bilingual documentation.
  These additions do not retroactively change the historical algorithm or results.

## Code preservation

11 core function source segments and ASTs are unchanged, mapped in provenance/core_function_map.json.
The grader is byte-identical to cs336_alignment/drgrpo_grader.py in the archive.
Text-only answer helpers are extracted unchanged to avoid importing the full symbolic stack for data checks.

The archived trainer is adapted only for package imports/worker invocation, a guarded CLI,
project-local output scope, preserved checkpoints and recovery evidence. Algorithm/loss settings
remain the archived baseline. The full diff is provenance/trainer_changes.diff; new GPU behavior is unvalidated.
Unsafe legacy evaluation/cache reuse is isolated in ignored originals; the new evaluator never reuses result directories.
Partly implemented optional course adapters are omitted from the public core, not claimed completed.

## Record transformations

Training metrics, Base/GRPO JSONL and supplementary automatic candidates are byte-identical copies.
cases.json and historical_comparison.json replace private absolute paths with portable labels;
question text, IDs, golds and scores are unchanged. The old 'manual_review' filename is renamed
to automatic_recovery_candidates to avoid implying completed human review.

## Identity limits

provenance/current_model_inventory.json records archive/member hashes,338 F32 tensors and
1,543,714,304 stored elements. It cannot prove which bytes historical vLLM evaluation loaded.
Model-only archive identity and full-state resume identity are separate. Future evaluate runs
record their actual files and protocol rather than trusting a step name or latest pointer.

## Licenses checked on 2026-11-09

Spring2025 course branch master (observed SHA a88071d1272112306f59257d946e43b56bb31773)
has a Stanford permission license, copied in LICENSES/Stanford-CS336-LICENSE.txt.
The moving main branch now describes Spring2026; its missing root LICENSE is not used to
infer the2025 license. See LICENSE_NOTES.md for per-file notices and unresolved authorship/mirror provenance.
