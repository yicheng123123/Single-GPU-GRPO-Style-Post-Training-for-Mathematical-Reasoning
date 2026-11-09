# Reproducibility: CPU evidence vs GPU execution

## Environment boundaries

Python3.12 is recommended. `requirements/analysis.txt` lists Matplotlib/PyArrow;
`requirements/test.txt` lists pytest; core tests need an existing CPU PyTorch.
No setup step in this preparation installed PyTorch, Transformers, vLLM or a model.
Four optional symbolic-grading dependencies were unavailable; the network installation
was rejected. Stored strict scores and automatic estimates are reaggregated, not silently rescored.

`requirements/course-original.toml` preserves the original course dependencies, including
vLLM0.7.2. `observed-cloud-freeze.txt` records vLLM0.28.0+rocm723, torch2.12.0+git6bbd260,
Transformers5.15.1 and tokenizers0.22.2. It contains system/image packages and is an inventory,
not a portable pip lockfile. Do not install that training stack on Windows or equate CPU
checks with a ROCm training reproduction. A rebuild still needs the original image/wheels.

## Verified CPU entrypoints (from repository root)

| Command | Existing input / dependencies | Output | GPU |
|---|---|---|---|
| `python scripts/project.py check-config` | configs/train_grpo.json; stdlib | static config validation | No |
| `python scripts/project.py doctor` | installed package metadata | version inventory, no GPU probe | No |
| `python -m pytest -q --basetemp=.local/pytest-temp -p no:cacheprovider` | pytest + CPU torch | test results | No |
| `python scripts/project.py report --output .local/reports/recomputed-001` | included records, metrics; stdlib | summary.json, REPORT.md | No |
| `python scripts/project.py figures --output .local/figures/recomputed-001` | included records + Matplotlib | five SVG/PNG figures, hashes | No |
| `python scripts/project.py examples --output .local/examples/recomputed-001` | included records | selected_cases.json, CASES_CN.md | No |

Choose a new output directory on every repeat; report/figure generation refuses to overwrite.
For development tests, pytest's private temporary area can be refreshed by pytest itself.
The copied original course tests are private evidence, not executed by this suite: their
model fixtures can download/load Qwen1.5B and some optional fixture weights are missing.

Data audit requires the existing local Train/Test parquet and old development records:

```sh
python scripts/project.py audit-data --train /path/to/train-00000-of-00001.parquet --test /path/to/test-00000-of-00001.parquet --legacy-dir /path/to/previous/evaluations --output .local/reports/isolation-001.json
```

The public data_isolation.json was computed using the unmodified private archive copies.
The original dataset is not included in Git; metadata and hashes permit checking local copies.

## Future GPU evaluation: NOT executed here

Use an existing compatible ROCm environment and local model/tokenizer directories. No
download fallback is provided. Run the models separately, using the same tokenizer and cases:

```sh
python scripts/project.py evaluate --model /local/base --tokenizer /local/base --label base --run-name base-confirm-001 --cases /local/frozen_cases.json --execute-gpu
python scripts/project.py evaluate --model /local/step105/model --tokenizer /local/base --label grpo-step105 --run-name grpo-confirm-001 --cases /local/frozen_cases.json --execute-gpu
python scripts/project.py compare-runs --left .local/runs/evaluations/base-confirm-001 --right .local/runs/evaluations/grpo-confirm-001 --output .local/reports/confirmed-comparison-001.json
```

Without --cases, the included historical1024 is used; it is not a new confirmation set.
Default decoding: temperature0, top_p1, min_tokens4, max_tokens1024, n1, seed2026, stop
`</answer>` included, BF16, context4096. All settings and hashes are recorded; no existing
JSONL cache is reused. New outputs are under ignored .local/runs/evaluations.
Evaluate/compare help was checked; actual GPU inference remains untested in this preparation.

## Future training and full-state resume: NOT executed here

```sh
python scripts/project.py train --model /local/base --train-parquet /local/math/data/train-00000-of-00001.parquet --run-name controlled-001 --max-steps 105 --execute-training
python scripts/project.py train --model /local/base --train-parquet /local/math/data/train-00000-of-00001.parquet --run-name controlled-001 --max-steps 110 --resume --execute-training
```

Training uses configs/train_grpo.json and writes only .local/runs/training/<name>.
Fresh training requires the Base; resume requires this run's complete model + training_state.pt
and matching configuration. The provided archived Step105 model alone cannot satisfy resume.
Local paths in recorded configuration must still resolve after relocation; arbitrary migration
is not validated. Retain all checkpoints in the maintained runner; provision storage explicitly.
The original disk-rotation/recovery behavior is documented in provenance/trainer_changes.diff.

## Rebuild from local backups (optional, not a download)

```sh
python scripts/import_backups.py --code-archive /local/a5_code_results_backup.tar.gz --model-archive /local/a5_step0105_model.tar --legacy-archive /local/a5_experiment_records_20261106.tar.gz --checksum-file /local/a5_code_results_backup.sha256
```

This imports originals into .local, curates selected evidence and hashes the current model
archive without extracting tensor payloads. It refuses differing existing outputs; it never
overwrites the supplied archives. Optional local grader dependencies are not installed by it.
