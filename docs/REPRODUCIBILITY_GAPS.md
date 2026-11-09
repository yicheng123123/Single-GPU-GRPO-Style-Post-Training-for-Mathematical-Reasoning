# Reproducibility gaps

- Historical Base/evaluation hashes unavailable; current hashes are not retroactive proof.
- Archived optimizer/RNG state absent. Full cloud checkpoint must be independently preserved for exact resume.
- Original Git commit/status/patch files empty. Code lineage is archive/segment based.
- Old actual CLI, tokenizer revision, raw token IDs and stop reasons not saved.
- Actual ROCm environment differs from the course pyproject; original image/wheel provenance is missing.
- New GPU wrappers not executed; no bitwise-resume equivalence test is claimed.
- Optional symbolic dependencies unavailable locally; network installation rejected, no workaround attempted.
- Six historical development questions are in the1024 subset. A new confirmation set must exclude past use.

Repair future evidence collection without changing historical JSONL or substituting newly computed scores for original reward.
