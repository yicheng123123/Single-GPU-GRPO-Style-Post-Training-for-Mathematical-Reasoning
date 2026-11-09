"""Import local experimental evidence without modifying archives or extracting weights.

Private source files remain under .local; publication uses separately curated files.
No model framework is imported. Existing differing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, data):
    path = ROOT / path
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Output escapes project")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(path)
        return
    with path.open("xb") as handle:
        handle.write(data)


def save_json(path, value):
    save(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def safe_members(archive):
    for member in archive:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
            raise ValueError("Unsafe archive member: " + member.name)
        if member.isfile():
            yield member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--legacy-archive", type=Path, required=True)
    parser.add_argument("--checksum-file", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.code_archive, args.model_archive, args.legacy_archive, args.checksum_file):
        if not path.is_file():
            parser.error("Missing local input: " + str(path))
    code_sha = digest(args.code_archive)
    if code_sha != args.checksum_file.read_text().split()[0].lower():
        raise ValueError("Code archive checksum does not match sidecar")
    entries = {}
    with tarfile.open(args.code_archive, "r:gz") as archive:
        for member in safe_members(archive):
            if member.size > 100 * 1024 * 1024:
                raise ValueError("Unexpected large code archive member")
            data = archive.extractfile(member).read()
            entries[member.name] = data
            save(Path(".local/originals/a5") / member.name, data)

    records = {
        "experiments/grpo_single_gpu/metrics.jsonl": "results/metrics/train_metrics.jsonl",
        "experiments/grpo_eval_step_0105/base.jsonl": "results/records/base.jsonl",
        "experiments/grpo_eval_step_0105/grpo.jsonl": "results/records/grpo.jsonl",
        "experiments/grpo_eval_step_0105/format_free_audit.json": "results/metrics/supplementary_audit.json",
        "experiments/grpo_eval_step_0105/format_free_manual_review.jsonl": "results/records/automatic_recovery_candidates.jsonl",
    }
    lineage = []
    for original, destination in records.items():
        save(destination, entries[original])
        lineage.append({"archive_member": original, "destination": destination,
                        "original_sha256": hashlib.sha256(entries[original]).hexdigest(),
                        "transformation": "byte-identical; review candidates are NOT human decisions"})
    cases = json.loads(entries["experiments/grpo_eval_step_0105/cases.json"])
    cases["test_path"] = "<LOCAL_MATH_TEST_PARQUET>"
    save_json("results/records/cases.json", cases)
    comparison = json.loads(entries["experiments/grpo_eval_step_0105/comparison.json"])
    comparison["cases_file"] = "results/records/cases.json"
    save_json("results/metrics/historical_comparison.json", comparison)
    run_config = json.loads(entries["experiments/grpo_single_gpu/train_config.json"])
    run_config.update(model="<LOCAL_BASE_MODEL>", train_parquet="<LOCAL_MATH_TRAIN_PARQUET>",
                      prompt_template="src/grpo_posttraining/prompts/r1_zero.prompt")
    save_json("configs/historical_train.sanitized.json", run_config)
    save("requirements/observed-cloud-freeze.txt", entries["a5_package_versions.txt"])
    save("requirements/course-original.toml", entries["pyproject.toml"])
    save("requirements/original-rocm-additions.txt", entries["requirements-a5-rocm.txt"])
    for name in ("train_grpo_a5.py", "tests/adapters.py", "eval_grpo_a5.py",
                 "audit_grpo_answers.py", "cs336_alignment/drgrpo_grader.py",
                 "cs336_alignment/prompts/r1_zero.prompt"):
        lineage.append({"archive_member": name, "original_sha256": hashlib.sha256(entries[name]).hexdigest(),
                        "private_original": ".local/originals/a5/" + name})

    with tarfile.open(args.legacy_archive, "r:gz") as archive:
        for member in safe_members(archive):
            if member.name.startswith("evaluations/") and member.name.endswith("/base.jsonl"):
                save(Path(".local/originals/legacy") / member.name, archive.extractfile(member).read())

    model_members = []
    tensor_summary = None
    config = None
    with tarfile.open(args.model_archive, "r:") as archive:
        for member in safe_members(archive):
            handle = archive.extractfile(member)
            sha = hashlib.sha256()
            if member.name.endswith("model.safetensors"):
                first = handle.read(8)
                header_length = struct.unpack("<Q", first)[0]
                if header_length > 10 * 1024 * 1024:
                    raise ValueError("Oversized tensor header")
                header_bytes = handle.read(header_length)
                sha.update(first + header_bytes)
                tensors = json.loads(header_bytes)
                tensors.pop("__metadata__", None)
                elements = 0
                for value in tensors.values():
                    count = 1
                    for dim in value["shape"]:
                        count *= dim
                    elements += count
                tensor_summary = {"tensor_count": len(tensors), "stored_elements": elements,
                                  "dtypes": dict(Counter(v["dtype"] for v in tensors.values()))}
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    sha.update(block)
            else:
                data = handle.read()
                sha.update(data)
                save(Path(".local/model_metadata") / Path(member.name).name, data)
                if member.name.endswith("/config.json"):
                    config = json.loads(data)
            model_members.append({"member": member.name, "bytes": member.size, "sha256": sha.hexdigest()})
    save_json("provenance/current_model_inventory.json", {
        "archive": args.model_archive.name, "archive_sha256": digest(args.model_archive),
        "members": model_members, "tensors": tensor_summary,
        "model_architecture": {k: config.get(k) for k in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size", "eos_token_id")},
        "historical_evaluation_weight_identity": "UNVERIFIED: no contemporaneous weight hash",
        "exact_resume_from_this_archive": False,
        "missing_training_state": ["optimizer state", "CPU RNG", "GPU RNG"],
        "scope": "Current local bytes only; no model loaded or inference attempted",
    })
    save_json("provenance/source_archives.json", {
        "code": {"file": args.code_archive.name, "sha256": code_sha, "sidecar_match": True,
                 "regular_files": len(entries)},
        "legacy": {"file": args.legacy_archive.name, "sha256": digest(args.legacy_archive)},
        "upstream_commit": "UNKNOWN: archived git metadata files are empty",
    })
    save_json("provenance/source_lineage.json", lineage)
    print(json.dumps({"private_archive_files": len(entries), "curated_records": len(records),
                      "model_tensors": tensor_summary, "GPU_used": False}))


if __name__ == "__main__":
    main()
