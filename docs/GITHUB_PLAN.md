# Publication checklist

The target is the existing public repository configured as local origin. No remote writes are authorized yet.

Before publication, the owner should confirm:

1. Publish the reviewed bilingual source/evidence snapshot to the specified public repository.
2. Review course/data/grader attribution and choose licensing for new project code.
3. Accept the documented historical-identity and model-only-resume limitations.
4. Confirm inclusion of selected historical question/response records, with no model/dataset archive.
5. Provide a GitHub write-capable authentication method if the connector remains read-only; never share credentials in chat.

The local release excludes .local, all model/checkpoint/dataset binaries, caches and private paths.
Raw training datasets and the6GB model package are not committed and Git LFS is not used.
No reset/force-push, A1 repository modification, remote visibility change or overwrite is part of this task.
See results/metrics/release_checks.json for the actual file/secret/parent-integrity verification.
