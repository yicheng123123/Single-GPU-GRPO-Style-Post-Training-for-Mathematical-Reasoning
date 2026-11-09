# Local packaging audit

Source audit conclusions are preserved in docs/evaluation*.md, limitations.md and provenance.md.
Recomputed: strict Base148/1024, GRPO743/1024; automatic audit
437/1024 versus744/1024, never official accuracy.
109 training metrics, evaluated label105,26,880 rollout responses through105; no new training or inference.

Important fixes/isolation: fresh manifest-bound evaluation, package imports, checkpoint preservation,
and recovery evidence retention. Core11 function bodies and archived grader are preserved.
The original archives and A1 tracked file contents are checked separately in release_checks.json.

Actual CPU test counts/environment are in results/metrics/validation.json; no original31-test suite or
GPU training reproduction is claimed. Optional symbolic regrading could not run without dependencies.
