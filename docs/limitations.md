# Limitations and unresolved reproducibility gaps

1. Current hashes describe current bytes, not historical Base or evaluated checkpoint identity.
   The original Base path used a mutable snapshot name; original git metadata are empty.
2. The local Step105 archive lacks optimizer/RNG state. Cloud logs record resume attempts,
   but local bitwise resume equivalence has not been tested.
3. 109 logged updates do not mean a 109-step evaluated model. Preserved evaluation label=105.
4. Strict reward is format gated; +58.11pp is not a pure reasoning gain.
5. The supplementary extraction estimate is automatic, post-hoc and not fully human reviewed.
   Trying several candidates can increase recall and ambiguity. Unreduced fractions can remain strict failures.
6. The1024 subset overlaps six older development questions; text decontamination does not cover pretraining or semantic duplicates.
7. Old raw Token IDs/counts/finish reasons and sampler log-probs were not preserved.
   Exact behavior-policy matching and historical truncation/EOS statistics cannot be recovered.
8. GPU type follows user experiment records; no hardware inventory or new GPU measurements were taken here.
   No invented utilization, memory peak, throughput or speedup is reported.
9. New GPU wrappers are CPU-reviewed only. The original course tests are not claimed rerun;
   new CPU tests cover primitives, evidence invariants and publication/entrypoint safety.
10. Full symbolic regrading was not run because local optional dependencies are unavailable and their installation was rejected.
11. Expert Iteration, full A5 writeup, matched algorithm ablations and optional DPO/safety completion are not evidenced here.
12. This is terminal-reward single-turn mathematical generation. No tools, interactive environment,
    multi-turn trajectories or tool-using Agent RL results are implemented.

Next priority: immutable identity/full-state backup, then preregistered confirmation evaluation,
then a group-standard-deviation normalization ablation. Existing step timers exclude export/checkpoint overhead.
