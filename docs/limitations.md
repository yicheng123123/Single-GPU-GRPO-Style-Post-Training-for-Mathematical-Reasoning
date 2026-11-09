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
7. RL raw token IDs/counts/finish reasons and sampler log-probs were not preserved.
   Exact behavior-policy matching and historical truncation/EOS statistics cannot be recovered.
8. GPU type follows user experiment records; no hardware inventory or new GPU measurements were taken here.
   No new GPU utilization, memory or speedup measurements were taken. SFT memory and
   throughput numbers are recomputed from the historical training logs, with their timer scope stated.
9. New GPU wrappers are CPU-reviewed only. The original course tests are not claimed rerun;
   new CPU tests cover primitives, evidence invariants and publication/entrypoint safety.
10. Full symbolic regrading was not run because local optional dependencies are unavailable and their installation was rejected.
11. Expert Iteration, full A5 writeup, matched algorithm ablations and optional DPO/safety completion are not evidenced here.
12. This is terminal-reward single-turn mathematical generation. No tools, interactive environment,
    multi-turn trajectories or tool-using Agent RL results are implemented.
13. SFT's archived 50-question development evaluation uses temperature 1.0, Hugging Face
    FP32 parameters/BF16 autocast; RL's fixed1024 uses greedy BF16 vLLM. These are not a matched ranking.
    SFT token counts were saved but raw IDs/finish reasons were not, so cap counts do not identify EOS.
14. SFT weight/optimizer files are absent from the evidence archive. Its R1-derived substitute
    training set has no authenticated upstream revision/license chain; only statistics/hashes are public.
    91/128 responses exceed the1024 evaluation budget, with zero EOS targets. This motivates,
    rather than proves, a termination explanation; one precision run per arm cannot isolate causality.

Next priority: immutable identity/full-state backup, then preregistered confirmation evaluation,
then a group-standard-deviation normalization ablation. Existing step timers exclude export/checkpoint overhead.
