# Resume evidence and role boundaries

Suggested claims, subject to the owner's actual contribution and provenance review:

0. Implemented teacher-forced SFT with response masking, gathered token NLL, four-step gradient
   accumulation and BF16/FP32 parameter modes. Completed two 128-trace, 32-update runs;
   measured FP32 token-weighted training NLL 0.6674 and 3/50 strict development successes,
   while identifying 44/50 budget hits. Evidence: `docs/sft_experiments.md`, raw metrics and paired outputs.
1. Built a single-GPU mathematical post-training workflow integrating PyTorch backward and serialized vLLM rollout,
   with256 responses/step, group8, accumulation128 and FP32/BF16 policy synchronization.
2. Evaluated the Step105-labelled policy on a fixed1024-question MATH Test subset: strict format-gated reward
   72.56% versus 14.45% Base; paired601 gains/6 losses; explicitly audited format confounding.
3. Packaged CPU-reproducible response-level analysis, source/model manifests, bilingual documentation and seven data-derived plots,
   while distinguishing109 logged updates from the105 evaluated checkpoint and model-only/full-state recovery.

SFT and RL are independent Base-initialized experiments. Do not describe the RL result as
an SFT-to-RL continuation or compare the separate 50/1024-question protocols as one benchmark.
The maintained SFT module has CPU test coverage; historical GPU runs used the archived scripts.

LLM post-training engineering evidence is stronger than tool-using Agent RL evidence: this project has no
interactive tool environment, multi-turn credit assignment or Agent task-success measurements. It is not full PPO/GRPO-Clip.

Next controlled ablation: group-standard-deviation normalization on/off, matched Base/data/seed/update budget,
fresh on-policy samples per arm and a preregistered evaluation. Clipping ablation has low immediate information value:
the observed preclip norm never exceeded1 through105. Independent500-question confirmation evaluation should
exclude all previous development use and normalized duplicates; its duration needs a new measured inference pilot.
