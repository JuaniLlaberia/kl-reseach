# Experiments Log

## 05/11/26 - Start of Research

Winogrande shots:
- What: Winogrande runs 5-shot in the retention suite.
- Why: Matches MMLU's 5-shot. Chosen before seeing any scores.
- Affects: all retention runs, starting with the base run.

KL estimator samples:
- What: k1, k2 and k3 use r = π₀(y)/π(y) with y sampled from the fine-tuned model π, so they estimate KL(π ‖ π₀). Fine-tuned responses are generated per checkpoint, in addition to the base responses saved once.
- Why: With this definition, the estimators are only valid on samples from π. On base samples, the k1 formula would average to −KL(π₀ ‖ π).
- Affects: the KL module's sampler and the k1 test, which compares against exact KL(π ‖ π₀).

KL variants and tokens:
- What: Exact forward and reverse KL are computed on both base and fine-tuned samples, named by estimator, direction, source and aggregation. Average per token is the mean of per-sequence means. Sum per sequence is the mean across prompts of each response's total. The first stop token counts as a response token. Responses that hit the cap are kept and flagged.
- Why: All four combinations let direction and sample source be compared separately, at no extra compute; only forward on base and reverse on fine-tuned are true sequence-level KLs. The mean of means weights each prompt equally, matching E_{x∼τ}. Stopping is part of the response distribution. Dropping capped responses would bias against long ones.
- Affects: every KL variant from Stage 1 on. Chosen before computing any KL.

KL sampling:
- What: Responses use `generate` with temperature=1.0, top_p=1.0, top_k=0 and repetition_penalty=1.0 all explicit, plus a check that processed logits equal raw logits. Batch size and order are fixed, with the seed derived from the batch index. Each family uses its default system prompt. max_new_tokens is 512 for Stage 1; Stage 2 will measure it and may set it per task.
- Why: `generate` inherits Qwen's top_k=20, top_p=0.8, temperature=0.7 and repetition_penalty=1.1 unless each is overridden, even when passing None (tested with transformers 5.22.1 on a tiny Qwen2 model). A batch-level seed allows resuming after a Colab disconnect.
- Affects: all base and fine-tuned responses.

KL storage:
- What: Each checkpoint gets a per-sequence file with each response's KL total and length per variant, alongside the results JSON.
- Why: Allows re-aggregation and sample-size variants without the checkpoints, which Stage 2 may delete.
- Affects: the `results/` layout in CONTEXT.md.

KL precision:
- What: Response sampling and KL forward passes both run in fp32. Checkpoints are loaded as fp32 for measurement. bf16 is only used in the KL module for the drift test.
- Why: KL measures small differences between models, so logits shouldn't be rounded first, and samples should come from the same distribution being measured. Costs more memory and compute per checkpoint.
- Affects: all KL measurement. May be revisited before the Nov 17 freeze, on the drift result alone and never on fit.

KL precision safeguard:
- What: `measure_kl` refuses non-fp32 models unless the drift test asks for bf16, and refuses responses not sampled in fp32. `save_kl_results` only saves fp32 values and records `kl_dtype` in the results JSON. bf16 values aren't stored per checkpoint; the drift is measured once and recorded in this log.
- Why: A bf16 run would otherwise produce the same variant names as fp32 and could overwrite reported results. Storing both for every checkpoint would double the compute for numbers no analysis uses.
- Affects: the results JSON gains the `kl_dtype` field.
