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

## 06/11/26 - End of Stage 1

GPU check:
- What: The A100 wasn't available on Colab, so the choice was the L4 (1.94 units per hour) or the G4 (RTX PRO 6000 Blackwell, 10 units per hour). The official base run uses the G4. On the same first eval step and batch size, the L4 took 14:32 (about 0.47 units) and the G4 took 2:52 (about 0.48 units). The L4 ran out of memory at batch 8 in fp32. Kaggle won't be used because it doesn't support bf16.
- Why: The two GPUs cost about the same per job on that one step, and the G4 is about 5x faster and fits batch 8. The training GPU is still decided from the Stage 2 timings.
- Affects: all retention runs. Stage 3's GRPO runs can no longer use Kaggle, so all compute comes from Colab's budget. A checkpoint run on the L4 needs a smaller batch and a score check first.

Retention setup:
- What: The suite is hellaswag, truthfulqa_mc2, winogrande (5-shot), mmlu (5-shot), ifeval and humaneval_instruct, each run separately. It runs in fp32 with the chat template on, batch size 8 and repetition_penalty=1.0, with the model pinned to commit 7ae5576.
- Why: Plain humaneval isn't valid with a chat template: in 3 of 3 samples the response started with prose, which gets appended to the code. Repetition penalty is explicit because generate() would otherwise inherit each model's own value (Qwen ships 1.1). Running benchmarks separately gives minutes per benchmark. Chosen before seeing any full-run scores.
- Affects: every retention run. Earlier L4 and G4 runs are discarded: they weren't on the pinned versions, and some used Winogrande 0-shot by mistake.

Base retention results:
- What: Qwen2.5-0.5B-Instruct, scores with 95% bootstrap CI and minutes: hellaswag acc 0.3971 (0.3878-0.4067) and acc_norm 0.4830 (0.4731-0.4928), 2.5 min; truthfulqa_mc2 0.4535 (0.4242-0.4826), 0.6 min; winogrande 0.5478 (0.5201-0.5754), 0.3 min; mmlu 0.4595 (0.4512-0.4680), 7.0 min; ifeval strict 0.2588 (0.2218-0.2976) and loose 0.2810 (0.2440-0.3198), 11.8 min; humaneval_instruct 0.3354 (0.2622-0.4085), 1.5 min.
- Why: Stage 1 baseline for measuring forgetting. TruthfulQA's CI includes its uniform chance level of 0.4484, the mean share of true answers per question. Winogrande is 0.048 above 0.50. HellaSwag and MMLU are clearly above 0.25. IFEval has no simple chance level. No benchmark is dropped; the near-chance rule is set in Stage 2.
- Affects: all forgetting calculations and the Stage 2 near-chance rule. The suite takes 23.7 minutes, about 4 G4 units per checkpoint.

Results provenance:
- What: results/qwen05_base/base.json records git_commit b3e8823, but the notebook that produced it was only committed afterwards, in the "Upload stage 1 and results" commits.
- Why: The official run started before the notebook was committed. The run wasn't repeated because the notebook is committed as it ran, and the base run has no KL code in it.
- Affects: this base run only. Later runs must be started from a committed notebook.

bf16 drift:
- What: Measured on Qwen with synthetic weight noise, 100 MMLU prompts and a 512-token cap. Exact KL per token: 0.01 noise gives KL about 0.007, with bf16 11-13% higher in every variant; 0.05 noise gives KL about 0.27-0.38, with under 1% difference. k3 at 0.05 noise went from 0.35 to 10.8 in bf16, consistent with one rare token with r around 68,000 (inferred, not checked per sequence). k2 at 0.05 noise is about 1.9x the exact KL, in fp32 too.
- Why: KL stays in fp32, because bf16 inflates small KL values. k3 isn't dropped; its fragility goes in the Stage 2 predictions.
- Affects: all KL measurement. Repeat on a real smoke-run checkpoint in Stage 2, before the freeze. The noise is synthetic, and it also rounds the noisy weights to bf16.

KL tests:
- What: All tests in tests/ passed, run by Juan.
- Why: Required to close Stage 1.
- Affects: nothing changes; the KL module is ready for Stage 2.

Pilot model decision rule:
- What: The candidates are Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, as Qwen2.5-1B-Instruct isn't publicly available. A benchmark counts as clear of chance if its CI lower bound is at least 0.05 above chance, checked on hellaswag acc, truthfulqa_mc2, winogrande and mmlu. 0.5B clears 2 of 4 (hellaswag, mmlu). Choose 1.5B if it clears more than 2 and the Stage 2 timings fit the $50/month budget; otherwise stay on 0.5B.
- Why: Choosing a model on base retention alone doesn't use any forgetting or KL results. Set after seeing the 0.5B scores, before running 1.5B.
- Affects: the Stage 2 pilot model choice. The near-chance rule is still set separately. If the chat-template-off diagnostic moves a benchmark clear of chance, log it and raise it before changing the suite.

Chat template diagnostic:
- What: Qwen2.5-0.5B-Instruct with the chat template off, on the G4 with all other settings the same as the base run. Scores with 95% bootstrap CI: hellaswag acc 0.4064 (0.3970-0.4161) and acc_norm 0.5245 (0.5146-0.5343); truthfulqa_mc2 0.4184 (0.3904-0.4467); winogrande 0.5422 (0.5154-0.5699). This doesn't replace the base run.
- Why: To check whether the chat format was holding near-chance scores down. No benchmark moves clear of chance under the pilot model rule: Winogrande's lower bound is below 0.55, and TruthfulQA's upper bound is just below its 0.4484 chance level. HellaSwag acc_norm does rise, so the format does affect that metric.
- Affects: the chat template stays on and the suite doesn't change. 0.5B still clears 2 of 4.

1.5B base retention results:
- What: Qwen2.5-1.5B-Instruct at commit 989aa79, same setup as the 0.5B base run (G4, fp32, chat template on, batch size 8). Scores with 95% bootstrap CI and minutes: hellaswag acc 0.4928 (0.4831-0.5027) and acc_norm 0.6250 (0.6155-0.6344), 5.6 min; truthfulqa_mc2 0.4654 (0.4352-0.4953), 1.5 min; winogrande 0.6772 (0.6519-0.7024), 0.7 min; mmlu 0.5966 (0.5887-0.6046), 14.5 min; ifeval strict 0.4122 (0.3715-0.4529) and loose 0.4418 (0.3993-0.4843), 14.3 min; humaneval_instruct 0.5549 (0.4756-0.6280), 4.1 min.
- Why: Input to the pilot model rule. 1.5B clears 3 of 4 benchmarks (hellaswag, mmlu, winogrande) against 0.5B's 2, so the retention condition is met. TruthfulQA's CI still includes chance. The pilot model isn't chosen until the Stage 2 timings are checked against the budget.
- Affects: the Stage 2 pilot model choice and the near-chance rule. The suite takes 41.7 minutes, about 7 G4 units per checkpoint.

Diagnostic and 1.5B provenance:
- What: Both runs record git_commit b3e8823, because the Colab clone hadn't pulled the later commits, and the cells that ran them weren't committed first. This breaks the rule above that later runs start from a committed notebook.
- Why: Neither run was repeated. Both use the same run_retention function, and each results file records its model, revision, precision, chat template setting, generation settings, benchmarks, batch sizes and package versions.
- Affects: these two runs only. From Stage 2 on, push first and pull on Colab before starting a run.
