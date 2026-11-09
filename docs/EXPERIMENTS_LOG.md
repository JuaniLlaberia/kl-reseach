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

## 06/11/26 - Start of Stage 2

Authorship:
- What: From Stage 2 on, Claude drafts the code and Juan reviews it in batches, with every choice that touches the math explained at review. This replaces the rule that Juan writes the core implementation. src/kl/ stays as Juan wrote it.
- Why: Juan's decision while planning Stage 2, where it was presented as the fastest option.
- Affects: src/data/, src/train/, src/eval/, tests and the Stage 2 notebook. CONTEXT.md and PLAN.md are updated.

Pre-registration order:
- What: A dated draft of the near-chance rule, the predictions and the Gate 1 science thresholds is committed before the mini-pilot runs. PLAN.md had the runs first.
- Why: These choices shouldn't depend on Stage 2 results, and the mini-pilot would show KL and retention numbers before they were written down.
- Affects: the Stage 2 order. The final pre-registration is still tagged by Nov 17.

Chemistry rows and prompt:
- What: SciKnowEval at commit 92ef969, Chemistry L3: all 2,400 rows, 2,100 mcq-4-choices and 300 filling. Prompts allow reasoning and require a final `Answer:` line, replacing the dataset's "without any explanation" instruction. The same text goes to GPT-4o, SFT, GRPO, the eval and KL. The wording is frozen once the 50-prompt test passes. Each family keeps its default system prompt on the local models.
- Why: RL's Razor says only "Chemistry L-3 subset", so no rows are filtered out. Direct answers would give responses of about 2 tokens, leaving per-token KL with few positions and GRPO little to vary. RL's Razor's eval matches the final answer and ignores reasoning, which suggests this format, but its prompt isn't stated. Chosen from row counts and the dataset's text, before any model ran on this task.
- Affects: all chemistry training, eval and KL. Accuracy is reported overall and per type. Departs from the dataset's instruction.

Chemistry split:
- What: 1,800 train and 600 held-out, seeded and stratified by subtask, with repeated questions kept on one side. The 500 KL prompts are drawn from held-out, stratified the same way. Saved to data/splits/chem_l3.json.
- Why: RL's Razor doesn't state a split, and the email to its authors is unanswered so far. Five questions repeat with different choices, which would otherwise leak across the split. Held-out must hold the 500 KL prompts; accuracy on 600 has a CI of about ±4 points.
- Affects: all chemistry runs. If the authors reply with their split, raise it before the Nov 17 freeze.

GPT-4o solutions:
- What: openai/gpt-4o-2024-08-06 through OpenRouter, with OpenAI as the only provider and fallbacks off, temperature 1.0 and no system prompt. Train rows only, up to 16 tries, one correct kept, no answer key in the prompt. Tries run in breadth-first rounds, with a hard stop at $15. Every attempt is logged. If rows stay unsolved, SFT and GRPO both train on the same solved rows.
- Why: A dated snapshot can't change during the study. To Claude's knowledge it's what the openai/gpt-4o alias pointed to when RL's Razor was written; the paper doesn't name one. Juan has $15 planned for this. Breadth-first rounds spread a limited budget across prompts rather than spending it on a few hard ones. A shared set of rows means the two methods see the same prompts.
- Affects: the SFT data. RL's Razor solved its whole dataset, so any shortfall is a deviation and is shown to Juan before the full run. A four-choice row can pass by chance within 16 tries, so the attempt number is saved per row and the first-try rate reported, with no filtering.

Mini-pilot checkpoints:
- What: The Reasoning SLM models, with Qwen/Qwen2.5-0.5B as the base and 500 GSM8K test prompts for KL. Drive no longer has the merged models, only the adapters, so they're rebuilt once: sft-r16-step400 merged into the base, then grpo_full-step400 merged into that result, in bf16 as the project's merge_adapter did. grpo_smoke isn't used. peft is installed only for this merge, in its own runtime, and stays out of requirements.txt. The mini-pilot uses the chat template, as the pilot will. TinyForge was considered and not used.
- Why: TinyForge's Hub repos have no GRPO checkpoint, and its custom format would need an adapter nothing else uses. Once merged, the Reasoning SLM models are ordinary Transformers models and should load with the existing KL and retention code. A separate runtime keeps peft out of the environment used for every run. Each merge is checked by confirming that the weights changed, not by reproducing the old scores, as this is only a pipeline check. They were trained with LoRA and then merged, GRPO was trained on top of SFT, and they trained on `Question: … Answer:` prompts, which is fine for a pipeline check. Their KL values aren't comparable to the earlier "KL about 0.13".
- Affects: the mini-pilot only. These are pipeline checks, not evidence, so the notebook has no forgetting-vs-KL plot or fit.

Trainer settings:
- What: GRPO sets loss_type="dr_grpo", scale_rewards=False and beta=0.0 explicitly, with no format reward. Both trainers load the model in fp32 before building the trainer, train under bf16 autocast and save in fp32.
- Why: TRL 1.17.1 defaults to loss_type="dapo" and scale_rewards="group", which isn't Dr. GRPO. TRL's docs say a model passed by path takes its dtype from the model config, and Qwen's declares bf16; a weight of 0.02 has a bf16 spacing of about 1.2e-4, so Adam steps near lr 1e-5 would round away. This risk comes from reading the docs and hasn't been run, so a test covers it. Saving in bf16 would round the weights being measured.
- Affects: all training. fp32 checkpoints are about 2 GB for 0.5B and 6 GB for 1.5B, which feeds the storage decision.

Stage 2 compute limit:
- What: Stage 2 spends at most 50 Colab units. The unmeasured estimate is about 34, of which only the retention costs come from measured runs.
- Why: Juan's limit. GRPO's seconds per step is the main unknown.
- Affects: the Stage 2 notebook. A section running well over its estimate stops, and it's raised before spending more.

Near-chance rule:
- What: A benchmark is clear of chance if the 95% CI lower bound of its base score is at least 0.05 above its chance level: hellaswag acc_norm 0.25, truthfulqa_mc2 0.4484, winogrande 0.50, mmlu 0.25, ifeval prompt_level_strict_acc 0 and humaneval_instruct pass@1 0. Flagged benchmarks are left out of the primary forgetting score and reported separately. The flag is set per model from its base score, before any training on that model, and doesn't change afterwards. On current base scores it flags truthfulqa_mc2 and winogrande for 0.5B, and truthfulqa_mc2 for 1.5B.
- Why: Chance differs by benchmark, so no single score threshold works. The 0.05 margin is reused from the pilot model rule, because choosing a new one with the Qwen scores in view could decide which benchmarks count. HumanEval's chance level is close to 0. IFEval's is above 0 and unknown, so 0 is a lenient floor, which still catches a model with no room to fall. acc_norm is a new choice, not stated earlier in the repo; the pilot model rule stays on hellaswag acc, and HellaSwag clears on both metrics for both models. Set from Stage 1 base scores only, before any Stage 2 run.
- Affects: the forgetting score and all retention reporting. Benchmark sets can differ by family, which has to be handled when the forgetting score is defined.

Pre-registration draft:
- What: docs/PREREG.md now holds the predictions, the Gate 1 science thresholds, the primary KL (kl_exact_fwd_base_mean_token) and the forgetting score (mean drop over a model's unflagged benchmarks, negative drops kept). Claude proposed these values and Juan accepted them; the predictions and the go, reshape or stop rule are Juan's to answer for. Written before any Stage 2 run. The draft tag covers these sections, widening the earlier plan, which covered only the near-chance rule, the predictions and the Gate 1 thresholds.
- Why: The mini-pilot shows KL variants and retention scores, which could sway the primary metric if it were still open. A missing gap doesn't stop the study on its own, as whether KL predicts forgetting is still testable, and a pre-registered null is a real result. Only one reshape is allowed, so the setup can't be changed until a preferred result appears.
- Affects: Gate 1, Stage 6 and all analyses. The R² prediction depends on the curve form, which is still open in Part 2 and has to be set by the final tag without looking at any fit.

50-prompt test:
- What: One GPT-4o first try on 50 train rows: 26 correct (0.52), with MCQ at 26 of 44 and filling at 0 of 6. It cost $0.2757, or $0.0055 per call, with a mean of 507 completion tokens and a maximum of 824. No response was truncated or lacked a valid answer, and all 50 came from openai/gpt-4o-2024-08-06 on OpenAI. The projected full run is $34.54-36.53, against the $15 limit, so it isn't approved and the prompt wording isn't frozen.
- Why: Required before the full run. The earlier estimate of $0.003 per call assumed about 250 completion tokens, about half of what was measured. Filling makes up as much as $19.85 of the projection, because 0 of 6 counts as 16 tries on all 225 train rows, and 6 rows can't show whether its true rate is really near 0.
- Affects: the solutions run, which waits on the filling responses and a decision on the prompt length. These attempts count towards the $15.

50-prompt test provenance:
- What: The test ran from commit 80fe16d with data/splits/chem_l3.json uploaded to Colab by hand, as the file wasn't committed. This breaks the rule that runs start from pushed code.
- Why: The test wasn't repeated. The notebook only continues if the split equals what split_rows generates from the pinned dataset, and it passed, so the file's content isn't in doubt.
- Affects: this test only. The split has to be committed and pushed before the full run.

Filling matcher:
- What: normalize_equation now also removes LaTeX (delimiters, \text, \mathrm, \ce, subscripts, arrows, ^ charges) and Unicode subscripts, and treats a hydrate dot as `*`. Saved attempts are rescored with the current is_correct whenever they're read, and the log itself is never rewritten. Rescored, the 50-prompt test is 30 of 50 correct (0.60), with filling at 4 of 6. The projected full run falls to $16.56-18.54, of which filling is about $1.86.
- Why: 4 of the 6 filling answers were correct but rejected for their typesetting, which was a bug in the matcher, found by reading the responses. The other 2 were really wrong and stay wrong; both cases are now tests. About 15 of the 300 filling answers contain charges, written inconsistently in the dataset (Mg{+2}, Al{3+}, Al3{+}), so some may stay unsolved. No single rule matches them all, so they're left as they are. Coefficients still have to match the dataset's answer.
- Affects: solution acceptance, the GRPO reward and the eval, which all use this function. Changed before any training or KL.

Solution quality:
- What: Of the 3 correct MCQ responses seen in the test, one reasoned its way to 4 rotatable bonds and then answered D, which was 2. Another rambled to its choice with no real derivation. Only the final answer is checked, so both count as solved.
- Why: This is a direct result of rejection sampling, which RL's Razor also uses. These were three responses from one task, so they don't show how common it is.
- Affects: the SFT targets, which can teach a final answer that doesn't follow from the reasoning. The attempt number is saved per row. Juan decided not to filter, to stay with RL's Razor's procedure, as any quality filter would be a new choice the paper doesn't have. It's reported as a limitation.

Prompt wording change:
- What: "Think step by step, then finish…" becomes "Reason briefly in plain text, then finish…", for GPT-4o, SFT, GRPO, the eval and KL. The same 50 rows are retested into a new log, chem_l3_gpt4o_attempts_v2.jsonl. The first test's $0.2757 counts towards the $15. The wording isn't frozen until this test passes, and credits are added only after it.
- Why: Completion tokens make up nearly all the cost, and at the current length the full run is projected at $16.56-18.54. Shorter responses also make GRPO's generation cheaper, and plain text stops Qwen learning LaTeX from the targets. Chosen for cost and format, from a test with no KL or forgetting in it.
- Affects: all chemistry prompts. The pass rate may fall on calculation tasks, which the retest will show. The first test's attempts are never used as solutions, as they came from another wording.

50-prompt retest:
- What: The same 50 rows with the shorter wording, from commit 020983a: 27 correct (0.54) against 30 (0.60) for the first test, rescored. Filling stayed at 4 of 6, molar weight went from 6 to 4 of 12, and molecular property from 4 to 3 of 11. Completion tokens fell from a mean of 507 to 344, with a maximum of 895, and cost per call from $0.0055 to $0.0039. No response was truncated or lacked a valid answer. The two tests cost $0.4698 in total. The projected full run is $13.01-16.10.
- Why: To check the new wording before freezing it. The difference of 3 rows on 50 can't separate a small real drop from noise. Tokens fell by about a third, less than the half that was assumed.
- Affects: the wording isn't frozen until the responses have been read.

Prompt wording frozen:
- What: The shorter wording is frozen, after reading 14 printed responses. No printed response was wrongly rejected. Of the 6 filling responses, 4 were correct with sound reasoning and 2 were really wrong. Answer lines were in plain text in 4 of the 6, and LaTeX still appears in some reasoning, which the matcher handles. Of the 6 correct MCQ responses printed, about 3 hold together, 2 don't match their own reasoning and 1 is a guess: one lists 4 acceptors and says 3, one concludes 4 rotatable bonds and answers D (2), and one estimates a peptide's weight without calculating it. All were from molecule structure and molar weight, as the printing limit was reached before the other tasks.
- Why: Cost and format were the reasons for the wording, and neither shows a problem. The mismatches look like GPT-4o picking the closest option when its own result isn't among the choices, which is inferred from a few responses. Reasoning quality is too small a sample to compare between the wordings.
- Affects: all chemistry prompts from here on. Any later change to the wording is a logged deviation. SFT targets stay unfiltered, as already decided, so some teach an answer that doesn't follow from the reasoning; this is reported as a limitation.

OpenRouter rate limit:
- What: The first full run stopped with status 429 after about 2 minutes of retries, starting from commit 020983a. The client now waits for about 5 to 10 minutes before giving up, with capped and randomized waits, runs 4 calls at once instead of 8, and reports OpenRouter's explanation in the error. Attempts already saved are kept and not repeated. The run stays on OpenAI only.
- Why: OpenRouter listed the OpenAI route at 100% uptime, so this was throttling and not an outage. The cause isn't known, as the first error didn't include OpenRouter's explanation. A refused call isn't an attempt, so it isn't logged and doesn't affect any scored result.
- Affects: how fast the run goes, not the solutions. Fewer calls at once should make it take longer.

GPT-4o solutions run:
- What: Stopped at the budget during round 4, at commit ef28272, with max_tokens 1024. Solved 1,381 of 1,800 train rows (76.7%): 989 on try 1, 220 on try 2, 104 on try 3, 68 on try 4. No row got more than 4 tries, so the limit of 16 was never reached; 402 unsolved rows got 4 tries and 17 got 3. Solved per task: reaction prediction 298/300, retrosynthesis 214/225, molecular property 278/375, molecule structure 153/225, molar weight 300/450, balancing equations 138/225. All 3,746 calls used the requested model and provider, none touched held-out rows, 1 was truncated and 53 had no valid answer.
- Why: The projection of 1,650-1,790 rows at $15 was wrong, because it assumed retries pass about as often as first tries. Pass rates among the rows tried in each round were 54.9%, 27.1%, 18.3% and 14.5%, so the limit wasn't raised: the extra rows would have cost an estimated $7 or more, and later passes are often lucky. Juan chose to stop at $15.
- Affects: SFT and GRPO both train on the same 1,381 rows, so one pass has fewer steps than with 1,800. Training leans towards easier rows: reaction prediction and retrosynthesis go from 29% of the train rows to 37% of the solved rows. Held-out rows stay unfiltered. 392 solutions (28%) passed after try 1, and the attempt number is saved per row. Comparisons with RL's Razor's R² are weaker.

Solutions spending and repeated calls:
- What: The total in the logs is $15.22 ($14.94 in the attempts log plus $0.28 from the first test), $0.22 over the $15 limit. The OpenRouter dashboard shows $15.3, as read by Juan. 64 row and attempt pairs in round 2 were called twice, at a cost of $0.29, and 10 more calls went to rows already solved by one of those repeats. Their second copies sit together in the log, at lines 2,553 to 2,675.
- Why: The budget check only counts calls made by the running cell. The repeats match the gap between the run's own total and the log, so a second writer is the explanation. It was most likely the earlier run still working when the run was resumed, which is inferred, as the log has no timestamps. The remaining gap of about $0.08 to the dashboard is unexplained; a rounded dashboard figure or calls that were paid for but never logged could account for it.
- Affects: spending, and very little else. Each row keeps one solution, the first correct one in log order, and the file was checked: 1,381 unique train rows, all correct under is_correct. For the 64 repeated rows, attempt 2 may have been the third draw, so a few had one extra chance to pass. The repeats share a seed but returned different text, so the seed doesn't reproduce a response; every response used is saved. No more GPT-4o calls are planned, so the budget check stays as it is.

RL's Razor chemistry annotations:
- What: Appendix B.1 says that for Science Q&A "we applied the same procedure with GPT-4o, obtaining correct annotations for the entire dataset". The procedure is stated for DeepSeek R1 on math: up to 16 responses per prompt, keeping one that matched the correct final answer, which gave valid annotations for 96% of that dataset. The paper doesn't state the GPT-4o snapshot, the temperature, the number of tries for GPT-4o, the number of chemistry rows, any filtering by question type, or the split. It does state that SFT and RL used "the same set of prompts".
- Why: To know what the shortfall can be compared against. "Correct" there means the final answer matched, and nothing says the reasoning was checked or that the chemistry annotations contained reasoning at all. The paper reports no first-try rate.
- Affects: The shortfall is a difference from the paper, reported as such, but its size can't be known without their data. Using the same prompts for both trainers matches the paper. The email to Shenfeld and Pari asks for their annotations or code, with no reply as of this entry.

Solutions kept out of the public repo:
- What: The solutions and attempt logs stay on Drive and in a local ignored folder, and only the run's summary is committed. The trainers read the solutions from Drive.
- Why: PLAN.md publishes them only if the license allows. SciKnowEval is MIT, but OpenAI's terms on sharing outputs and on training other models with them haven't been checked, and pushing to a public repo publishes them.
- Affects: the data step's reproducibility until the check is done, before the public release in Stage 6.

Correction to the trainer settings:
- What: The "Trainer settings" entry says a model passed to GRPO by path takes its dtype from the model config. TRL 1.17.1's source says the opposite: a path defaults to float32, unlike from_pretrained. Both trainers still load the model themselves in fp32, fail on anything else, and verify after saving that the checkpoint reloads with identical fp32 weights.
- Why: That risk was inferred from reading the docs and was wrong for GRPO. Loading the model ourselves costs nothing and works the same way for both trainers.
- Affects: nothing in the code that was planned, only the stated reason.

Trainer batch:
- What: SFT uses the plain Transformers Trainer with a small collator, not TRL's SFTTrainer. GRPO uses a small subclass of TRL's trainer that generates in chunks of a set size. Each run has a time limit. Labels end at the first stop token and are never cut, as a longer row raises an error. Both trainers use the 1,381 solved rows, and GRPO's stop tokens must equal the ones KL uses.
- Why: The data is already tokenized with the same prompt IDs as KL, so SFTTrainer's templating and truncation would only add ways to differ. TRL makes one generate call for the whole batch, and 64 by 8 at 512 tokens on 1.5B would need about 26 GB for the generation cache alone, an estimate from the model's size. The subclass overrides a private TRL method, which is safe only because TRL is pinned. The time limit protects the 50-unit budget.
- Affects: all training. Dr. GRPO's loss is divided by max_completion_length, so that choice also rescales the gradient; Adam mostly cancels a constant scale, but gradient clipping doesn't.

Memory and smoke configs:
- What: The memory test runs SFT for 10 steps and GRPO for 5 steps, not 10, at 64 by 8. The GRPO smoke run uses a group of 16 by 8 for 60 steps, and SFT uses batch 16. All use lr 3e-5, with 50 warmup steps in the smoke runs, 8-bit AdamW, bf16 autocast, and up to 512 new tokens. The estimates are now 14 units for section B, 5 for C, 11 for D and 2 for E, about 32 in total. These estimates haven't been measured.
- Why: A GRPO step at 64 by 8 generates 512 completions, so 5 steps should show memory and timing at a lower cost. Timing comes from two group sizes, 64 in section C and 16 in section D. None of these are pre-registered choices.
- Affects: Stage 2's budget and the Part 3 inputs. The SFT targets run up to 825 tokens with a 90th percentile of 508, so a cap of 512 may be too short. Section E measures this on base responses, and it's decided from those.

Mini-pilot pins:
- What: Qwen/Qwen2.5-0.5B at 060db64 and GSM8K at 740312a. peft 0.21.2 is installed with no dependencies, only in the one-off merge runtime.
- Why: The earlier plan left these open. The merge was tested with peft 0.21.2 and the pinned Transformers on a tiny model with adapters made for the test, not on the real adapters.
- Affects: the mini-pilot only. If the real adapters fail to load, the peft version may need to change.

Dry run of the trainer batch:
- What: On CPU, with the real solutions, Qwen's real tokenizer, chat template and stop tokens, and a tiny random model: 2 SFT steps, 1 GRPO step, saving, reloading and the new-task eval all ran. Tests: 259 CPU tests and 2 network tests pass. A dry run with the real 0.5B weights was stopped after 11 minutes on Juan's laptop, as it was using nearly all the memory and hadn't finished its first run.
- Why: To catch integration errors before spending Colab units. Real weights weren't needed for that.
- Affects: Not yet run: the GPU tests, the 8-bit optimizer, bf16 autocast on a GPU, W&B logging, any real weights, and the moved retention code on a real model, which is only tested as identical to the Stage 1 source. Section C runs the GPU tests first.

## 07/11/26

Tests as a package:
- What: tests/__init__.py added. On Colab, collection failed on test_gpu, test_grpo and test_new_task with "No module named 'tests.conftest'", before any model ran.
- Why: Without the file, `tests` was a namespace package, which loses to any regular `tests` package elsewhere on the path, apparently one installed on Colab. Reproduced locally with a fake one; with the file, all 264 tests are collected and the 259 CPU tests pass.
- Affects: test collection only. No run used a GPU before the fix.

RQ1 curve, filtering, RQ3 matching and sample sizes:
- What: RQ1 uses a quadratic with a free intercept, fit on all checkpoints. RL's Razor's Pareto filter, within 2 accuracy points of each method's frontier, is a secondary fit for comparison only. RQ3 pairs SFT and GRPO checkpoints within 2 accuracy points, one-to-one, closest pair first, with the gap averaged over pairs and a bootstrap CI over pairs. The primary KL is also computed on nested, subtask-stratified subsets of 50, 100 and 250 of the 500 KL prompts. Written into PREREG.md Part 2.
- Why: The quadratic is RL's Razor's form, chosen by Juan so R² is comparable; the paper states no intercept constraint. The Pareto filter selects points partly by forgetting, the outcome being fit, so it isn't primary; the paper doesn't say whether its R² of 0.71 used it, and "within 2 points of the frontier" is our reading of its wording. The matching rule reuses Gate 1's threshold, with one-to-one pairing so no checkpoint enters several pairs. RL's Razor doesn't state how many prompts its KL used, so the subsets show how many the law needs; they cost no GPU, as they're recomputed from the per-sequence files. They're limited to the primary KL so the number of comparisons stays small. All chosen before any chemistry KL or forgetting exists, after the prereg-draft tag.
- Affects: the Stage 6 analyses and the final pre-registration. A quadratic with 3 parameters on about 8 to 12 checkpoints per family gives an optimistic in-sample R², which goes in the limitations. The subset IDs have to be drawn and committed before any chemistry KL.

Mini-pilot (section B):
- What: Retention and KL on Qwen2.5-0.5B (base, 060db64), sft-r16-step400 merged into it, and grpo_full-step400 merged into that, from commit 742aaae. The results JSONs record only Drive paths for the merged models. All 14 KL variants were saved for SFT and GRPO, KL(base ‖ base) was 0 on every variant, and the non-negativity and prompt-coverage checks passed. Retention took 38.6, 35.8 and 35.6 minutes, of which IFEval took about 21, against 23.7 and 11.8 for the 0.5B Instruct run. MMLU was at chance for all three models (0.254 to 0.265), with right and wrong items matching on 92 to 98% between them. 197 of 500 base responses hit the 512-token cap. The KL minutes weren't recorded. The results were copied into results/minipilot_gsm8k/ from a folder Finder had named "minipilot_gsm8k 2".
- Why: Pipeline check, not evidence. MMLU at chance is consistent with a base model given the chat template and 5-shot examples (Qwen reports about 0.47 for this model without a template, and the Instruct model scored 0.46), so it isn't treated as a code fault: HellaSwag and Winogrande score normally with the same code. The base model's IFEval responses repeat themselves, which explains the longer runtime. SFT and GRPO scores weren't compared with the base.
- Affects: The moved retention code and the KL module are confirmed on real fine-tuned checkpoints. The mini-pilot can't show that MMLU detects a change, as it sits at its floor, which doesn't apply to the Instruct models in the study. Retention cost depends on how much a checkpoint writes, so the budget uses fine-tuned checkpoints' costs, not base runs'.

GPU test fix:
- What: test_autocast_runs_in_bf16_while_weights_stay_in_fp32 hooked a decoder layer's output and expected bf16. It now hooks q_proj. The training code didn't change.
- Why: Autocast only changes matmuls. A decoder layer's output stays fp32, because the residual adds the bf16 output of its matmuls to fp32. Checked on CPU with a tiny Qwen2: under autocast q_proj, down_proj and lm_head give bf16 and the layer output fp32; without it, all fp32. The other two GPU tests passed: both trainers ran with 8-bit AdamW, bf16 autocast and gradient checkpointing, and saved in fp32.
- Affects: the GPU tests only.

Memory and smoke runs (sections C and D):
- What: On the G4, seconds per step and peak reserved memory: SFT 1.5B 0.97 s and 88.7 GB (91.7 GB in the smoke run), SFT 0.5B 0.40 s and 81.6 GB, GRPO 1.5B at 64 by 8 90.5 s and 29.7 GB, GRPO 0.5B 78.0 s and 17.4 GB. One pass over the solved rows is 86 SFT steps (about 0.2 units) and 172 GRPO steps (about 43 units for 1.5B). A 1.5B checkpoint is 6.2 GB in fp32. The SFT smoke loss fell from 1.35 to 1.20 between its first and last thirds, with a mean gradient norm of 2.9, so clipping at 1.0 was active on most steps. Stage 2 has used 35 units of the 70 at its start.
- Why: Inputs to Part 3. GRPO's time is mostly generation: 0.5B is barely faster than 1.5B, and each step makes 4 generate calls of 128 completions, each up to the cap. SFT's reserved memory is mostly the allocator's cache, as only 52 GB was allocated.
- Affects: the budget and Part 3. Evaluating a checkpoint (about 9 to 13 units) costs more than an SFT run, so evaluation sets the floor on the grid. SFT memory is close to the 96 GB limit; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is the first fix if it runs out, as it changes no computation.

GRPO collapse:
- What: Both 1.5B GRPO runs at lr 3e-5 collapsed. In the smoke run, mean completion length went from 179 tokens to 5 and entropy from 0.96 to 0.05, starting by step 15 at lr 8e-6 during warmup. Reward rose from 0.15 to about 0.40 at around 30 tokens, then fell to about 0.25, chance on four choices, with 78 to 95% of groups giving every completion the same reward. The memory test collapsed by its third step. Its 90.5 s per step was measured partly on collapsed steps, and the smoke run's 8.5 s entirely on them, so neither is the cost of an uncollapsed run.
- Why: Found from the trainer's log of reward, length and entropy, with no KL or forgetting. The learning rates are RL's Razor's (1e-5 to 5e-5, constant with 50 warmup steps, 64 by 8, Dr. GRPO, no KL, one epoch), but they trained Qwen2.5-3B, and their table lists bfloat16 with their Section 6 mentioning sparse updates due to bf16. If their weights were stored in bf16, most small Adam steps rounded away, while fp32 master weights apply every step. That's inferred, not stated by the paper.
- Affects: the GRPO learning rates, which wait on the section D2 rerun. RL's Razor's grid can't be used as is on this setup.

Lengths and cap:
- What: On 300 chemistry train prompts with a 1024 cap, base responses have a median of 161 tokens, a 90th percentile of 437 and a 99th of 941; 7.7% run past 512 and 0.7% hit 1024. SFT targets have a median of 352 and a maximum of 825, and full SFT sequences reach 1,200 tokens. max_new_tokens for KL sampling and the new-task eval, and GRPO's max_completion_length, are set to 1024.
- Why: At 512, KL would be measured on cut responses, SFT would learn responses the eval cuts, and GRPO's completions past 512 would score 0, pushing it shorter for a reason that has nothing to do with the method. Dr. GRPO divides the loss by max_completion_length, so the gradient halves; Adam mostly cancels a constant scale, and clipping wasn't active in GRPO (mean gradient norm 0.07). Chosen from lengths, with no KL or forgetting.
- Affects: generation cost, which can double for a batch with a long response, and Part 3. SFT's max_length of 2048 stays.

Drift on smoke checkpoints:
- What: bf16 against fp32 on 100 train prompts: the primary KL moves by 5.73% on the GRPO smoke checkpoint and 1.54% on the SFT one. Other exact variants move by 0.2 to 6.8%, and k1, k2 and k3 by 0.0 to 1.5%.
- Why: Required before the freeze, as Stage 1 only had synthetic noise. Only the relative differences were looked at.
- Affects: KL stays in fp32. In bf16, the error would differ by method, which would bias the comparison between them. The GRPO smoke checkpoint was a collapsed model.

Lenient matcher:
- What: Of 300 base responses on train prompts, 54% of MCQ ones failed the old matcher on format (84 with no "Answer:" line, 58 with one it rejected), against 37 correct and 81 valid but wrong; filling had 1 of 38 correct. The matcher now reads the last answer label anywhere in a line ("Answer:", "Final answer:", "The correct answer is:", "Final line:"), the next line if the label stands alone, and the last non-empty line if there's no label. An MCQ answer is a letter A to D, alone or followed by its option's text ("B. 22", "D (4)"). A word that starts with a letter, a bare value, or a letter outside A to D stays wrong. Tests now accept "Answer:\nB", "Answer: B) 18" and "Final answer: B", which were tested as invalid, and include 22 real Qwen endings. Rescoring the 3,746 GPT-4o attempts flips 4 MCQ attempts from wrong to correct and none the other way, so 1,384 rows are solved (990 on the first try), with one row (15261) keeping its first attempt instead of a later one. The saved solutions and summary were rewritten; the summary's commit field still names the run's commit, ef28272. Juan approved the change and chose 1,384.
- Why: New-task accuracy should measure chemistry, not format. SFT learns the format from GPT-4o's targets while GRPO has to find it, so the old matcher would have favored SFT in the matched-accuracy comparison, and GRPO's first reward came mostly from format. Rescoring follows the logged rule that saved attempts are always scored with the current is_correct. The newly accepted GPT-4o answers hedge ("Answer: D (though it's not accurate…)"), which fits the logged limitation that only the final answer is checked. Designed from Qwen's real responses, with no KL or forgetting.
- Affects: the GRPO reward, the new-task eval and the SFT data, which gains 3 rows (molar weight, molecular property, retrosynthesis). Drive needs the new solutions file before any training. Among answers the base model states clearly, MCQ accuracy was about 31% (37 of 118), against 25% chance.

GRPO rerun (section D2):
- What: Two runs on the G4. A smoke rerun at lr 3e-6 with 10 warmup steps, 60 steps, 16 by 8, and a 1024 cap. A timing test at 64 by 8 with all 512 completions in one generate call, at lr 3e-6 and the 1024 cap, for 5 steps. Neither saves a checkpoint. A CPU cell first scores the 300 saved base responses with the new matcher, as the bar the reward has to beat.
- Why: Collapse began between lr 5e-6 and 8e-6 in the first smoke run, so 3e-6 sits below it. 10 warmup steps keep most of the run at the full rate; the pilot configs keep RL's Razor's 50. One generate call should be faster, as the first memory test only used 30 of 96 GB; it's timed at lr 3e-6 so the responses being timed don't collapse. Juan chose these settings.
- Affects: the decision on the task. If length and entropy hold and reward rises above the base accuracy, chemistry stays for the pilot. If not, the main task is reconsidered (math, or a study of public checkpoints) before the final pre-registration. An estimated 9 units, so Stage 2 ends at about 44 of 50.
