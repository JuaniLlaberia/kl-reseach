# KL Research: Stage-by-Stage Checklist

## Stage 1 · Nov 4–10: reading and measurement tools

By the end of this week you can measure forgetting and KL on any checkpoint. No training happens yet.

**Read**

- [x] RLHF book ch. 15, and forward vs reverse KL
- [x] KL estimators (Xihuai Wang's post, sections 3–5)
- [x] RL's Razor, including Appendix B.1 (C.1 in the camera-ready)
- [x] Retaining by Doing, including Appendix A.5; send notes in the usual format

**Set up the repo**

- [x] Git repo with pinned versions of transformers, TRL and lm-evaluation-harness
- [x] One config file per run, and one results file per checkpoint
- [x] A lab log: a dated line for every decision and deviation

**Code the KL module** (you write it; I review)

- [x] Exact per-token KL over the full vocabulary, forward and reverse, computed in fp32
- [x] Response tokens only, never prompt tokens
- [x] Two aggregations: average per token (primary) and sum per sequence
- [x] k1, k2 and k3 estimators as secondary variants
- [x] Base-response sampler: temperature 1.0, no top-p, fixed seed; responses saved once and reused for every checkpoint

**Test it**

- [x] KL(base ‖ base) = 0
- [x] KL is never negative
- [x] The average k1 value is close to the exact KL
- [x] Record how far bf16 drifts from fp32

**Run**

- [ ] Retention suite on Qwen2.5-0.5B-Instruct: score, bootstrap CI and minutes per benchmark
- [ ] Note which benchmarks sit near chance
- [ ] GPU check: which GPUs Colab Pro and Kaggle actually give you, and whether they support bf16

**Admin**

- [ ] Email Shenfeld and Pari with the four questions (KL computation, chemistry split, which points the fit uses, annotations or code). Don't wait for a reply.

**Done when** (by Nov 10, 2026): base retention scores are logged and the KL tests pass.

## Stage 2 · Nov 11–17: data, trainers and pre-registration

This week ends with a frozen pilot spec and written predictions. If time runs short, cut papers, never the pre-registration.

**Read** (in this order; cut from the bottom)

- [ ] Mechanistic origins (RL forgot less despite larger KL)
- [ ] Context distillation, §6.2
- [ ] ES study, §A.4.1
- [ ] Lai et al.

**Chemistry data**

- [ ] Filter SciKnowEval to Chemistry L-3; record the row count and the `type` values
- [ ] Seeded train / held-out split; save the row IDs to the repo
- [ ] Choose 500 held-out prompts for KL
- [ ] GPT-4o test on 50 prompts: tokens per call and first-try pass rate; recompute the cost
- [ ] GPT-4o solutions for the train split only. Rejection sampling, up to 16 tries, one correct kept, no answer key in the prompt
- [ ] Log the snapshot, system prompt, temperature, max tokens and share of prompts solved
- [ ] One answer-extraction function, shared by the GRPO reward and the eval, with unit tests on awkward outputs

**Trainers**

- [ ] SFT: loss on response tokens only, max length set
- [ ] GRPO: Dr. GRPO loss, binary reward, no format reward, temperature 1.0, max completion length set

**Runs** (pipeline checks, not evidence)

- [ ] Zero-training mini-pilot: retention and KL on your Reasoning SLM checkpoints (base, SFT, GRPO). Log anything that tempts you to change the metric
- [ ] Memory smoke test: 1B full fine-tuning, about 10 steps, 8-bit AdamW, gradient checkpointing
- [ ] Smoke run: about 50 steps per trainer on chemistry. Loss falls, reward varies within groups, checkpoints save and load
- [ ] Seconds per step for SFT and GRPO, extrapolated to a full run

**Decide and write down** (the pre-registration)

- [ ] Pilot model: 0.5B or 1B
- [ ] Which checkpoints count as points: final only, or every N steps
- [ ] Near-chance rule as a number
- [ ] GRPO group size and prompts per step
- [ ] Checkpoint storage: Hugging Face repo, Drive, or compute every KL variant and then delete
- [ ] Gate 1 criteria as numbers
- [ ] Predictions: whether the gap appears, the R² range, which KL variants fit worse
- [ ] Freeze the primary metric (KL definition and forgetting aggregation)
- [ ] Timestamp all of it: a tagged git commit pushed to GitHub, or an OSF registration
- [ ] Claude: novelty re-check

**Done when** (by Nov 17, 2026): the pilot spec and predictions are committed with a timestamp.

## Stage 3 · Nov 18–25: pilot and Gate 1

The pilot is one family on one task, run exactly as pre-registered. It answers two questions: does the RL-vs-SFT forgetting gap show at this scale, and do the runs fit the budget?

**Train**

- [ ] 3–4 SFT configs, learning rates from 1e-5 to 9e-5
- [ ] 3–4 GRPO configs, learning rates from 1e-5 to 5e-5 (Colab Pro if Kaggle's quota runs short)
- [ ] 2–3 extra seeds on one config, to measure noise

**Measure every checkpoint**

- [ ] New-task accuracy on the held-out split
- [ ] Retention suite, with the same harness version and settings as the base run
- [ ] Primary KL, plus every secondary variant (direction, aggregation, sample source, estimator, sample size)
- [ ] Results appended to one table: one row per checkpoint

**Look**

- [ ] Plot forgetting against new-task accuracy, SFT vs GRPO
- [ ] Plot forgetting against primary KL
- [ ] Log GPU hours per run and the total used

**Gate 1** (review the numbers together)

- [ ] Check each pre-registered Gate 1 criterion: pass or fail, stated plainly
- [ ] Decide: go, reshape (for example a larger model on rented GPUs), or stop
- [ ] If go: size the core grid from the timings (families, tasks, configs, seeds)
- [ ] Decide on math as an optional third task, from the same timings

**Done when** (by Nov 25, 2026): Gate 1 is decided and written in the lab log.

## Stage 4 · Nov 26 – Dec 3: finals buffer

Finals come first; exams are on Dec 2 and 3. Only low-effort research work happens here.

- [ ] If Gate 1 passed: queue only runs that need no debugging (more Qwen chemistry configs, extra seeds)
- [ ] If there's a free evening: check ToolAlpaca's format (ground-truth calls, how to match them with regular expressions)
- [ ] Nothing that needs new code

## Stage 5 · Dec 4 – Jan 31: core runs and Gate 2

The grid size comes from Gate 1's timings, not from this list. The full plan is 3 families × 2 tasks × about 12 configs per method, plus seeds on a subset.

**Set up** (first week of December)

- [ ] Pick the other two families' instruct models (Llama-3.2-1B-Instruct, plus OLMo or SmolLM); check each one's chat template in the harness
- [ ] ToolAlpaca: seeded split, SFT on ground-truth calls, regex-based reward and eval shared by both trainers
- [ ] Per family and task: base retention scores, and base responses for the KL prompts (generated once)

**Run** (in this order, so a cut leaves complete blocks)

- [ ] Qwen × chemistry (extend the pilot)
- [ ] Qwen × tool use
- [ ] Llama × chemistry, then tool use
- [ ] Third family × chemistry, then tool use
- [ ] Extra seeds on a subset
- [ ] For every checkpoint: accuracy, retention and all KL variants, added to the results table

**Monitor**

- [ ] Weekly: GPU hours and money spent vs the $50/month budget; rerun failed runs
- [ ] Informal spread check around Jan 15, 2027: the ranges of KL and forgetting, not the fit
- [ ] Gate 2: is there enough spread in KL and forgetting to fit curves? Pass or fail, stated plainly

**Read while runs train**

- [ ] Context literature: Wolfe's overview, LoRA Forgets Less, then the rest of the list
- [ ] Start an arXiv endorsement request (needed before March)

**Done when** (by Jan 31, 2027): the results table is complete and Gate 2 is decided.

## Stage 6 · February: analysis, then writing

Run the pre-registered analyses first, exactly as written, and only then explore. Anything not in the pre-registration is labelled exploratory in the paper.

**Pre-registered analyses**

- [ ] RQ1: primary KL vs forgetting, with R² and a bootstrap CI, per family and pooled. Do SFT and GRPO sit on one curve?
- [ ] RQ2: every KL variant on the same checkpoints, all reported, including the ones that fit badly
- [ ] RQ3: SFT vs GRPO forgetting at matched new-task accuracy
- [ ] Check every prediction against the result: right or wrong, stated plainly

**Exploratory** (labelled as such)

- [ ] Per-benchmark drops, and the version normalized by distance from chance
- [ ] Anything else that came up in the lab log

**Write** (figures first)

- [ ] Final figures and tables
- [ ] Setup section, including every deviation from RL's Razor and why it was needed
- [ ] Results, then introduction and related work
- [ ] Limitations: scale (0.5–1.7B), number of families and tasks, approximate replication
- [ ] Optional: a short non-archival workshop version
- [ ] Claude: a review by a separate agent that hasn't seen the drafting

## Stage 7 · March: release and submit

The code and results go public with the preprint, so anyone can check the fits.

- [ ] Public repo: code, configs, data split IDs, GPT-4o solutions (if the license allows), the full results table, the pre-registration
- [ ] Final novelty check for papers that appeared since November
- [ ] arXiv endorsement confirmed, then the preprint posted
- [ ] TMLR submission on OpenReview, anonymized, with an anonymized code link
- [ ] Share the preprint with RL's Razor's authors

## Throughout

- **Pre-registration:** after Nov 17, any change to the primary metric, the analyses or the gate criteria is a logged deviation with a reason, never a silent edit.
- **Don't steer by early numbers:** the mini-pilot and pilot are pipeline checks. If a result tempts you to change a choice, log it and raise it before changing anything.
- **Same setup for every checkpoint:** harness version, prompts, chat template, KL prompts and precision.
- **Scope:** add a family, task or method only when the pilot timings show it fits the budget.
- **Reading notes:** send them after each paper (claim, KL as measured, forgetting as measured, pilot implications, questions).
- **Code:** you write the core implementation; I help with design, review and debugging.
