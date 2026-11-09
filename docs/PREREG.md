# Pre-registration

**Status:** draft written, not tagged yet.

- **Draft tag:** `prereg-draft`, pushed before the mini-pilot runs. Covers Part 1, plus Primary KL and Forgetting in Part 2.
- **Final tag:** `prereg-final`, pushed by Nov 17, 2026. Covers Parts 1 to 3.
- After the final tag, any change is a logged deviation with a reason, never a silent edit.

`TBD` means not decided yet. `Logged` values come from `docs/EXPERIMENTS_LOG.md`.

## Part 1: written before the mini-pilot

These may not use any Stage 2 run.

### Near-chance rule

There's no single score threshold, because chance differs by benchmark. The rule is one test applied per benchmark and per model, using that model's base score.

| Benchmark          | Chance level | Source                                          |
| ------------------ | ------------ | ----------------------------------------------- |
| hellaswag          | 0.25         | Four choices                                    |
| truthfulqa_mc2     | 0.4484       | Logged: mean share of true answers per question |
| winogrande         | 0.50         | Two choices                                     |
| mmlu               | 0.25         | Four choices                                    |
| ifeval             | 0 (floor)    | True chance is above 0 and unknown, so this is a lenient floor, not a chance test |
| humaneval_instruct | 0            | Code rarely passes its tests by accident        |

- Test: a benchmark is clear of chance if the 95% CI lower bound of its base score is at least 0.05 above its chance level; otherwise it's flagged as near chance. This is the same margin as the logged pilot model rule.
- HellaSwag metric: acc_norm.
- IFEval is tested on `prompt_level_strict_acc`, and HumanEval on `pass@1`.
- A benchmark flagged as near chance is left out of the primary forgetting score and reported separately. The flag is set from the model's base score, before any training on that model, and doesn't change afterwards.
- Input: Stage 1 base scores only.

### Predictions

These are forecasts, checked against the results in Stage 6. "Points" means percentage points.

- **The gap appears:** at matched new-task accuracy, GRPO forgets at least 1 point less than SFT.
- **R²:** primary KL vs forgetting gives 0.3–0.6 per family, below RL's Razor's 0.71, because these models are smaller and the retention scores are noisier.
- **One curve:** pooling SFT and GRPO fits worse than fitting each method alone, as Mechanistic origins found RL forgetting less despite larger KL.
- **KL variants that fit worse than the primary:**
  - k3 fits worst: rare tokens can blow it up, as in the drift test (0.35 to 10.8 at 0.05 noise, in bf16).
  - k2 fits next worst: it was about 1.9x the exact KL in the drift test. R² ignores a constant scale, so this only hurts if the bias varies across checkpoints.
  - `sum_seq` fits worse than `mean_token`, as it also picks up differences in response length.

### Gate 1 science thresholds

Seed noise is the standard deviation of forgetting across the extra seeds of one config.

- **The gap shows** if all three hold:
  - At least 2 SFT–GRPO pairs of checkpoints are within 2 accuracy points of each other.
  - The mean gap across those pairs is at least 1 point, with GRPO forgetting less.
  - The mean gap is at least 2× the seed noise.
- **Enough spread** means both hold:
  - Forgetting's range across checkpoints is at least 3× the seed noise and at least 2 points.
  - The largest primary KL is at least 3× the smallest.
- **Decision:**
  - **Go:** the gap, the spread and the budget criteria in Part 3 all pass.
  - **Reshape, once only:** any of them fails.
  - **After the reshape:** stop if the spread or the budget still fails. If only the gap is missing, go, because whether KL predicts forgetting is still testable and a pre-registered null is a real result.

## Part 2: the primary metric and analyses

Chosen from their definitions, never from fit.

### Primary KL

- Logged: exact KL over the full vocabulary, response tokens only, fp32, average per token (mean of per-sequence means).
- Direction and sample source: forward, KL(base ‖ fine-tuned), on base samples. RL's Razor writes the law as KL(π₀ ‖ π), and this is one of the two combinations that give a true sequence-level KL.
- Full variant name: `kl_exact_fwd_base_mean_token`.

### Forgetting

- Per benchmark: base score minus checkpoint score. Negative drops are kept, as a model can improve.
- Aggregation: the mean of the drops over a model's unflagged benchmarks, with each benchmark weighted equally.
- Metrics: hellaswag `acc_norm`, truthfulqa_mc2 `acc`, winogrande `acc`, mmlu `acc`, ifeval `prompt_level_strict_acc`, humaneval_instruct `pass@1`.
- Near-chance benchmarks: left out and reported separately, following the rule in Part 1.
- Across families: a per-family fit uses that family's unflagged benchmarks. A pooled fit uses only the benchmarks that are clear on every family in it.

### Secondary variants

- Logged: `kl_exact_{fwd,rev}_{base,ft}` and `kl_{k1,k2,k3}_rev_ft`, each as `mean_token` and `sum_seq`.
- Sample-size variants: the primary KL recomputed on nested subsets of the 500 KL prompts, of 50, 100 and 250 prompts, with the full 500 as the primary. Each subset is stratified by subtask like the 500, drawn with a fixed seed, and saved to the repo before any chemistry KL is computed. They're computed from the per-sequence KL files, with no extra GPU. Reported: R² at each size, and the bootstrap spread of each checkpoint's KL at each size. Only the primary KL gets these; the other variants use all 500.
- All of them are reported, including those that fit badly.

### Analyses

- **RQ1:** primary KL vs forgetting, with R² and a bootstrap CI over checkpoints, per family and pooled. Do SFT and GRPO sit on one curve?
  - Curve form: quadratic with a free intercept, forgetting = a + b·KL + c·KL², fit by least squares. This is RL's Razor's form ("a quadratic fit achieving R²=0.71"; "2nd deg. polynomial" in its Table 1), which states no intercept constraint.
  - Points: primary fit on all checkpoints that count as points, with no filtering.
  - Secondary fit, for comparison with RL's Razor only: their filter, keeping models "within 2 accuracy points of the Pareto frontier" (their Appendix B.1). Read here as: per method, the frontier is the checkpoints that no other checkpoint beats on both new-task accuracy and forgetting, and a checkpoint is kept if its forgetting is at most 2 points above the frontier, interpolated linearly at its new-task accuracy. The paper doesn't say whether its R² uses the filtered set.
  - Why the filter isn't primary: the frontier is defined partly by forgetting, so filtering on it selects points by the outcome being predicted.
- **RQ2:** every KL variant on the same checkpoints.
- **RQ3:** SFT vs GRPO forgetting at matched new-task accuracy.
  - Matching rule, the same as Gate 1: SFT–GRPO pairs of checkpoints within 2 accuracy points. Pairs are one-to-one: take the closest pair, remove both checkpoints, and repeat until no remaining pair is within 2 points.
  - Gap: SFT forgetting minus GRPO forgetting, averaged over the pairs, with a bootstrap CI over pairs.
- Anything else is labelled exploratory.

## Part 3: written from the Stage 2 timings

These may use timings, memory and storage, never KL or forgetting.

### Pilot spec

- Pilot model: TBD. Logged rule: 1.5B if it clears more than 2 of 4 benchmarks (it clears 3) and the timings fit the budget; otherwise 0.5B.
- Training GPU: TBD
- Checkpoints that count as points (final only, or every N steps): TBD
- GRPO group size and prompts per step: TBD
- GRPO learning rates: TBD from the section D2 rerun. RL's Razor's grid of 1e-5 to 5e-5 collapsed to bare guesses on the 1.5B model (logged 07/11/26), so any other grid is a logged deviation.
- `max_new_tokens` and max completion length: 1024, for KL sampling, the new-task eval and GRPO alike. Logged: 7.7% of base responses on chemistry train prompts run past 512 tokens and 0.7% past 1024, and SFT targets reach 825.
- Checkpoint storage (Hugging Face repo, Drive, or compute every KL variant and then delete): TBD
- KL precision: fp32, confirmed on real smoke checkpoints. Logged: bf16 moves the primary KL by 5.7% on the GRPO checkpoint and 1.5% on the SFT one, an error that differs by method.

### Gate 1 budget criteria

- GPU hours and money per run, and the pilot total: TBD

### Task setup (logged)

- SciKnowEval commit 92ef969, Chemistry L3, all 2,400 rows.
- Split: 1,800 train, 600 held-out, 500 KL prompts from held-out, in `data/splits/chem_l3.json`.
- Prompt wording: frozen in `build_prompt` (`src/data/chem.py`). Every prompt asks to "Reason briefly in plain text, then finish with a final line in exactly this form:", followed by `Answer: <letter>` or `Answer: <balanced equation>`. Filling prompts also ask to balance the equation and keep the reactants and products in the order given.
- Share of train rows solved by GPT-4o: logged as 1,384 of 1,800 (76.9%), with 990 passing on the first try, after rescoring with the lenient matcher (1,381 and 989 before). SFT and GRPO both train on these rows.
- Answer extraction: the last answer label ("Answer:", "Final answer:", "The correct answer is:", "Final line:") wins, with the next line read if the label stands alone, and the last line read if there's no label. An MCQ answer is a letter A to D, alone or followed by its option's text. Logged 07/11/26.
