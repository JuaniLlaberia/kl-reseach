# KL Forgetting Study: Revised Plan

Nov 7, 2026 · @Juan Ignacio Llaberia

## Question and what changed

The question stays the same: does new-task forward KL predict forgetting, as RL's Razor claims, and where does it break? Only the design changes, so that a failure can be explained.

1. **RQ1:** Does the law hold in small LLMs, starting from a setting close to the original?
2. **RQ2:** How much does the fit depend on how KL is measured?
3. **RQ3:** Does RL forget less than SFT at matched new-task accuracy, in each setting?

|  | Plan of 2 Nov | Revised |
| --- | --- | --- |
| Tasks | Chemistry QA + tool use | Math + tool use (chemistry showed no learning at 0.5B/1.5B) |
| Design | Full grid: every family × every task | Star: one anchor close to the original, then one change at a time |
| Configs | 10–15 per method | \~8 per method |
| Runs | 120–180 | \~80, plus an optional 16 |
| Families, sizes, methods, KL variants | Qwen, Llama, OLMo/SmolLM · 0.5–1.7B · SFT + GRPO, full fine-tuning · every variant | Unchanged |

## Design

Five cells, each differing from the anchor in one thing, so every difference in the result traces to one change. Each cell runs SFT and GRPO with \~8 configs each, and every KL variant is computed on every checkpoint.

| Cell | Model | Task | What it tests | Runs |
| --- | --- | --- | --- | --- |
| Anchor | Qwen 2.5 1.5B Instruct | Math | Closest affordable match to the original (3B) | \~16 |
| Size | Qwen 2.5 0.5B Instruct | Math | Does the law weaken at smaller size? | \~16 |
| Family A | Llama 3.2 1B | Math | Does it hold outside Qwen? | \~16 |
| Family B | OLMo 2 1B or SmolLM2 1.7B | Math | Same, second family | \~16 |
| Task | Qwen 2.5 1.5B Instruct | Tool use | Does it hold on a second task? | \~16 |
| Optional | Llama 3.2 1B | Tool use | Family × task interaction (a small 2×2) | \~16, only if budget allows |

Rules fixed before the pilot:

- The anchor uses the original's math task and its exact KL definition as the primary metric.
- Math is excluded from the retention suite while math is the new task.
- On math, report strict and lenient accuracy separately, so format gains are not counted as learning.
- The optional cell is pre-registered now and run only if Gate 1 timings leave room.

## Timeline and gates

The dates stay as planned; a new Gate 0 at the end of November checks that each task is learnable before any training.

&#91;embedded content: study roadmap · 4 phases, 3 gates\]

Each gate is pass or fail against thresholds you write down before the pilot:

- **Gate 0 (end of November):** base pass@8 on new-task prompts for each family; a task where too few prompts get any correct sample is swapped. The novelty re-check finds no study that already tests the law across families and KL definitions. Eval and KL cost per checkpoint is measured.
- **Gate 1 (Dec 7):** strict new-task accuracy in the anchor rises by at least \_\_\_ points on a held-out set, for both SFT and GRPO. The RL-vs-SFT forgetting gap is visible. Hours per run × \~80 runs fits free hours plus the budget. If it fails: drop 0.5B or filter prompts to the learnable range (pre-registered); if that still fails, plan B.
- **Gate 2 (Jan 31):** enough spread in KL and forgetting within each cell to fit curves.

## Floor and plan B

The floor is the anchor plus one other family on math, SFT and GRPO with 6–8 configs each, and every KL variant. If even that doesn't fit, switch to plan B instead of cutting further.

**Plan B, in order of preference:**

1. **KL measurement study, no training.** Compute every KL variant and the forgetting between public checkpoints from different training stages (check which families release them; OLMo is the first to check). Keeps RQ2, loses RQ1 and RQ3 control.
2. **Short report or workshop paper** on the pilot results and the KL-variant comparison.
3. **Reproduction report** of RL's Razor through a reproducibility challenge such as MLRC (check current deadlines).

**If stopping entirely:** close it in one to two weeks. Public repo with a README (status, why it stopped, what works), the pilot results, the pre-registration, a license and a tagged release. Write a few lines for yourself on why you stopped.

## What to verify yourself, not from AI

Several parts of this plan came from Claude's memory or estimates, not from sources you've checked. Confirm each one before it goes into the pre-registration.

- [ ] RL's Razor: which math task and dataset it used, and its exact KL computation (experimental section and appendices).
- [ ] The conflicting results (R² 0.52, no association, negative) and how each paper measured KL: your comparison table from the readings.
- [ ] Which families release per-stage checkpoints, for plan B.
- [ ] Access and licenses for Llama 3.2 1B, OLMo 2 1B and SmolLM2 1.7B.
- [ ] Memory and time per run: the \~16 bytes per parameter estimate is a rule of thumb; measure it in the pilot.
- [ ] Compute grant programs currently open (e.g. TPU Research Cloud, academic credits).
- [ ] Every threshold in the gates: set them yourself, in writing, before the pilot.

Write the pre-registration yourself, then use AI to review it, not to write it.
