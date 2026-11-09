# KL and forgetting study

> **Status: stopped on 9 Nov 2026, before the pilot.** Other projects took priority. No confirmatory experiment was run, so this repo holds no results about the research question. The predictions in `docs/PREREG.md` were written down and tagged (`prereg-draft`) but never tested.

## The question

[RL's Razor](https://arxiv.org/abs/2509.04259) (Shenfeld, Pari and Agrawal, ICLR 2026) claims that the KL divergence between a fine-tuned model and its base, measured on the new task, predicts how much the model forgets, and that this explains why on-policy RL forgets less than SFT. This study was designed to test that claim outside its original setting:

- **RQ1:** Does new-task KL predict forgetting in small LLMs (0.5–1.7B), across model families, for both SFT and GRPO?
- **RQ2:** How much does the fit depend on how KL is measured: direction, sample source, aggregation and estimator?
- **RQ3:** Does GRPO forget less than SFT at matched new-task accuracy?

The design was pre-registered: primary metric, forgetting score, near-chance rule, predictions and gate thresholds were fixed before any training (see `docs/PREREG.md`). Full fine-tuning only, with no KL penalty in GRPO.

## How far it got

| Stage | Status |
|---|---|
| 1. Measurement tools: KL module, retention suite, base scores | Done |
| 2. Data, trainers, pipeline checks, pre-registration | About two thirds done |
| 3. Pilot and Gate 1 | Not started |

Every decision and deviation is in `docs/EXPERIMENTS_LOG.md`, a dated, append-only lab log.

## What works and can be reused

- **KL module** (`src/kl/`): exact per-token KL over the full vocabulary, forward and reverse, on base or fine-tuned samples. It covers response tokens only, runs in fp32, and computes both a mean-per-token and a sum-per-sequence aggregation. It also includes the k1, k2 and k3 estimators. The sampler overrides every generation default explicitly, because `generate` otherwise inherits each model's own temperature, top-p, top-k and repetition penalty. Tested: KL(base ‖ base) = 0, KL is never negative, and k1 matches the exact KL.
- **Retention suite** (`src/eval/retention.py`): HellaSwag, TruthfulQA-MC2, Winogrande (5-shot), MMLU (5-shot), IFEval and HumanEval-Instruct through lm-evaluation-harness. It runs in fp32 with the chat template, with bootstrap CIs and minutes per benchmark.
- **Trainers** (`src/train/`): SFT with loss on response tokens only, and Dr. GRPO through TRL with binary reward, no format reward and generation in chunks. Both keep fp32 master weights with bf16 autocast, and check that a saved checkpoint reloads with identical weights.
- **Chemistry task** (`src/data/`, `data/splits/`): SciKnowEval Chemistry L3 with a seeded, stratified train/held-out split, plus a GPT-4o rejection-sampling pipeline through OpenRouter with a budget cap.
- **Answer matcher** (`src/eval/extract.py`): one definition of "correct", shared by solution acceptance, the GRPO reward and the eval, with tests built from real model outputs.

## What Stage 2 found

These are pipeline observations from smoke runs, not findings about the research question.

- **GRPO collapsed at RL's Razor's learning rates.** On Qwen2.5-1.5B-Instruct (chemistry, 1e-5 to 5e-5 as in the paper), the mean response length fell from about 180 tokens to 5 within 15–40 steps, and entropy fell from 0.96 to 0.05. Reward settled at about 0.25, chance on four choices. One possible reason, inferred and not confirmed: RL's Razor may have stored weights in bf16, so most small updates rounded away, while this setup keeps fp32 master weights. A rerun at lr 3e-6 was prepared (notebook section D2) but not run.
- **bf16 biases KL differently by method.** On real smoke checkpoints, bf16 moved the primary KL by 5.7% for GRPO and 1.5% for SFT. That would bias an SFT-vs-GRPO comparison, so KL was kept in fp32.
- **Small models fail answer formats.** 54% of Qwen2.5-1.5B-Instruct's chemistry MCQ answers failed a strict "Answer: X" matcher, mostly on format. Among answers it did state clearly, it was right 31% of the time, against 25% chance. A strict matcher makes new-task accuracy partly a format score, which favors SFT, because SFT learns the format from its targets.
- **A 512-token cap is too short** for chemistry reasoning: 7.7% of base responses ran past it.
- **Costs on a Colab G4** (RTX PRO 6000, about 10 units per hour):
  - Retention suite: about 7 units per 1.5B checkpoint.
  - SFT: about 0.2 units per pass over 1,384 rows.
  - GRPO at 64×8: about 43 units per pass.
  
  Evaluating each checkpoint costs more than SFT training, so evaluation sets the floor on how many checkpoints a budget allows.

## Resuming

The next step was notebook section D2: a GRPO rerun at lr 3e-6 with a 1024-token cap and the lenient matcher, plus a timing test at 64×8 in one generate call. Then decide the task. `KL Forgetting Study Revised Plan.md` proposes switching the anchor task to math, with a star design (one anchor, one change per cell) and a learnability gate before training. It was never adopted.

To resume, you'll need two things that are kept outside git:
- **The GPT-4o solutions and the log of every attempt.** They aren't in the repo, because OpenAI's terms on sharing outputs haven't been checked. Without them, the solutions have to be regenerated, for about $15.
- **The Stage 2 training results** (memory, smoke, lengths, drift), which are on Google Drive. Their summaries are in the lab log.

## Layout

```
configs/        one YAML per training run; the file name is the run name
data/splits/    seeded train/held-out row IDs (chemistry)
data/solutions/ summary of the GPT-4o run (solutions and attempt logs are git-ignored)
docs/           PREREG.md (pre-registration), EXPERIMENTS_LOG.md (lab log), PLAN.md, CONTEXT.md
notebooks/      one Colab notebook per stage; every run happened there
results/        one JSON per checkpoint: base retention for Qwen2.5-0.5B/1.5B-Instruct, and the mini-pilot
src/kl/         KL module
src/eval/       answer extraction, new-task eval, retention suite, mini-pilot checks
src/train/      SFT and GRPO trainers
src/data/       chemistry data and GPT-4o solutions
tests/          pytest
```

## Setup

Versions are pinned in `requirements.txt`: torch 2.15.0, transformers 5.22.1, TRL 1.17.1 and lm-eval 0.4.13. Copy `.env.example` to `.env` for the W&B and OpenRouter keys; on Colab, they go in Colab secrets.

```bash
pip install -r requirements.txt
pytest tests/                 # CPU, no network
pytest tests/ -m network      # downloads real data
pytest tests/ -m gpu          # on a GPU
```

Training, evaluation and KL runs were done from `notebooks/` on Colab, always starting from a pushed commit.
