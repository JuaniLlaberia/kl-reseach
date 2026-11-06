# KL and forgetting study

## What this project is
Tests whether the KL divergence between a fine-tuned model and its base, measured on new-task prompts, predicts how much the model forgets after SFT and GRPO. The claim comes from RL's Razor (Shenfeld, Pari, Agrawal, ICLR 2026). This study tests it across model families (Qwen, Llama, OLMo or SmolLM, 0.5–1.7B) and across ways of measuring KL. Tasks: SciKnowEval Chemistry L-3 and ToolAlpaca. Full fine-tuning only.

Each module folder has its own CLAUDE.md with its spec (`src/kl/`, later `src/train/` and `src/eval/`).

## Ground rules
- From Stage 2 on, Claude drafts the code and I (Juan) review it in batches. Explain every choice that touches the math. `src/kl/` is my code: when you change it or any other math code, explain the change; don't silently rewrite it.
- The primary metric, the KL variant names, the data splits and the analyses are pre-registered in `docs/PREREG.md` (frozen 17 Nov 2026). Never change them unless I ask. If a change would affect them, stop and say so.
- Never choose KL variants, checkpoints or analyses based on how well they fit.
- No LoRA. No KL penalty in GRPO training (coefficient 0, Dr. GRPO loss).
- Versions are pinned in `requirements.txt`. Don't upgrade packages.
- The same setup applies to every checkpoint: harness version, prompts, chat template, KL prompts and precision.

## Layout
```
configs/       one YAML per training run, named like qwen05_chem_sft_lr3e-5_s0.yaml
results/<run>/ one JSON per checkpoint (final.json, step500.json)
data/splits/   seeded train/held-out row IDs per task
data/solutions/ GPT-4o solutions and the log of every attempt
docs/PREREG.md frozen pre-registration (source of truth for every spec)
docs/EXPERIMENTS_LOG.md      append-only lab log; never edit old entries
notebooks/     one Colab notebook per stage; every run happens here
src/data/      task data, prompts and GPT-4o solutions
src/kl/        KL module
src/train/     SFT and GRPO trainers
src/eval/      answer extraction, new-task eval and retention suite
tests/         pytest
```

## Conventions
- **Run name** = config file name; use it for the results folder and the W&B run.
- **Results JSON** fields: run, checkpoint, git_commit, harness_version, gpu, new_task_acc, raw retention scores per benchmark (never drops), and every KL variant under its full name.
- Analysis scripts read `results/` only, never W&B.
- **W&B** is for training curves. At `wandb.init`, log the config, git commit and GPU type, and tag the run with stage, family, task and method.

## Commands
- Tests (CPU, no network, no downloads): `pytest tests/`
- GPU tests (Colab): `pytest tests/ -m gpu`
- Tests that download real data: `pytest tests/ -m network`
- Training, eval and KL all run from `notebooks/`, starting from a pushed commit.