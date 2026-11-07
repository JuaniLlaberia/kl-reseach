import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.kl.measure import load_model
from src.train.common import StepTimer, check_fp32, folder_size_gb, load_config, summarize_log, verify_saved
from src.train.grpo import build_grpo_trainer
from src.train.sft import build_sft_trainer

BUILDERS = {"sft": build_sft_trainer, "grpo": build_grpo_trainer}
# Colab's rates, as logged in Stage 1
UNITS_PER_HOUR = {"RTX PRO 6000": 10.0, "L4": 1.94}

def peak_memory_gb() -> dict[str, float | None]:
    """
    Gets the most GPU memory used since the last reset.

    Returns:
        The peak allocated and reserved memory in GB, or None without a GPU.
    """
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    return {"allocated": torch.cuda.max_memory_allocated() / 1e9, "reserved": torch.cuda.max_memory_reserved() / 1e9}

def gpu_tag() -> str:
    """
    Gets a short name for the current GPU, used to keep the timings of each GPU apart.

    Returns:
        g4, l4, cpu or the GPU's name in lowercase with no spaces.
    """
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" in name:
        return "g4"
    return "l4" if "L4" in name else name.lower().replace(" ", "_")

def run_config(
    config_path: Path,
    solved_rows: list[dict],
    results_root: Path,
    checkpoints_root: Path,
    use_wandb: bool = True,
    save: bool = True,
    time_limit_minutes: float | None = None,
) -> dict:
    """
    Runs one training config. Running out of memory is saved as a result, since finding that out is what some runs are for.

    Args:
        config_path (Path): Path of the YAML config.
        solved_rows (list[dict]): Rows with a solution.
        results_root (Path): Folder for the summary and log of each run.
        checkpoints_root (Path): Folder for the checkpoint of each run.
        use_wandb (bool): Whether to log the run to W&B.
        save (bool): Whether to save and verify the final checkpoint.
        time_limit_minutes (float | None): Stops the run at the next step after this long, to protect the unit budget.
    Returns:
        The summary of the run, which is also saved to summary.json.
    """
    config = load_config(config_path)
    device = "cpu" if config.get("use_cpu") or not torch.cuda.is_available() else "cuda"
    results_dir = Path(results_root) / config["run"]
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    model = load_model(config["model"], torch.float32, device, config["revision"])
    check_fp32(model)

    run = None
    if use_wandb:
        from utils.wb import start_experiment

        run = start_experiment(config["run"], config, ["stage-2", config["method"], "chemistry"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = BUILDERS[config["method"]](config, model, tokenizer, solved_rows, results_dir / "trainer", "wandb" if use_wandb else "none")
    timer = StepTimer(time_limit_minutes)
    trainer.add_callback(timer)

    status = "finished"
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        status = "out_of_memory"

    if timer.hit_time_limit:
        status = "stopped_at_time_limit"

    checkpoint_gb = None
    if status != "out_of_memory" and save:
        checkpoint_dir = Path(checkpoints_root) / config["run"] / "final"
        trainer.save_model(str(checkpoint_dir))
        verify_saved(trainer.model, checkpoint_dir)
        checkpoint_gb = folder_size_gb(checkpoint_dir)

    summary = {
        "run": config["run"],
        "status": status,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": config,
        "steps_finished": trainer.state.global_step,
        "seconds_per_step": timer.seconds_per_step(),
        "peak_memory_gb": peak_memory_gb(),
        "checkpoint_gb": checkpoint_gb,
        "log": summarize_log(trainer.state.log_history),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "trainer_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    if run is not None:
        run.finish()
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

def project_run(summary: dict, n_rows: int) -> dict:
    """
    Extrapolates a timed run to one pass through the training rows. It's rough, as it assumes later steps take as long.

    Args:
        summary (dict): Summary of a timed run.
        n_rows (int): Number of training rows.
    Returns:
        The steps per pass, hours and Colab units of one pass, with None for the units on an unknown GPU.
    """
    config = summary["config"]
    rows_per_step = config["batch_size"] if config["method"] == "sft" else config["prompts_per_step"]
    steps = n_rows // rows_per_step
    hours = steps * summary["seconds_per_step"] / 3600 if summary["seconds_per_step"] is not None else None
    rate = next((rate for name, rate in UNITS_PER_HOUR.items() if name in summary["gpu"]), None)
    return {"steps_per_pass": steps, "hours": hours, "units": hours * rate if hours is not None and rate is not None else None}
