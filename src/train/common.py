import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import TrainerCallback

COMMON_KEYS = {
    "run", "method", "model", "revision", "seed", "learning_rate", "warmup_steps", "max_steps",
    "weight_decay", "max_grad_norm", "optim", "bf16", "gradient_checkpointing", "logging_steps",
}
METHOD_KEYS = {
    "sft": {"batch_size", "micro_batch_size", "max_length"},
    "grpo": {"group_size", "prompts_per_step", "loss_batch_size", "generation_chunk_size", "max_completion_length"},
}
# Only for the tests, which train on the CPU of a laptop
OPTIONAL_KEYS = {"use_cpu"}

def load_config(path: Path) -> dict:
    """
    Loads the config of one run, and fails on any missing or unknown setting, so nothing falls back to a library default.

    Args:
        path (Path): Path of the YAML file, which is named after the run.
    Returns:
        The settings of the run.
    """
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("run") != path.stem:
        raise ValueError(f"{path.name} says its run is {config.get('run')}, but they must have the same name.")
    if config.get("method") not in METHOD_KEYS:
        raise ValueError(f"Unknown method: {config.get('method')}")

    expected = COMMON_KEYS | METHOD_KEYS[config["method"]]
    missing = expected - set(config)
    unknown = set(config) - expected - OPTIONAL_KEYS
    if missing or unknown:
        raise ValueError(f"{path.name} has missing settings {sorted(missing)} and unknown settings {sorted(unknown)}.")

    if config["method"] == "sft" and config["batch_size"] % config["micro_batch_size"] != 0:
        raise ValueError(f"{path.name}: batch_size must be divisible by micro_batch_size.")
    if config["method"] == "grpo" and config["group_size"] * config["prompts_per_step"] % config["loss_batch_size"] != 0:
        raise ValueError(f"{path.name}: group_size * prompts_per_step must be divisible by loss_batch_size.")
    return config

def check_fp32(model) -> None:
    """
    Fails unless every weight is in fp32. Near lr 1e-5, bf16 weights would round most Adam steps away.

    Args:
        model: Model to check.
    Returns:
        None.
    """
    wrong = {parameter.dtype for parameter in model.parameters() if parameter.dtype != torch.float32}
    if wrong:
        raise ValueError(f"Weights must be fp32 to train, but some are {wrong}.")

def training_arguments(config: dict, out_dir: Path, report_to: str) -> dict:
    """
    Builds the settings shared by both trainers. The learning rate is constant after warmup, as in RL's Razor.

    Args:
        config (dict): Settings of the run.
        out_dir (Path): Folder for the trainer's own files.
        report_to (str): "wandb" or "none".
    Returns:
        Keyword arguments for the trainer's config.
    """
    return {
        "output_dir": str(out_dir),
        "seed": config["seed"],
        "max_steps": config["max_steps"],
        "learning_rate": config["learning_rate"],
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": config["warmup_steps"],
        "weight_decay": config["weight_decay"],
        "max_grad_norm": config["max_grad_norm"],
        "optim": config["optim"],
        # Weights stay in fp32, and bf16 only applies inside the forward pass
        "bf16": config["bf16"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "logging_steps": config["logging_steps"],
        "save_strategy": "no",
        "report_to": report_to,
        "remove_unused_columns": False,
        "use_cpu": config.get("use_cpu", False),
    }

class StepTimer(TrainerCallback):
    """Records when each optimizer step ends, including GRPO's generation, and stops a run that passes its time limit."""

    def __init__(self, time_limit_minutes: float | None = None):
        self.marks = []
        self.time_limit_minutes = time_limit_minutes
        self.hit_time_limit = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.marks = [time.perf_counter()]

    def on_step_end(self, args, state, control, **kwargs):
        self.marks.append(time.perf_counter())
        if self.time_limit_minutes is not None and self.marks[-1] - self.marks[0] > self.time_limit_minutes * 60:
            self.hit_time_limit = True
            control.should_training_stop = True

    def seconds_per_step(self) -> float | None:
        """
        Gets the typical duration of a step. The first one is skipped when possible, as it also sets things up.

        Returns:
            The median seconds per step, or None if no step finished.
        """
        gaps = [end - start for start, end in zip(self.marks, self.marks[1:])]
        if not gaps:
            return None
        return statistics.median(gaps[1:] or gaps)

def verify_saved(model, path: Path) -> None:
    """
    Fails unless a saved checkpoint reloads in fp32 with exactly the weights of the model in memory.

    Args:
        model: Model that was saved.
        path (Path): Folder of the checkpoint.
    Returns:
        None.
    """
    from transformers import AutoModelForCausalLM

    # With no dtype, this keeps the precision of the saved files instead of converting them
    reloaded = AutoModelForCausalLM.from_pretrained(path, dtype="auto")
    check_fp32(reloaded)
    saved = reloaded.state_dict()
    for name, weight in model.state_dict().items():
        if not torch.equal(weight.cpu(), saved[name]):
            raise ValueError(f"{name} changed when the checkpoint was saved.")

def folder_size_gb(path: Path) -> float:
    """
    Gets the size of a folder.

    Args:
        path (Path): Folder.
    Returns:
        Its size in GB.
    """
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file()) / 1e9

def mean_of(entries: list[dict], key: str) -> float | None:
    """
    Averages one logged value.

    Args:
        entries (list[dict]): Log entries.
        key (str): Name of the value.
    Returns:
        The mean, or None if it was never logged.
    """
    values = [entry[key] for entry in entries if key in entry]
    return sum(values) / len(values) if values else None

def summarize_log(history: list[dict]) -> dict:
    """
    Summarizes a trainer's log for the smoke checks. GRPO's loss isn't expected to fall, as it isn't a measure of progress.

    Args:
        history (list[dict]): The trainer's log_history.
    Returns:
        The loss in the first and last third of the run, and the means of GRPO's reward and completion values.
    """
    entries = [entry for entry in history if "loss" in entry]
    third = max(1, len(entries) // 3)
    return {
        "loss_first_third": mean_of(entries[:third], "loss"),
        "loss_last_third": mean_of(entries[-third:], "loss"),
        "grad_norm_mean": mean_of(entries, "grad_norm"),
        "reward_mean": mean_of(entries, "reward"),
        "share_of_groups_with_no_reward_variation": mean_of(entries, "frac_reward_zero_std"),
        "completion_length_mean": mean_of(entries, "completions/mean_length"),
        "completion_length_max": max((entry["completions/max_length"] for entry in entries if "completions/max_length" in entry), default=None),
        "truncated_share": mean_of(entries, "completions/clipped_ratio"),
        "clipped_token_share": mean_of(entries, "clip_ratio/region_mean"),
    }
