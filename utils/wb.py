import os
import subprocess

import torch
import wandb
from dotenv import load_dotenv

def get_git_commit() -> str:
    """
    Gets the current git commit. Fails if there is none, as every run must be traceable.

    Returns:
        The full hash of the current commit.
    """
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def get_gpu_type() -> str:
    """
    Gets the name of the GPU used by this run.

    Returns:
        The GPU name, or "cpu" if there is no GPU available.
    """
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

def start_experiment(run_name: str, config: dict, tags: list[str], **kwargs):
    """
    Initialize a W&B run with a per-experiment config.

    Args:
        run_name (str): Name of the run (same as the config file name).
        config (dict): Dictionary containing the parameters of the experiment.
        tags (list[str]): Stage, family, task and method of the run.
    Returns:
        A wandb run object.
    """
    load_dotenv()
    wandb.login()
    full_config = {**config, "git_commit": get_git_commit(), "gpu": get_gpu_type()}
    return wandb.init(project=os.environ["WANDB_PROJECT"], name=run_name, config=full_config, tags=tags, **kwargs)

def log_metrics(run, metrics: dict) -> None:
    """
    Logs the given metrics for a given wandb run.

    Args:
        run: Initialized wandb run.
        metrics (dict): Dictionary containing metrics about the experiment.
    Returns:
        None.
    """
    run.log(metrics)
