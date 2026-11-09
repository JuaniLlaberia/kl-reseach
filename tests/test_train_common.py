import time
from pathlib import Path

import pytest
import torch
import yaml
from transformers import TrainerControl

from src.train.common import StepTimer, check_fp32, folder_size_gb, load_config, summarize_log, training_arguments, verify_saved

CONFIGS = sorted(Path("configs").glob("*.yaml"))

def write_config(tmp_path, **changes) -> Path:
    config = {
        "run": "test_run", "method": "sft", "model": "m", "revision": None, "seed": 0, "learning_rate": 3e-5,
        "warmup_steps": 1, "max_steps": 2, "weight_decay": 0.0, "max_grad_norm": 1.0, "optim": "adamw_torch",
        "bf16": False, "gradient_checkpointing": False, "logging_steps": 1, "batch_size": 4, "micro_batch_size": 2, "max_length": 64,
    }
    config.update(changes)
    path = tmp_path / "test_run.yaml"
    path.write_text(yaml.safe_dump({key: value for key, value in config.items() if value != "DROP"}))
    return path

def test_valid_config_loads(tmp_path):
    assert load_config(write_config(tmp_path))["batch_size"] == 4

@pytest.mark.parametrize("changes", [
    {"run": "other_name"},
    {"method": "ppo"},
    {"learning_rate": "DROP"},
    {"unknown_setting": 1},
    {"micro_batch_size": 3},
    {"group_size": 2},
])
def test_bad_config_is_refused(tmp_path, changes):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, **changes))

def test_grpo_group_must_split_into_micro_batches(tmp_path):
    path = tmp_path / "grpo_run.yaml"
    path.write_text(yaml.safe_dump({
        "run": "grpo_run", "method": "grpo", "model": "m", "revision": None, "seed": 0, "learning_rate": 3e-5,
        "warmup_steps": 1, "max_steps": 2, "weight_decay": 0.0, "max_grad_norm": 1.0, "optim": "adamw_torch",
        "bf16": False, "gradient_checkpointing": False, "logging_steps": 1, "group_size": 3, "prompts_per_step": 3,
        "loss_batch_size": 2, "generation_chunk_size": 4, "max_completion_length": 8,
    }))
    with pytest.raises(ValueError, match="divisible"):
        load_config(path)

def test_there_are_configs_for_every_section():
    assert {path.stem for path in CONFIGS} == {
        "qwen05_chem_sft_memory_s0", "qwen05_chem_grpo_memory_s0", "qwen15_chem_sft_memory_s0",
        "qwen15_chem_grpo_memory_s0", "qwen15_chem_sft_smoke_s0", "qwen15_chem_grpo_smoke_s0",
        # Section D2, after the collapse
        "qwen15_chem_grpo_smoke_lr3e-6_s0", "qwen15_chem_grpo_memory_chunk512_s0",
    }

@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_committed_configs(path):
    config = load_config(path)
    assert config["revision"], "Models must be pinned to a commit."
    assert config["weight_decay"] == 0.0
    assert config["max_grad_norm"] == 1.0
    assert config["optim"] == "adamw_bnb_8bit"
    assert config["bf16"] is True
    assert config["gradient_checkpointing"] is True
    assert "use_cpu" not in config

def test_shared_settings_are_explicit(tmp_path):
    config = load_config(write_config(tmp_path))
    arguments = training_arguments(config, tmp_path, "none")
    assert arguments["lr_scheduler_type"] == "constant_with_warmup"
    assert arguments["learning_rate"] == config["learning_rate"]
    assert arguments["save_strategy"] == "no"

def test_check_fp32_refuses_bf16(chat_model):
    check_fp32(chat_model)
    with pytest.raises(ValueError, match="fp32"):
        check_fp32(chat_model.to(torch.bfloat16))

def test_verify_saved_passes_on_identical_weights(chat_model, tmp_path):
    chat_model.save_pretrained(tmp_path / "saved")
    verify_saved(chat_model, tmp_path / "saved")
    assert folder_size_gb(tmp_path / "saved") > 0

def test_verify_saved_catches_a_bf16_checkpoint(chat_model, tmp_path):
    chat_model.to(torch.bfloat16).save_pretrained(tmp_path / "saved")
    with pytest.raises(ValueError, match="fp32"):
        verify_saved(chat_model, tmp_path / "saved")

def test_verify_saved_catches_changed_weights(chat_model, tmp_path):
    chat_model.save_pretrained(tmp_path / "saved")
    with torch.no_grad():
        next(chat_model.parameters()).add_(1e-3)
    with pytest.raises(ValueError, match="changed"):
        verify_saved(chat_model, tmp_path / "saved")

def test_timer_skips_the_first_step():
    timer = StepTimer()
    timer.marks = [0.0, 100.0, 101.0, 102.0, 103.0]
    assert timer.seconds_per_step() == 1.0

def test_timer_stops_training_past_the_limit():
    control = TrainerControl()
    timer = StepTimer(time_limit_minutes=0)
    timer.on_train_begin(None, None, control)
    time.sleep(0.01)
    timer.on_step_end(None, None, control)
    assert control.should_training_stop is True
    assert timer.hit_time_limit is True

def test_timer_with_no_limit_never_stops_training():
    control = TrainerControl()
    timer = StepTimer()
    timer.on_train_begin(None, None, control)
    timer.on_step_end(None, None, control)
    assert control.should_training_stop is False

def test_timer_with_no_steps():
    assert StepTimer().seconds_per_step() is None

def test_summarize_log_reads_thirds_and_skips_entries_with_no_loss():
    history = [{"loss": value, "reward": 0.5, "frac_reward_zero_std": 0.25} for value in (3.0, 3.0, 2.0, 2.0, 1.0, 1.0)]
    history.append({"train_runtime": 10.0})
    summary = summarize_log(history)
    assert summary["loss_first_third"] == 3.0
    assert summary["loss_last_third"] == 1.0
    assert summary["reward_mean"] == 0.5
    assert summary["share_of_groups_with_no_reward_variation"] == 0.25
    assert summary["truncated_share"] is None
