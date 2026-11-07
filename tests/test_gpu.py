import pytest
import torch
import yaml

from src.train.run import run_config
from tests.test_grpo import grpo_config
from tests.test_sft import sft_config

pytestmark = pytest.mark.gpu

def on_gpu(config: dict) -> dict:
    # Runs the real training settings, which a laptop can't
    changes = {"optim": "adamw_bnb_8bit", "bf16": True, "gradient_checkpointing": True, "learning_rate": 1e-3}
    return {key: value for key, value in {**config, **changes}.items() if key != "use_cpu"}

@pytest.mark.parametrize("config", [sft_config(max_steps=3), grpo_config(max_steps=2)], ids=["sft", "grpo"])
def test_real_settings_run_and_save_in_fp32(config, model_dir, solved_rows, tmp_path):
    path = tmp_path / f"{config['run']}.yaml"
    path.write_text(yaml.safe_dump({**on_gpu(config), "model": str(model_dir)}))
    summary = run_config(path, solved_rows, tmp_path / "results", tmp_path / "checkpoints", use_wandb=False)
    assert summary["status"] == "finished"
    assert summary["peak_memory_gb"]["allocated"] > 0

def test_autocast_runs_in_bf16_while_weights_stay_in_fp32(chat_tokenizer, chat_model, solved_rows, tmp_path):
    from src.train.sft import build_sft_trainer

    seen = []
    trainer = build_sft_trainer({**on_gpu(sft_config(max_steps=1))}, chat_model.to("cuda"), chat_tokenizer, solved_rows, tmp_path)
    # A linear layer, as autocast only changes matmuls: a decoder layer's output stays fp32, because the residual adds fp32 to bf16
    handle = trainer.model.model.layers[0].self_attn.q_proj.register_forward_hook(lambda module, inputs, output: seen.append(output.dtype))
    trainer.train()
    handle.remove()
    assert seen and seen[0] == torch.bfloat16
    assert {parameter.dtype for parameter in trainer.model.parameters()} == {torch.float32}
    assert trainer.args.optim.value == "adamw_bnb_8bit"
