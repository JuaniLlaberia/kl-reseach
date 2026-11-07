import json

import pytest
import torch
import yaml

from src.kl.sampling import build_prompt_ids, extract_response, get_stop_ids
from src.data.chem import build_prompt
from src.train.run import run_config
from src.train.sft import IGNORE_INDEX, build_sft_rows, build_sft_trainer, collate_sft

def sft_config(**changes) -> dict:
    config = {
        "run": "tiny_sft", "method": "sft", "model": "m", "revision": None, "seed": 0, "learning_rate": 1e-2,
        "warmup_steps": 1, "max_steps": 8, "weight_decay": 0.0, "max_grad_norm": 1.0, "optim": "adamw_torch",
        "bf16": True, "gradient_checkpointing": False, "logging_steps": 1, "batch_size": 4, "micro_batch_size": 2,
        "max_length": 256, "use_cpu": True,
    }
    return {**config, **changes}

def test_prompt_tokens_are_masked_and_response_tokens_are_not(chat_tokenizer, solved_rows):
    example = build_sft_rows(chat_tokenizer, solved_rows[:1], chat_tokenizer.eos_token_id, 256)[0]
    prompt_ids = build_prompt_ids(chat_tokenizer, [{"prompt_id": 0, "text": build_prompt(solved_rows[0])}])[0]["prompt_ids"]
    assert example["labels"][:len(prompt_ids)] == [IGNORE_INDEX] * len(prompt_ids)
    assert IGNORE_INDEX not in example["labels"][len(prompt_ids):]
    assert example["input_ids"][:len(prompt_ids)] == prompt_ids

def test_labels_end_at_the_first_stop_token_like_a_sampled_response(chat_tokenizer, chat_model, solved_rows):
    example = build_sft_rows(chat_tokenizer, solved_rows[:1], chat_tokenizer.eos_token_id, 256)[0]
    response = [label for label in example["labels"] if label != IGNORE_INDEX]
    stop_ids = get_stop_ids(chat_model)
    # The chat template puts a newline after the stop token, which must not be a target
    assert response[-1] == chat_tokenizer.eos_token_id
    assert extract_response(response, stop_ids) == (response, False)
    assert chat_tokenizer.decode(response, skip_special_tokens=True) == solved_rows[0]["solution"]

def test_too_long_rows_fail_instead_of_being_cut(chat_tokenizer, solved_rows):
    with pytest.raises(ValueError, match="longer than max_length"):
        build_sft_rows(chat_tokenizer, solved_rows, chat_tokenizer.eos_token_id, 10)

def test_padding_is_masked():
    batch = collate_sft([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}, {"input_ids": [4], "labels": [4]}], pad_id=9)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 9, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]

def test_loss_is_the_mean_over_response_tokens_only(chat_tokenizer, chat_model, solved_rows):
    examples = build_sft_rows(chat_tokenizer, solved_rows[:2], chat_tokenizer.eos_token_id, 256)
    batch = collate_sft(examples, chat_tokenizer.pad_token_id)
    chat_model.eval()
    with torch.no_grad():
        output = chat_model(**batch)
    # The token at position t is predicted by the logits at position t - 1
    log_probs = torch.log_softmax(output.logits[:, :-1], dim=-1)
    targets = batch["labels"][:, 1:]
    keep = targets != IGNORE_INDEX
    manual = -log_probs[keep].gather(1, targets[keep].unsqueeze(1)).mean()
    assert output.loss.item() == pytest.approx(manual.item(), rel=1e-5)

def test_trainer_uses_the_batch_settings(chat_tokenizer, chat_model, solved_rows, tmp_path):
    trainer = build_sft_trainer(sft_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)
    assert trainer.args.per_device_train_batch_size == 2
    assert trainer.args.gradient_accumulation_steps == 2
    assert trainer.args.weight_decay == 0.0
    assert trainer.args.bf16 is True

def test_refuses_a_tokenizer_whose_end_token_never_stops_the_model(chat_tokenizer, chat_model, solved_rows, tmp_path):
    chat_model.generation_config.eos_token_id = [chat_tokenizer.pad_token_id]
    with pytest.raises(ValueError, match="stop tokens"):
        build_sft_trainer(sft_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)

def test_weights_stay_fp32_while_training_and_every_weight_is_trainable(chat_tokenizer, chat_model, solved_rows, tmp_path):
    before = {name: weight.clone() for name, weight in chat_model.state_dict().items()}
    trainer = build_sft_trainer(sft_config(max_steps=2), chat_model, chat_tokenizer, solved_rows, tmp_path)
    assert all(parameter.requires_grad for parameter in trainer.model.parameters())
    trainer.train()
    assert {parameter.dtype for parameter in trainer.model.parameters()} == {torch.float32}
    assert all(not torch.equal(before[name], weight) for name, weight in trainer.model.named_parameters())

def test_run_config_end_to_end(model_dir, solved_rows, tmp_path):
    path = tmp_path / "tiny_sft.yaml"
    path.write_text(yaml.safe_dump(sft_config(model=str(model_dir), max_steps=30)))
    summary = run_config(path, solved_rows, tmp_path / "results", tmp_path / "checkpoints", use_wandb=False)
    assert summary["status"] == "finished"
    assert summary["steps_finished"] == 30
    assert summary["seconds_per_step"] > 0
    assert summary["checkpoint_gb"] > 0
    # The tiny model memorizes one repeated target, so its loss has to fall
    assert summary["log"]["loss_last_third"] < summary["log"]["loss_first_third"]
    assert json.loads((tmp_path / "results" / "tiny_sft" / "summary.json").read_text())["run"] == "tiny_sft"
    assert (tmp_path / "checkpoints" / "tiny_sft" / "final" / "config.json").exists()

def test_time_limit_stops_the_run_and_still_saves(model_dir, solved_rows, tmp_path):
    path = tmp_path / "tiny_sft.yaml"
    path.write_text(yaml.safe_dump(sft_config(model=str(model_dir), max_steps=50)))
    summary = run_config(path, solved_rows, tmp_path / "results", tmp_path / "checkpoints", use_wandb=False, time_limit_minutes=0)
    assert summary["status"] == "stopped_at_time_limit"
    assert summary["steps_finished"] == 1
    assert summary["checkpoint_gb"] > 0

def test_running_out_of_memory_is_saved_as_a_result(model_dir, solved_rows, tmp_path, monkeypatch):
    from transformers import Trainer

    def fail(self, *args, **kwargs):
        raise torch.cuda.OutOfMemoryError("no memory")

    monkeypatch.setattr(Trainer, "train", fail)
    path = tmp_path / "tiny_sft.yaml"
    path.write_text(yaml.safe_dump(sft_config(model=str(model_dir))))
    summary = run_config(path, solved_rows, tmp_path / "results", tmp_path / "checkpoints", use_wandb=False)
    assert summary["status"] == "out_of_memory"
    assert summary["checkpoint_gb"] is None

@pytest.mark.network
def test_real_qwen_tokenizer_labels_match_a_sampled_response():
    from transformers import AutoTokenizer, GenerationConfig

    name, revision = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    stop_ids = GenerationConfig.from_pretrained(name, revision=revision).eos_token_id
    assert tokenizer.eos_token_id in stop_ids
    row = {
        "row_id": 0, "type": "mcq-4-choices", "question": "Which one?", "answerKey": "B",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]}, "solution": "It's the second.\nAnswer: B",
    }
    example = build_sft_rows(tokenizer, [row], tokenizer.eos_token_id, 512)[0]
    response = [label for label in example["labels"] if label != IGNORE_INDEX]
    assert tokenizer.decode(response) == row["solution"] + "<|im_end|>"
    assert extract_response(response, stop_ids) == (response, False)
