import pytest
import torch
import yaml
from trl import GRPOTrainer

from src.data.chem import build_prompt
from src.eval.new_task import score_responses
from src.kl.sampling import build_prompt_ids, check_unmodified_sampling, get_stop_ids
from src.train.grpo import build_grpo_args, build_grpo_dataset, build_grpo_trainer, build_reward
from src.train.run import run_config
from tests.conftest import make_row

def grpo_config(**changes) -> dict:
    config = {
        "run": "tiny_grpo", "method": "grpo", "model": "m", "revision": None, "seed": 0, "learning_rate": 1e-2,
        "warmup_steps": 1, "max_steps": 2, "weight_decay": 0.0, "max_grad_norm": 1.0, "optim": "adamw_torch",
        "bf16": False, "gradient_checkpointing": False, "logging_steps": 1, "group_size": 4, "prompts_per_step": 2,
        "loss_batch_size": 4, "generation_chunk_size": 3, "max_completion_length": 8, "use_cpu": True,
    }
    return {**config, **changes}

def fake_generation(tokenizer, n_correct_per_group: int, group_size: int):
    """Replaces TRL's sampling: each group gets n_correct completions answering B, and the rest answering C."""
    calls = []

    def generate(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        calls.append(len(prompt_ids))
        offset = sum(calls[:-1])
        completions = []
        for index in range(len(prompt_ids)):
            letter = "B" if (index + offset) % group_size < n_correct_per_group else "C"
            completions.append(tokenizer(f"Answer: {letter}", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id])
        return completions, None, None

    return generate, calls

def test_settings_are_dr_grpo_and_not_trls_defaults(tmp_path):
    args = build_grpo_args(grpo_config(), tmp_path)
    assert args.loss_type == "dr_grpo"
    assert args.scale_rewards == "none"
    assert args.beta == 0.0
    assert args.mask_truncated_completions is False
    assert args.top_entropy_quantile == 1.0
    assert args.reward_weights is None

def test_sampling_settings_match_the_kl_measurement(tmp_path):
    from src.kl.sampling import SAMPLING_SETTINGS

    args = build_grpo_args(grpo_config(), tmp_path)
    assert args.temperature == SAMPLING_SETTINGS["temperature"]
    assert args.top_p == SAMPLING_SETTINGS["top_p"]
    assert args.top_k == SAMPLING_SETTINGS["top_k"]
    assert args.repetition_penalty == SAMPLING_SETTINGS["repetition_penalty"]
    assert args.min_p is None

def test_one_generation_per_optimizer_step(tmp_path):
    args = build_grpo_args(grpo_config(group_size=64, prompts_per_step=8, loss_batch_size=16), tmp_path)
    assert args.generation_batch_size == 512
    assert args.gradient_accumulation_steps == args.steps_per_generation == 32
    assert args.num_iterations == 1

def test_generate_leaves_the_logits_unmodified_despite_qwens_settings(chat_tokenizer, chat_model, solved_rows, tmp_path):
    assert chat_model.generation_config.top_k == 20
    trainer = build_grpo_trainer(grpo_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)
    prompt_ids = chat_tokenizer("Answer: B", add_special_tokens=False)["input_ids"]
    check_unmodified_sampling(chat_model, prompt_ids, trainer.generation_config)

def test_prompt_ids_match_the_kl_measurement(chat_tokenizer, chat_model, solved_rows, tmp_path):
    trainer = build_grpo_trainer(grpo_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)
    prompts = [row["prompt"] for row in build_grpo_dataset(solved_rows[:2])]
    grpo_ids, _, _ = trainer._tokenize_prompts(prompts)
    kl_ids = build_prompt_ids(chat_tokenizer, [{"prompt_id": row["row_id"], "text": build_prompt(row)} for row in solved_rows[:2]])
    assert [list(ids) for ids in grpo_ids] == [prompt["prompt_ids"] for prompt in kl_ids]

def test_grpo_stops_at_the_same_tokens_as_kl(chat_tokenizer, chat_model, solved_rows, tmp_path):
    trainer = build_grpo_trainer(grpo_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)
    assert set(trainer.eos_token_ids) == set(get_stop_ids(chat_model))

def test_refuses_to_stop_at_tokens_kl_does_not(chat_tokenizer, chat_model, solved_rows, tmp_path):
    chat_model.generation_config.eos_token_id = [chat_tokenizer.pad_token_id]
    with pytest.raises(ValueError, match="stops at"):
        build_grpo_trainer(grpo_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)

def test_dataset_never_contains_the_solution(solved_rows):
    dataset = build_grpo_dataset(solved_rows)
    assert set(dataset.column_names) == {"prompt", "gold", "answer_type"}
    assert solved_rows[0]["solution"] not in str(dataset[0]["prompt"])

def test_full_fine_tuning_with_no_adapter(chat_tokenizer, chat_model, solved_rows, tmp_path):
    trainer = build_grpo_trainer(grpo_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)
    assert trainer.model is chat_model
    assert all(parameter.requires_grad for parameter in trainer.model.parameters())
    assert trainer.ref_model is None

def test_reward_is_binary_with_no_format_reward():
    reward = build_reward([9])
    completions = [[{"role": "assistant", "content": text}] for text in ("Answer: B", "**Answer: B**", "Answer: C", "no answer")]
    values = reward(None, completions, [[1, 9]] * 4, ["B"] * 4, ["mcq-4-choices"] * 4)
    assert values == [1.0, 1.0, 0.0, 0.0]

def test_reward_is_zero_when_the_length_cap_cut_the_response():
    reward = build_reward([9])
    completions = [[{"role": "assistant", "content": "Answer: B"}]]
    assert reward(None, completions, [[1, 2]], ["B"], ["mcq-4-choices"]) == [0.0]
    assert reward(None, completions, [[]], ["B"], ["mcq-4-choices"]) == [0.0]

def test_reward_and_eval_agree_on_every_case():
    row = make_row(0, gold="B")
    filling = make_row(1, gold="H2 + O2 = H2O", kind="filling")
    cases = [(row, "Answer: B", False), (row, "Answer: C", False), (row, "Answer: B", True), (filling, "Answer: H_2 + O_2 → H_2O", False), (filling, "Answer: H2O", False)]
    reward = build_reward([9])
    for case_row, text, truncated in cases:
        rewarded = reward(None, [[{"role": "assistant", "content": text}]], [[1, 2] if truncated else [1, 9]], [case_row["answer"] or case_row["answerKey"]], [case_row["type"]])[0]
        scores, _ = score_responses([case_row], [{"prompt_id": case_row["row_id"], "text": text, "truncated": truncated}])
        assert rewarded == scores["overall"]["accuracy"]

def test_generation_is_split_into_chunks_and_keeps_every_completion(chat_tokenizer, chat_model, solved_rows, tmp_path, monkeypatch):
    generate, calls = fake_generation(chat_tokenizer, 1, 4)
    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", generate)
    trainer = build_grpo_trainer(grpo_config(generation_chunk_size=3), chat_model, chat_tokenizer, solved_rows, tmp_path)
    completions, logprobs, mask = trainer._generate_single_turn([[1]] * 8, None, {}, 4)
    assert calls == [3, 3, 2]
    assert len(completions) == 8
    assert logprobs is None and mask is None

def group_inputs(solved_rows, group_size: int) -> list[dict]:
    dataset = build_grpo_dataset(solved_rows[:2])
    return [dict(dataset[index]) for index in range(2) for _ in range(group_size)]

def test_advantages_are_the_reward_minus_the_group_mean_with_no_scaling(chat_tokenizer, chat_model, solved_rows, tmp_path, monkeypatch):
    generate, _ = fake_generation(chat_tokenizer, 1, 4)
    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", generate)
    trainer = build_grpo_trainer(grpo_config(), chat_model, chat_tokenizer, solved_rows, tmp_path)
    trainer.model.train()
    scored = trainer._generate_and_score_completions(group_inputs(solved_rows, 4))
    assert scored["advantages"].tolist() == pytest.approx([0.75, -0.25, -0.25, -0.25] * 2)

def test_loss_is_normalized_by_batch_size_and_max_completion_length(chat_tokenizer, chat_model, solved_rows, tmp_path, monkeypatch):
    generate, _ = fake_generation(chat_tokenizer, 1, 4)
    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", generate)
    config = grpo_config(loss_batch_size=8, max_completion_length=8)
    trainer = build_grpo_trainer(config, chat_model, chat_tokenizer, solved_rows, tmp_path)
    trainer.model.train()
    scored = trainer._generate_and_score_completions(group_inputs(solved_rows, 4))
    # The training loop sets this, and it's 1 here as the whole batch is one micro-batch
    trainer.current_gradient_accumulation_steps = 1
    loss = trainer._compute_loss(trainer.model, scored)
    # Every importance ratio is 1, so the loss is -sum_i(A_i * tokens_i) / (batch * max_completion_length)
    tokens = scored["completion_mask"].sum(dim=1).float()
    expected = -(scored["advantages"] * tokens).sum() / (8 * 8)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

def test_run_config_end_to_end_moves_the_weights_and_saves_in_fp32(model_dir, solved_rows, tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    generate, _ = fake_generation(tokenizer, 1, 4)
    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", generate)
    path = tmp_path / "tiny_grpo.yaml"
    path.write_text(yaml.safe_dump(grpo_config(model=str(model_dir), max_completion_length=16)))
    summary = run_config(path, solved_rows, tmp_path / "results", tmp_path / "checkpoints", use_wandb=False)

    assert summary["status"] == "finished"
    assert summary["steps_finished"] == 2
    assert summary["log"]["share_of_groups_with_no_reward_variation"] == 0.0
    assert summary["log"]["reward_mean"] == pytest.approx(0.25)
    assert summary["log"]["clipped_token_share"] == 0.0

    from transformers import AutoModelForCausalLM

    before = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).state_dict()
    after = AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoints" / "tiny_grpo" / "final", dtype="auto").state_dict()
    assert {weight.dtype for weight in after.values()} == {torch.float32}
    assert any(not torch.equal(before[name], after[name]) for name in before)

def test_real_sampling_works_end_to_end(model_dir, solved_rows, tmp_path):
    path = tmp_path / "tiny_grpo.yaml"
    path.write_text(yaml.safe_dump(grpo_config(model=str(model_dir), max_steps=1)))
    summary = run_config(path, solved_rows, tmp_path / "results", tmp_path / "checkpoints", use_wandb=False, save=False)
    assert summary["status"] == "finished"
    assert summary["checkpoint_gb"] is None
    # An untrained tiny model never answers, so it hits the cap with a reward of 0
    assert summary["log"]["reward_mean"] == 0.0
    assert summary["log"]["share_of_groups_with_no_reward_variation"] == 1.0
