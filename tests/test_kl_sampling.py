import pytest
from transformers import GenerationConfig

from src.kl.sampling import (
    build_generation_config,
    check_unmodified_sampling,
    extract_response,
    get_stop_ids,
    load_responses,
    sample_responses,
)

def test_sampling_settings_leave_the_logits_unmodified(base_model):
    """The tiny model ships Qwen's settings, so this fails if any of them is inherited."""
    config = build_generation_config(8, get_stop_ids(base_model), 0)
    check_unmodified_sampling(base_model, [1, 2, 3], config)

@pytest.mark.parametrize("settings", [
    {"temperature": 1.0},
    {"temperature": 1.0, "top_p": 1.0},
    {"temperature": 1.0, "top_p": 1.0, "top_k": None, "repetition_penalty": None},
    {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
])
def test_guard_rejects_inherited_settings(base_model, settings):
    config = GenerationConfig(do_sample=True, max_new_tokens=8, eos_token_id=[199], pad_token_id=0, **settings)
    with pytest.raises(ValueError):
        check_unmodified_sampling(base_model, [1, 2, 3], config)

@pytest.mark.parametrize("generated, expected", [
    ([5, 9, 0, 0], ([5, 9], False)),
    ([9, 0, 0, 0], ([9], False)),
    ([5, 9, 6, 9], ([5, 9], False)),
    ([5, 6, 9], ([5, 6, 9], False)),
    ([5, 6, 7], ([5, 6, 7], True)),
])
def test_extract_response_keeps_the_first_stop_token_and_flags_the_cap(generated, expected):
    assert extract_response(generated, [9]) == expected

def test_resumed_sampling_matches_an_uninterrupted_run(base_model, tokenizer, prompts, tmp_path):
    subset = prompts[:12]
    sample_responses(base_model, tokenizer, subset, tmp_path / "full.jsonl", 7, 4, 20, {"model": "base"})
    sample_responses(base_model, tokenizer, subset[:8], tmp_path / "resumed.jsonl", 7, 4, 20, {"model": "base"})
    sample_responses(base_model, tokenizer, subset, tmp_path / "resumed.jsonl", 7, 4, 20, {"model": "base"})
    assert (tmp_path / "full.jsonl").read_text() == (tmp_path / "resumed.jsonl").read_text()

def test_each_batch_gets_its_own_seed(base_model, tokenizer, tmp_path):
    """Two batches with the same prompts must not produce the same samples."""
    prompts = [{"prompt_id": index, "prompt_ids": [1, 2, 3]} for index in range(4)]
    path = tmp_path / "responses.jsonl"
    sample_responses(base_model, tokenizer, prompts, path, 7, 2, 20, {"model": "base"})
    _, records = load_responses(path)
    first_batch = [record["response_ids"] for record in records[:2]]
    second_batch = [record["response_ids"] for record in records[2:]]
    assert first_batch != second_batch

def test_resuming_with_a_different_setup_fails(base_model, tokenizer, prompts, tmp_path):
    path = tmp_path / "responses.jsonl"
    sample_responses(base_model, tokenizer, prompts[:4], path, 7, 4, 20, {"model": "base"})
    with pytest.raises(ValueError):
        sample_responses(base_model, tokenizer, prompts[:4], path, 8, 4, 20, {"model": "base"})

def test_saved_responses_have_the_expected_fields(base_path, prompts):
    header, records = load_responses(base_path)
    assert header["settings"]["top_k"] == 0
    assert header["dtype"] == "torch.float32"
    assert len(records) == len(prompts)
    assert [record["prompt_ids"] for record in records] == [prompt["prompt_ids"] for prompt in prompts]
    assert all(len(record["response_ids"]) >= 1 for record in records)
