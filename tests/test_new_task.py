import json

import pytest

from src.eval.new_task import build_chem_prompts, length_stats, measure_accuracy, score_responses
from src.data.chem import build_prompt
from tests.conftest import make_row

def test_scores_overall_and_per_group():
    rows = [make_row(0, task="a"), make_row(1, task="a"), make_row(2, gold="H2 = H2", kind="filling", task="b")]
    responses = [
        {"prompt_id": 0, "text": "Answer: B", "truncated": False},
        {"prompt_id": 1, "text": "Answer: C", "truncated": False},
        {"prompt_id": 2, "text": "Answer: H2 = H2", "truncated": False},
    ]
    scores, scored = score_responses(rows, responses)
    assert scores["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert scores["overall"]["n"] == 3
    assert scores["task:a"]["accuracy"] == 0.5
    assert scores["type:filling"]["accuracy"] == 1.0
    assert [row["correct"] for row in scored] == [True, False, True]

def test_truncated_response_is_wrong():
    scores, _ = score_responses([make_row(0)], [{"prompt_id": 0, "text": "Answer: B", "truncated": True}])
    assert scores["overall"]["accuracy"] == 0.0

@pytest.mark.parametrize("responses", [
    [],
    [{"prompt_id": 9, "text": "", "truncated": False}],
    [{"prompt_id": 0, "text": "", "truncated": False}, {"prompt_id": 0, "text": "", "truncated": False}],
])
def test_missing_or_repeated_responses_fail(responses):
    with pytest.raises(ValueError, match="one response"):
        score_responses([make_row(0)], responses)

def test_prompts_contain_no_answer_key(chat_tokenizer):
    row = make_row(0, gold="H2 = H2", kind="filling")
    prompt = build_chem_prompts(chat_tokenizer, [row])[0]
    assert prompt["prompt_id"] == 0
    assert "H2 = H2" not in chat_tokenizer.decode(prompt["prompt_ids"])
    assert build_prompt(row) in chat_tokenizer.decode(prompt["prompt_ids"])

def test_measure_accuracy_is_reproducible_and_writes_its_files(chat_tokenizer, chat_model, tmp_path):
    rows = [make_row(index) for index in range(5)]
    first = measure_accuracy(chat_model, chat_tokenizer, rows, tmp_path / "a", "base", 7, 2, 4, {"model": "tiny"})
    second = measure_accuracy(chat_model, chat_tokenizer, rows, tmp_path / "b", "base", 7, 2, 4, {"model": "tiny"})
    assert first == second
    assert first["overall"]["n"] == 5
    assert (tmp_path / "a" / "base_new_task_responses.jsonl").read_text() == (tmp_path / "b" / "base_new_task_responses.jsonl").read_text()
    assert json.loads((tmp_path / "a" / "base_new_task.json").read_text()) == first
    assert len((tmp_path / "a" / "base_new_task_scores.jsonl").read_text().splitlines()) == 5

def test_length_stats():
    stats = length_stats([100, 200, 300, 600], [False, False, False, True])
    assert stats["max"] == 600
    assert stats["truncated_share"] == 0.25
    assert stats["share_longer_than"]["256"] == 0.5
    assert stats["share_longer_than"]["1024"] == 0.0
