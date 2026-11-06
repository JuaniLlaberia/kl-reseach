import json
from collections import Counter
from pathlib import Path

import pytest

from src.data.chem import (
    EXPECTED_COUNTS,
    N_HELDOUT,
    N_KL,
    allocate,
    build_prompt,
    check_counts,
    filter_chem_rows,
    get_gold,
    load_chem_rows,
    load_split,
    save_split,
    select_rows,
    split_rows,
    stratified_sample,
)
from src.data.solutions import MAX_INPUT_TOKENS
from src.eval.extract import FILLING, MCQ, is_correct, normalize_equation

def make_raw_row(index: int, subtask: str, row_type: str = MCQ, question: str | None = None, domain: str = "Chemistry", level: str = "L3") -> dict:
    is_mcq = row_type == MCQ
    return {
        "prompt": {"default": "Please directly give the answer without any explanation."},
        "question": question or f"Question {index} of {subtask}?",
        "answer": "" if is_mcq else f"2H{index}(g) = H{index}2(g)",
        "type": row_type,
        "domain": domain,
        "details": {"level": level, "task": f"task_{subtask}", "subtask": subtask, "source": "test"},
        "answerKey": "BCDA"[index % 4] if is_mcq else "",
        "choices": {"text": [f"choice {index} {label}" for label in "ABCD"], "label": list("ABCD")} if is_mcq else {"text": [], "label": []},
    }

@pytest.fixture
def rows():
    """A small dataset with subtasks of different sizes, questions that repeat with different choices, and rows of other domains."""
    raw = []
    for subtask, size in (("large", 240), ("medium", 100), ("small", 40), ("tiny", 7)):
        raw += [make_raw_row(len(raw) + i, subtask) for i in range(size)]
    raw += [make_raw_row(len(raw) + i, "equations", FILLING) for i in range(50)]
    raw += [make_raw_row(len(raw) + i, "large", question=f"Repeated question {i // 2}?") for i in range(10)]
    raw += [make_raw_row(len(raw), "large", domain="Biology"), make_raw_row(len(raw) + 1, "large", level="L2")]
    return filter_chem_rows(raw)

def ids(split, side):
    return [item["row_id"] for item in split[side]]

def test_filter_keeps_only_chemistry_l3_and_file_positions():
    raw = [make_raw_row(0, "a"), make_raw_row(1, "a", domain="Physics"), make_raw_row(2, "a", level="L1"), make_raw_row(3, "a")]
    assert [row["row_id"] for row in filter_chem_rows(raw)] == [0, 3]

def test_check_counts(rows):
    check_counts(rows, dict(Counter(row["type"] for row in rows)))
    with pytest.raises(ValueError):
        check_counts(rows, EXPECTED_COUNTS)

def test_split_is_reproducible_and_depends_on_the_seed(rows):
    first = split_rows(rows, seed=3, n_heldout=100, n_kl=80)
    assert first == split_rows(rows, seed=3, n_heldout=100, n_kl=80)
    assert first != split_rows(rows, seed=4, n_heldout=100, n_kl=80)

def test_split_is_the_same_for_any_row_order(rows):
    assert split_rows(rows, seed=3, n_heldout=100, n_kl=80) == split_rows(rows[::-1], seed=3, n_heldout=100, n_kl=80)

def test_split_is_disjoint_and_complete(rows):
    split = split_rows(rows, n_heldout=100, n_kl=80)
    train, heldout = set(ids(split, "train")), set(ids(split, "heldout"))
    assert not train & heldout
    assert train | heldout == {row["row_id"] for row in rows}
    assert len(heldout) == 100

def test_kl_prompts_are_inside_heldout(rows):
    split = split_rows(rows, n_heldout=100, n_kl=80)
    assert len(ids(split, "kl")) == 80
    assert set(ids(split, "kl")) <= set(ids(split, "heldout"))

def test_repeated_questions_stay_on_one_side(rows):
    for seed in range(20):
        split = split_rows(rows, seed=seed, n_heldout=100, n_kl=80)
        heldout = set(ids(split, "heldout"))
        sides = {}
        for row in rows:
            sides.setdefault(row["question_hash"], set()).add(row["row_id"] in heldout)
        assert all(len(side) == 1 for side in sides.values())

def test_split_preserves_subtask_shares(rows):
    split = split_rows(rows, n_heldout=100, n_kl=80)
    heldout = select_rows(rows, split, "heldout")
    counts = Counter(row["details"]["subtask"] for row in heldout)
    totals = Counter(row["details"]["subtask"] for row in rows)
    for subtask, total in totals.items():
        assert abs(counts[subtask] - total * 100 / len(rows)) <= 1

def test_allocate_sums_to_the_total_and_breaks_ties_by_name():
    assert sum(allocate({"a": 340, "b": 160, "c": 7}, 123).values()) == 123
    assert allocate({"b": 1, "a": 1}, 1) == {"a": 1, "b": 0}

def test_stratified_sample_is_reproducible(rows):
    first = stratified_sample(rows, 50, 1)
    assert first == stratified_sample(rows, 50, 1)
    assert len(first) == 50

def test_select_rows_fails_if_a_row_changed(rows):
    split = split_rows(rows, n_heldout=100, n_kl=80)
    assert [row["row_id"] for row in select_rows(rows, split, "kl")] == ids(split, "kl")
    changed = [{**row, "question_hash": "different"} for row in rows]
    with pytest.raises(ValueError):
        select_rows(changed, split, "kl")

def test_saved_split_is_byte_stable(rows, tmp_path):
    split = split_rows(rows, n_heldout=100, n_kl=80)
    save_split(split, tmp_path / "a.json")
    save_split(split_rows(rows, n_heldout=100, n_kl=80), tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert json.loads((tmp_path / "a.json").read_text())["heldout"] == split["heldout"]

def test_mcq_prompt_has_choices_and_no_answer(rows):
    row = next(row for row in rows if row["type"] == MCQ)
    prompt = build_prompt(row)
    assert row["question"] in prompt
    for label, text in zip(row["choices"]["label"], row["choices"]["text"]):
        assert f"{label}. {text}" in prompt
    assert prompt.endswith("Answer: <letter>")
    assert "Reason briefly in plain text, then finish with a final line in exactly this form:" in prompt
    assert "without any explanation" not in prompt

def test_prompts_never_contain_the_answer_key_field(rows):
    for row in rows:
        marked = {**row, "answerKey": "SECRET-KEY" if row["type"] == MCQ else "", "answer": "SECRET-KEY" if row["type"] == FILLING else ""}
        assert "SECRET-KEY" not in build_prompt(marked)

def test_filling_prompt_has_no_gold_equation(rows):
    row = next(row for row in rows if row["type"] == FILLING)
    prompt = build_prompt(row)
    assert get_gold(row) not in prompt
    assert prompt.endswith("Answer: <balanced equation>")

def test_get_gold(rows):
    mcq = next(row for row in rows if row["type"] == MCQ)
    filling = next(row for row in rows if row["type"] == FILLING)
    assert get_gold(mcq) == mcq["answerKey"]
    assert get_gold(filling) == filling["answer"]

@pytest.mark.network
def test_real_dataset_matches_the_log():
    real_rows = load_chem_rows()
    check_counts(real_rows)
    split = split_rows(real_rows)
    assert (len(split["train"]), len(split["heldout"]), len(split["kl"])) == (2400 - N_HELDOUT, N_HELDOUT, N_KL)
    assert split == load_split(Path(__file__).parents[1] / "data" / "splits" / "chem_l3.json")
    assert all(is_correct(f"Answer: {get_gold(row)}", get_gold(row), row["type"]) for row in real_rows)
    # Normalizing must never make two different dataset answers look the same
    golds = {row["answer"] for row in real_rows if row["type"] == FILLING}
    assert len({normalize_equation(gold) for gold in golds}) == len(golds)
    # Each token holds at least one character, so this keeps the budget reservation valid
    assert max(len(build_prompt(row)) for row in real_rows) < MAX_INPUT_TOKENS
