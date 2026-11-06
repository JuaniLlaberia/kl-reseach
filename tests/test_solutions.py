import json
import threading

import pytest
import requests

from src.data.chem import build_prompt, filter_chem_rows, split_rows
from src.data.solutions import (
    MAX_BACKOFF_SECONDS,
    MAX_RETRIES,
    MAX_TRIES,
    MODEL,
    PROVIDER_NAME,
    ApiError,
    SetupMismatch,
    attempt_seed,
    best_attempts,
    build_payload,
    call_cost,
    expected_tries,
    generate_solutions,
    load_attempts,
    max_call_cost,
    project_full_run,
    prompt_hash,
    rescore_attempts,
    retry_wait,
    save_solutions,
    summarize_attempts,
)
from src.eval.extract import MCQ

KEY = "test-key"
COST = 0.003
MAX_TOKENS = 1024

def make_row(index: int, task: str = "molar") -> dict:
    return {
        "prompt": {"default": "x"},
        "question": f"Question {index}?",
        "answer": "",
        "type": MCQ,
        "domain": "Chemistry",
        "details": {"level": "L3", "task": task, "subtask": task, "source": "test"},
        "answerKey": "B",
        "choices": {"text": [f"c{index}{label}" for label in "ABCD"], "label": list("ABCD")},
    }

@pytest.fixture
def rows():
    return filter_chem_rows([make_row(index) for index in range(30)])

@pytest.fixture
def split(rows):
    return split_rows(rows, n_heldout=10, n_kl=5)

@pytest.fixture
def train(rows, split):
    train_ids = {item["row_id"] for item in split["train"]}
    return [row for row in rows if row["row_id"] in train_ids]

def response(text="Reasoning\nAnswer: B", finish_reason="stop", cost=COST, model=MODEL, provider=PROVIDER_NAME):
    return 200, {
        "model": model,
        "provider": provider,
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "cost": cost},
    }

class FakeApi:
    """Stands in for OpenRouter. fail_tries is the number of wrong answers a row gets before a right one."""
    def __init__(self, fail_tries=lambda row_seed: 0):
        self.fail_tries = fail_tries
        self.payloads = []
        self.lock = threading.Lock()

    def __call__(self, api_key, payload):
        assert api_key == KEY
        with self.lock:
            self.payloads.append(payload)
            prompt = payload["messages"][0]["content"]
            tries = sum(p["messages"][0]["content"] == prompt for p in self.payloads)
        return response("Answer: A" if tries <= self.fail_tries(prompt) else "Answer: B")

def run(train, split, path, post, budget=100.0, **kwargs):
    return generate_solutions(train, split, path, KEY, budget, MAX_TOKENS, workers=kwargs.pop("workers", 4), post=post, sleep=lambda _: None, **kwargs)

def test_payload_pins_the_setup_and_has_no_system_prompt():
    payload = build_payload("Hello", 7, 512)
    assert payload["model"] == MODEL == "openai/gpt-4o-2024-08-06"
    assert payload["temperature"] == 1.0
    assert payload["seed"] == 7
    assert payload["max_tokens"] == 512
    assert [message["role"] for message in payload["messages"]] == ["user"]
    assert payload["provider"] == {"order": ["openai"], "allow_fallbacks": False, "require_parameters": True}

def test_stops_at_the_first_correct_try(train, split, tmp_path):
    api = FakeApi()
    result = run(train, split, tmp_path / "log.jsonl", api)
    assert len(api.payloads) == len(train)
    assert result["solved"] == len(train)
    assert result["stop_reason"] == "finished"

def test_keeps_the_first_correct_of_several_tries(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train, split, log, FakeApi(lambda prompt: 2))
    attempts = load_attempts(log)
    assert len(attempts) == 3 * len(train)
    assert {record["attempt"] for record in best_attempts(attempts).values()} == {3}

def test_never_passes_sixteen_tries(train, split, tmp_path):
    api = FakeApi(lambda prompt: 1000)
    result = run(train, split, tmp_path / "log.jsonl", api)
    assert len(api.payloads) == MAX_TRIES * len(train) == 16 * len(train)
    assert result["solved"] == 0

def test_rounds_are_breadth_first(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train, split, log, FakeApi(lambda prompt: 3), workers=1)
    attempts = [record["attempt"] for record in load_attempts(log)]
    assert attempts == sorted(attempts)

def test_budget_is_a_hard_stop(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    # The real cost per call is far below the reservation, so calls in flight can't take it over the limit
    budget = max_call_cost(MAX_TOKENS) * 2
    assert len(train) * COST > budget
    result = run(train, split, log, FakeApi(), budget=budget)
    assert result["stop_reason"] == "budget"
    assert 0 < result["spent_usd"] <= budget
    assert sum(record["cost_usd"] for record in load_attempts(log)) <= budget

def test_worst_case_costs_can_never_pass_the_budget(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    worst = max_call_cost(MAX_TOKENS)
    budget = worst * 7.5
    run(train, split, log, lambda key, payload: response(cost=worst), budget=budget, workers=8)
    assert sum(record["cost_usd"] for record in load_attempts(log)) <= budget

def test_prior_spend_counts_towards_the_budget(train, split, tmp_path):
    api = FakeApi()
    result = run(train, split, tmp_path / "log.jsonl", api, budget=15.0, prior_spend_usd=15.0)
    assert api.payloads == []
    assert result["stop_reason"] == "budget"

def test_spend_in_the_log_counts_after_a_restart(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    budget = max_call_cost(MAX_TOKENS) + COST * 3
    run(train[:3], split, log, FakeApi())
    api = FakeApi()
    run(train, split, log, api, budget=budget)
    assert len(api.payloads) == 1

def test_resume_repeats_no_paid_call(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    calls = []

    def crashes_midway(key, payload):
        calls.append(payload)
        if len(calls) > 5:
            raise ApiError("Crash")
        return response("Answer: A")

    with pytest.raises(ApiError):
        run(train, split, log, crashes_midway, workers=1)
    done = {(record["row_id"], record["attempt"]) for record in load_attempts(log)}
    assert len(done) == 5

    api = FakeApi()
    run(train, split, log, api)
    final = load_attempts(log)
    keys = [(record["row_id"], record["attempt"]) for record in final]
    assert len(keys) == len(set(keys))
    assert done <= set(keys)
    assert best_attempts(final).keys() == {row["row_id"] for row in train}

def test_resume_skips_solved_rows(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train, split, log, FakeApi())
    api = FakeApi()
    run(train, split, log, api)
    assert api.payloads == []

def test_max_tries_one_only_makes_first_tries_and_can_continue(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train[:5], split, log, FakeApi(lambda prompt: 1000), max_tries=1)
    assert [record["attempt"] for record in load_attempts(log)] == [1] * 5
    api = FakeApi()
    run(train[:5], split, log, api, max_tries=2)
    assert {record["attempt"] for record in load_attempts(log)} == {1, 2}
    assert len(api.payloads) == 5

def test_a_line_cut_by_a_crash_is_removed(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train[:2], split, log, FakeApi())
    with open(log, "a", encoding="utf-8") as file:
        file.write('{"row_id": 1, "attem')
    assert len(load_attempts(log)) == 2
    assert all(json.loads(line) for line in log.read_text().splitlines())

def test_refuses_heldout_rows(rows, split, tmp_path):
    heldout_id = split["heldout"][0]["row_id"]
    heldout_row = next(row for row in rows if row["row_id"] == heldout_id)
    api = FakeApi()
    with pytest.raises(ValueError):
        run([heldout_row], split, tmp_path / "log.jsonl", api)
    assert api.payloads == []

def test_no_answer_in_any_payload(rows, split, tmp_path):
    marked = [{**row, "answerKey": "B", "choices": row["choices"]} for row in rows]
    train_ids = {item["row_id"] for item in split["train"]}
    api = FakeApi()
    run([row for row in marked if row["row_id"] in train_ids], split, tmp_path / "log.jsonl", api)
    for payload in api.payloads:
        assert "answerKey" not in json.dumps(payload)
        assert set(payload) == {"model", "messages", "temperature", "max_tokens", "seed", "provider", "usage"}

def test_key_is_never_logged(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train[:2], split, log, FakeApi())
    assert KEY not in log.read_text()

@pytest.mark.parametrize("status", [429, 500, 502, 503, 408])
def test_retries_temporary_failures(train, split, tmp_path, status):
    outcomes = iter([(status, {}), (status, {"error": {"code": status, "message": "x"}}), response()])
    result = run(train[:1], split, tmp_path / "log.jsonl", lambda key, payload: next(outcomes))
    assert result["solved"] == 1

def test_retries_an_error_inside_a_200_body(train, split, tmp_path):
    outcomes = iter([(200, {"error": {"code": 502, "message": "Provider error"}}), (200, {"choices": []}), response()])
    result = run(train[:1], split, tmp_path / "log.jsonl", lambda key, payload: next(outcomes))
    assert result["solved"] == 1

def test_retries_network_errors(train, split, tmp_path):
    calls = []

    def flaky(key, payload):
        calls.append(payload)
        if len(calls) == 1:
            raise requests.ConnectionError("Reset")
        return response()

    assert run(train[:1], split, tmp_path / "log.jsonl", flaky)["solved"] == 1

def test_gives_up_after_too_many_retries(train, split, tmp_path):
    calls = []

    def always_down(key, payload):
        calls.append(payload)
        return 503, {}

    with pytest.raises(ApiError):
        run(train[:1], split, tmp_path / "log.jsonl", always_down)
    assert len(calls) == MAX_RETRIES + 1

def test_waits_are_capped_and_cover_minutes_of_rate_limiting(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda low, high: high)
    longest = sum(retry_wait(retry) for retry in range(1, MAX_RETRIES + 1))
    assert max(retry_wait(retry) for retry in range(1, 50)) == MAX_BACKOFF_SECONDS
    monkeypatch.setattr("random.uniform", lambda low, high: low)
    shortest = sum(retry_wait(retry) for retry in range(1, MAX_RETRIES + 1))
    assert shortest >= 5 * 60
    assert longest <= 11 * 60

def test_survives_a_long_streak_of_rate_limits(train, split, tmp_path):
    outcomes = iter([(429, {"error": {"code": 429, "message": "Rate limited"}})] * MAX_RETRIES + [response()])
    assert run(train[:1], split, tmp_path / "log.jsonl", lambda key, payload: next(outcomes))["solved"] == 1

def test_rate_limit_error_keeps_the_explanation_and_saves_earlier_attempts(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    calls = []
    body = {"error": {"code": 429, "message": "Provider returned error", "metadata": {"provider_name": "OpenAI", "raw": "Rate limit reached for gpt-4o"}}}

    def limited_after_two(key, payload):
        calls.append(payload)
        return response() if len(calls) <= 2 else (429, body)

    with pytest.raises(ApiError, match="Provider returned error | OpenAI | Rate limit reached") as error:
        run(train[:3], split, log, limited_after_two, workers=1)
    assert KEY not in str(error.value)
    assert len(load_attempts(log)) == 2

    api = FakeApi()
    assert run(train[:3], split, log, api)["solved"] == 3
    assert len(api.payloads) == 1

@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_permanent_errors_stop_right_away(train, split, tmp_path, status):
    calls = []

    def refuses(key, payload):
        calls.append(payload)
        return status, {"error": {"code": status, "message": "No"}}

    with pytest.raises(ApiError):
        run(train[:1], split, tmp_path / "log.jsonl", refuses)
    assert len(calls) == 1

@pytest.mark.parametrize("kwargs", [{"model": "openai/gpt-4o"}, {"provider": "Azure"}])
def test_mismatch_stops_the_run_but_logs_the_paid_call(train, split, tmp_path, kwargs):
    log = tmp_path / "log.jsonl"
    with pytest.raises(SetupMismatch):
        run(train, split, log, lambda key, payload: response(**kwargs), workers=1)
    assert len(load_attempts(log)) == 1

def test_changed_prompt_can_not_mix_with_old_attempts(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train[:2], split, log, FakeApi())
    changed = [{**row, "question": row["question"] + " Changed.", "question_hash": row["question_hash"]} for row in train[:2]]
    with pytest.raises(SetupMismatch):
        run(changed, split, log, FakeApi())

def test_truncated_response_is_rejected_even_with_a_right_answer(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    outcomes = iter([response("Answer: B", finish_reason="length"), response("Answer: B")])
    run(train[:1], split, log, lambda key, payload: next(outcomes))
    assert [record["correct"] for record in load_attempts(log)] == [False, True]

def write_stale_log(log, row, cost=COST):
    """A paid attempt that an older version of the matcher scored as wrong."""
    record = {
        "row_id": row["row_id"], "attempt": 1, "seed": 1, "prompt_hash": prompt_hash(build_prompt(row)), "text": "Reasoning\nAnswer: B",
        "finish_reason": "stop", "correct": False, "prompt_tokens": 100, "completion_tokens": 200, "cost_usd": cost,
        "model": MODEL, "provider": PROVIDER_NAME, "temperature": 1.0, "max_tokens": MAX_TOKENS,
    }
    log.write_text(json.dumps(record) + "\n", encoding="utf-8")

def test_a_matcher_fix_applies_to_paid_attempts_without_paying_again(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    write_stale_log(log, train[0])
    original = log.read_text()
    api = FakeApi()
    result = run(train[:1], split, log, api)
    assert api.payloads == []
    assert result["solved"] == 1
    assert log.read_text() == original

def test_every_report_uses_the_current_matcher(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    write_stale_log(log, train[0])
    attempts = load_attempts(log)
    assert attempts[0]["correct"] is False
    assert rescore_attempts(train[:1], attempts)[0]["correct"] is True
    assert summarize_attempts(train[:1], attempts)["overall"]["first_try_pass_rate"] == 1.0
    assert save_solutions(train[:1], attempts, tmp_path / "solutions.jsonl") == 1
    assert project_full_run(train[:1], train[:1], attempts, 15.0)["pass_rates_by_task"] == {"molar": 1.0}

def test_rescoring_keeps_truncated_attempts_wrong_and_other_rows_untouched(train):
    truncated = {"row_id": train[0]["row_id"], "attempt": 1, "text": "Answer: B", "finish_reason": "length", "correct": True}
    other = {"row_id": -1, "attempt": 1, "text": "Answer: B", "finish_reason": "stop", "correct": False}
    rescored = rescore_attempts(train[:1], [truncated, other])
    assert rescored[0]["correct"] is False
    assert rescored[1] == other

def test_seeds_are_unique_per_attempt():
    seeds = {attempt_seed(0, row_id, attempt) for row_id in range(50) for attempt in range(1, MAX_TRIES + 1)}
    assert len(seeds) == 50 * MAX_TRIES

def test_cost_falls_back_to_the_listed_price():
    assert call_cost({"cost": 0.5}) == 0.5
    assert call_cost({"prompt_tokens": 1000, "completion_tokens": 1000}) == pytest.approx(0.0125)

def test_saved_solutions_hold_one_per_solved_row(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    run(train[:4], split, log, FakeApi(lambda prompt: 1))
    out = tmp_path / "solutions.jsonl"
    assert save_solutions(train[:4], load_attempts(log), out) == 4
    saved = [json.loads(line) for line in out.read_text().splitlines()]
    assert [record["row_id"] for record in saved] == sorted(row["row_id"] for row in train[:4])
    assert {record["attempt"] for record in saved} == {2}
    assert all(record["solution"] == "Answer: B" for record in saved)

def test_summary_reports_first_try_rate_separately_from_solved(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    selected = train[:4]
    run(selected, split, log, FakeApi(lambda prompt: 1 if prompt == build_prompt(selected[0]) else 0))
    summary = summarize_attempts(selected, load_attempts(log))
    assert summary["calls"] == 5
    assert summary["cost_usd"] == pytest.approx(5 * COST)
    assert summary["overall"]["first_try_pass_rate"] == 0.75
    assert summary["overall"]["solved"] == 4
    assert summary["models"] == {MODEL: 5}
    assert summary["providers"] == {PROVIDER_NAME: 5}
    assert summary["no_valid_answer"] == 0

def test_summary_counts_responses_without_a_valid_answer(train, split, tmp_path):
    log = tmp_path / "log.jsonl"
    outcomes = iter([response("The answer is B."), response("Answer: B")])
    run(train[:1], split, log, lambda key, payload: next(outcomes))
    assert summarize_attempts(train[:1], load_attempts(log))["no_valid_answer"] == 1

def attempt_record(row_id, correct):
    return {"row_id": row_id, "attempt": 1, "text": "Answer: B" if correct else "Answer: A", "finish_reason": "stop", "correct": correct, "cost_usd": 0.01}

def test_projection_uses_the_truncated_geometric_mean():
    train = filter_chem_rows([make_row(index) for index in range(100)])
    test_rows = train[:4]
    attempts = [attempt_record(row["row_id"], index < 2) for index, row in enumerate(test_rows)]
    projection = project_full_run(train, test_rows, attempts, budget_usd=15.0)
    tries = (1 - 0.5 ** 16) / 0.5
    assert expected_tries(0.5, 16) == pytest.approx(tries)
    assert projection["pass_rates_by_task"] == {"molar": 0.5}
    assert projection["expected_cost_high_usd"] == pytest.approx(100 * tries * 0.01)
    assert projection["all_tries_cost_usd"] == pytest.approx(100 * 16 * 0.01)
    assert projection["calls_within_budget"] == pytest.approx(1500)

def test_projection_range_holds_a_task_that_scored_zero_in_a_small_sample():
    train = filter_chem_rows([make_row(index, "easy" if index < 50 else "unlucky") for index in range(100)])
    easy = train[:3]
    unlucky = train[50:53]
    attempts = [attempt_record(row["row_id"], True) for row in easy] + [attempt_record(row["row_id"], False) for row in unlucky]
    projection = project_full_run(train, easy + unlucky, attempts, budget_usd=15.0)
    # Pooled: every row at 50%. Per task: half the rows use all 16 tries, half use one
    assert projection["expected_cost_low_usd"] == pytest.approx(100 * expected_tries(0.5, 16) * 0.01)
    assert projection["expected_cost_high_usd"] == pytest.approx((50 * 1 + 50 * 16) * 0.01)
    assert projection["expected_cost_low_usd"] < projection["expected_cost_high_usd"]

def test_projection_counts_an_unseen_task_as_all_tries():
    train = filter_chem_rows([make_row(0, "seen"), make_row(1, "unseen")])
    projection = project_full_run(train, train[:1], [attempt_record(train[0]["row_id"], True)], budget_usd=0.01)
    assert projection["pass_rates_by_task"]["unseen"] == 0.0
    assert projection["expected_cost_high_usd"] == pytest.approx((1 + 16) * 0.01)
