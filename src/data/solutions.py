import hashlib
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import requests

from src.data.chem import build_prompt, get_gold
from src.eval.extract import extract_answer, is_correct

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-2024-08-06"
PROVIDER_TAG = "openai"
PROVIDER_NAME = "OpenAI"
TEMPERATURE = 1.0
MAX_TRIES = 16

# Listed OpenRouter price, used to reserve budget and if a response doesn't report its cost
USD_PER_INPUT_TOKEN = 2.5e-6
USD_PER_OUTPUT_TOKEN = 10e-6
# More than any prompt in Chemistry L3 could use, so a reservation always covers the real cost
MAX_INPUT_TOKENS = 4000

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 6

class BudgetReached(Exception):
    """The next call could pass the spending limit."""

class SetupMismatch(Exception):
    """OpenRouter served a different model or provider, or the saved attempts used a different setup."""

class ApiError(Exception):
    """OpenRouter kept failing, or returned an error that retrying can't fix."""

def build_payload(prompt: str, seed: int, max_tokens: int) -> dict:
    """
    Builds one request. There's no system prompt, and fallbacks are off because Azure also serves this snapshot.

    Args:
        prompt (str): User message.
        seed (int): Seed of this attempt.
        max_tokens (int): Maximum number of tokens per response.
    Returns:
        The JSON body of the request.
    """
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "seed": seed,
        "provider": {"order": [PROVIDER_TAG], "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True},
    }

def default_post(api_key: str, payload: dict) -> tuple[int, dict]:
    """
    Sends one request to OpenRouter.

    Args:
        api_key (str): OpenRouter key. It's only sent in this header, and never logged.
        payload (dict): JSON body of the request.
    Returns:
        The status code and the JSON body of the response.
    """
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=180)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {}

def call_with_retries(post: Callable, api_key: str, payload: dict, sleep: Callable = time.sleep) -> dict:
    """
    Sends one request, retrying with exponential backoff on temporary failures.

    Args:
        post (Callable): Function that sends the request.
        api_key (str): OpenRouter key.
        payload (dict): JSON body of the request.
        sleep (Callable): Function used to wait between retries.
    Returns:
        The JSON body of a successful response.
    """
    failure = None
    for retry in range(MAX_RETRIES + 1):
        if retry > 0:
            sleep(2 ** retry)
        try:
            status, body = post(api_key, payload)
        except requests.RequestException as error:
            # The error's text can include request details, so only its type is kept
            failure = type(error).__name__
            continue

        # OpenRouter can report an error in a body with status 200
        error_code = (body.get("error") or {}).get("code")
        if status == 200 and error_code is None and body.get("choices"):
            return body
        if status in RETRYABLE_STATUS or error_code in RETRYABLE_STATUS or (status == 200 and error_code is None):
            failure = f"status {status}, error {error_code}"
            continue
        raise ApiError(f"OpenRouter returned status {status}: {(body.get('error') or {}).get('message')}")
    raise ApiError(f"OpenRouter kept failing after {MAX_RETRIES} retries: {failure}")

def call_cost(usage: dict) -> float:
    """
    Gets the cost of one call.

    Args:
        usage (dict): Usage field of the response.
    Returns:
        The cost OpenRouter reports, or the cost at the listed price if it reports none.
    """
    cost = usage.get("cost")
    if cost is not None:
        return float(cost)
    return usage.get("prompt_tokens", 0) * USD_PER_INPUT_TOKEN + usage.get("completion_tokens", 0) * USD_PER_OUTPUT_TOKEN

def max_call_cost(max_tokens: int) -> float:
    """
    Gets the most that one call can cost, which is what gets reserved before sending it.

    Args:
        max_tokens (int): Maximum number of tokens per response.
    Returns:
        The maximum cost of one call.
    """
    return MAX_INPUT_TOKENS * USD_PER_INPUT_TOKEN + max_tokens * USD_PER_OUTPUT_TOKEN

def prompt_hash(prompt: str) -> str:
    """
    Hashes a prompt, so attempts made with a different wording can't be mixed.

    Args:
        prompt (str): User message.
    Returns:
        The hash of the prompt.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

def attempt_seed(base_seed: int, row_id: int, attempt: int) -> int:
    """
    Gets the seed of one attempt, which doesn't depend on the order in which attempts run.

    Args:
        base_seed (int): Seed of the whole run.
        row_id (int): Row of the attempt.
        attempt (int): Attempt number, starting at 1.
    Returns:
        The seed of this attempt.
    """
    return base_seed + row_id * (MAX_TRIES + 1) + attempt

def load_attempts(log_path: Path) -> list[dict]:
    """
    Loads the log of attempts. A last line cut by a crash is removed from the file, so appending stays valid.

    Args:
        log_path (Path): Path of the JSONL log.
    Returns:
        One record per attempt.
    """
    if not log_path.exists():
        return []

    attempts = []
    valid_bytes = 0
    with open(log_path, "rb") as file:
        for raw_line in file:
            try:
                attempts.append(json.loads(raw_line))
            except ValueError:
                break
            if not raw_line.endswith(b"\n"):
                attempts.pop()
                break
            valid_bytes += len(raw_line)

    if valid_bytes != log_path.stat().st_size:
        with open(log_path, "r+b") as file:
            file.truncate(valid_bytes)
    return attempts

def check_train_only(rows: list[dict], split: dict) -> None:
    """
    Fails unless every row is on the train side of the split, so no held-out row ever reaches GPT-4o.

    Args:
        rows (list[dict]): Rows to send.
        split (dict): The split.
    Returns:
        None.
    """
    train_ids = {item["row_id"] for item in split["train"]}
    outside = [row["row_id"] for row in rows if row["row_id"] not in train_ids]
    if outside:
        raise ValueError(f"{len(outside)} rows aren't in the train split, for example {outside[:5]}.")

def generate_solutions(
    rows: list[dict],
    split: dict,
    log_path: Path,
    api_key: str,
    budget_usd: float,
    max_tokens: int,
    max_tries: int = MAX_TRIES,
    seed: int = 0,
    workers: int = 8,
    prior_spend_usd: float = 0.0,
    post: Callable = default_post,
    sleep: Callable = time.sleep,
) -> dict:
    """
    Runs rejection sampling in breadth-first rounds: every unsolved row gets its first try, then its second, and so on.
    Every attempt is appended to the log right away, so a rerun continues without repeating any paid call.

    Args:
        rows (list[dict]): Train rows to solve.
        split (dict): The split, used to refuse any row outside train.
        log_path (Path): Path of the JSONL log of attempts.
        api_key (str): OpenRouter key.
        budget_usd (float): Spending limit across every call in this log and prior_spend_usd.
        max_tokens (int): Maximum number of tokens per response.
        max_tries (int): Maximum number of tries per row.
        seed (int): Seed of the run.
        workers (int): Number of calls in flight.
        prior_spend_usd (float): Money spent on earlier logs that count towards the limit.
        post (Callable): Function that sends a request, replaced in tests.
        sleep (Callable): Function used to wait between retries, replaced in tests.
    Returns:
        Why the run ended, the money spent and the number of rows solved.
    """
    check_train_only(rows, split)
    if max_tries > MAX_TRIES:
        raise ValueError(f"max_tries can't be above {MAX_TRIES}.")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = load_attempts(log_path)
    prompts = {row["row_id"]: build_prompt(row) for row in rows}

    tries = Counter()
    solved = set()
    for record in attempts:
        row_id = record["row_id"]
        if row_id in prompts and record["prompt_hash"] != prompt_hash(prompts[row_id]):
            raise SetupMismatch(f"Row {row_id} was attempted with another prompt. Archive {log_path} and pass its cost as prior_spend_usd.")
        tries[row_id] = max(tries[row_id], record["attempt"])
        if record["correct"]:
            solved.add(row_id)

    lock = threading.Lock()
    stop = threading.Event()
    state = {"spent": prior_spend_usd + sum(record["cost_usd"] for record in attempts), "reserved": 0.0, "stop_reason": None}
    reservation = max_call_cost(max_tokens)

    def run_attempt(row: dict, attempt: int) -> None:
        row_id = row["row_id"]
        with lock:
            if stop.is_set():
                return
            # Reserving first means that calls in flight can never take the total past the limit
            if state["spent"] + state["reserved"] + reservation > budget_usd:
                state["stop_reason"] = "budget"
                stop.set()
                return
            state["reserved"] += reservation

        try:
            payload = build_payload(prompts[row_id], attempt_seed(seed, row_id, attempt), max_tokens)
            body = call_with_retries(post, api_key, payload, sleep)
        except ApiError:
            with lock:
                state["reserved"] -= reservation
                stop.set()
            raise

        choice = body["choices"][0]
        usage = body.get("usage") or {}
        text = (choice.get("message") or {}).get("content") or ""
        finish_reason = choice.get("finish_reason")
        truncated = finish_reason != "stop"
        record = {
            "row_id": row_id,
            "attempt": attempt,
            "seed": payload["seed"],
            "prompt_hash": prompt_hash(prompts[row_id]),
            "text": text,
            "finish_reason": finish_reason,
            "correct": is_correct(text, get_gold(row), row["type"], truncated),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": call_cost(usage),
            "model": body.get("model"),
            "provider": body.get("provider"),
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
        }
        with lock:
            state["reserved"] -= reservation
            state["spent"] += record["cost_usd"]
            with open(log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
                file.flush()
                os.fsync(file.fileno())
            tries[row_id] = attempt
            if record["correct"]:
                solved.add(row_id)
            # It was paid for and saved first, so the mismatch only blocks the calls that follow
            if record["model"] != MODEL or record["provider"] != PROVIDER_NAME:
                stop.set()
                raise SetupMismatch(f"Asked for {MODEL} on {PROVIDER_NAME}, got {record['model']} on {record['provider']}.")

    for attempt in range(1, max_tries + 1):
        pending = [row for row in rows if row["row_id"] not in solved and tries[row["row_id"]] < attempt]
        if not pending or stop.is_set():
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_attempt, row, attempt) for row in pending]
        for future in futures:
            future.result()

    return {
        "stop_reason": state["stop_reason"] or "finished",
        "spent_usd": state["spent"],
        "solved": sum(row["row_id"] in solved for row in rows),
        "rows": len(rows),
    }

def best_attempts(attempts: list[dict]) -> dict[int, dict]:
    """
    Gets the solution kept for each row, which is its first correct attempt.

    Args:
        attempts (list[dict]): Records from the log.
    Returns:
        Dictionary from row_id to the kept attempt.
    """
    kept = {}
    for record in sorted(attempts, key=lambda record: (record["row_id"], record["attempt"])):
        if record["correct"] and record["row_id"] not in kept:
            kept[record["row_id"]] = record
    return kept

def save_solutions(rows: list[dict], attempts: list[dict], out_path: Path) -> int:
    """
    Writes the one solution kept per solved row, with its attempt number.

    Args:
        rows (list[dict]): Train rows.
        attempts (list[dict]): Records from the log.
        out_path (Path): Path of the JSONL file to write.
    Returns:
        The number of solutions written.
    """
    kept = best_attempts(attempts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        for row in sorted(rows, key=lambda row: row["row_id"]):
            if row["row_id"] in kept:
                record = kept[row["row_id"]]
                file.write(json.dumps({
                    "row_id": row["row_id"],
                    "question_hash": row["question_hash"],
                    "attempt": record["attempt"],
                    "solution": record["text"],
                }) + "\n")
    temporary.replace(out_path)
    return sum(row["row_id"] in kept for row in rows)

def percentile(values: list[float], share: float) -> float | None:
    """
    Gets a nearest-rank percentile.

    Args:
        values (list[float]): Values.
        share (float): Percentile as a share, for example 0.99.
    Returns:
        The percentile, or None if there are no values.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]

def summarize_attempts(rows: list[dict], attempts: list[dict]) -> dict:
    """
    Summarizes the attempts for the lab log. Accepted solutions pass by chance too, so the first-try rate is reported.

    Args:
        rows (list[dict]): Rows the attempts were for.
        attempts (list[dict]): Records from the log.
    Returns:
        Calls, money, tokens, truncation, responses with no valid answer, and the first-try and solved shares, overall and per task and type.
    """
    kept = best_attempts(attempts)
    types = {row["row_id"]: row["type"] for row in rows}
    first_tries = {record["row_id"]: record["correct"] for record in attempts if record["attempt"] == 1}
    completion_tokens = [record["completion_tokens"] for record in attempts if record["completion_tokens"] is not None]

    def shares(selected: list[dict]) -> dict:
        tried = [row for row in selected if row["row_id"] in first_tries]
        return {
            "rows": len(selected),
            "first_try_rows": len(tried),
            "first_try_pass_rate": sum(first_tries[row["row_id"]] for row in tried) / len(tried) if tried else None,
            "solved": sum(row["row_id"] in kept for row in selected),
        }

    groups = defaultdict(list)
    for row in rows:
        groups[f"task:{row['details']['task']}"].append(row)
        groups[f"type:{row['type']}"].append(row)

    return {
        "calls": len(attempts),
        "cost_usd": sum(record["cost_usd"] for record in attempts),
        "cost_per_call_usd": sum(record["cost_usd"] for record in attempts) / len(attempts) if attempts else None,
        "completion_tokens_mean": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
        "completion_tokens_p99": percentile(completion_tokens, 0.99),
        "completion_tokens_max": max(completion_tokens, default=None),
        "truncated": sum(record["finish_reason"] != "stop" for record in attempts),
        "no_valid_answer": sum(extract_answer(record["text"], types[record["row_id"]]) is None for record in attempts if record["row_id"] in types),
        "models": dict(Counter(str(record["model"]) for record in attempts)),
        "providers": dict(Counter(str(record["provider"]) for record in attempts)),
        "overall": shares(rows),
        "groups": {name: shares(group) for name, group in sorted(groups.items())},
    }

def expected_tries(pass_rate: float, max_tries: int) -> float:
    """
    Gets the expected number of tries per row, from a geometric distribution cut at max_tries.

    Args:
        pass_rate (float): Chance that one try is correct.
        max_tries (int): Maximum number of tries per row.
    Returns:
        The expected number of tries.
    """
    return (1 - (1 - pass_rate) ** max_tries) / pass_rate if pass_rate > 0 else float(max_tries)

def project_full_run(train_rows: list[dict], test_rows: list[dict], test_attempts: list[dict], budget_usd: float, max_tries: int = MAX_TRIES) -> dict:
    """
    Projects the cost of the full run from a first-try test, as a range. The test has few rows per task, so neither end is exact.
    The low end pools pass rates by type, which ignores that hard rows fail repeatedly.
    The high end uses each task's own rate, which overstates tasks that scored low by chance in a small sample.

    Args:
        train_rows (list[dict]): All the train rows.
        test_rows (list[dict]): Rows of the test.
        test_attempts (list[dict]): First-try records of the test.
        budget_usd (float): Spending limit.
        max_tries (int): Maximum number of tries per row.
    Returns:
        The cost per call, the pass rates, the low and high expected cost, the cost of using every try, and the calls the budget can pay for.
    """
    first_tries = {record["row_id"]: record["correct"] for record in test_attempts if record["attempt"] == 1}
    if not first_tries:
        raise ValueError("The test has no first tries.")
    cost_per_call = sum(record["cost_usd"] for record in test_attempts) / len(test_attempts)

    def calls_by(group: Callable) -> tuple[float, dict]:
        results = defaultdict(list)
        for row in test_rows:
            if row["row_id"] in first_tries:
                results[group(row)].append(first_tries[row["row_id"]])
        # A group the test missed counts as unsolvable, which errs towards a higher cost
        rates = {name: sum(results[name]) / len(results[name]) if results[name] else 0.0 for name in {group(row) for row in train_rows}}
        return sum(expected_tries(rates[group(row)], max_tries) for row in train_rows), rates

    low_calls, type_rates = calls_by(lambda row: row["type"])
    high_calls, task_rates = calls_by(lambda row: row["details"]["task"])
    return {
        "cost_per_call_usd": cost_per_call,
        "pass_rates_by_type": type_rates,
        "pass_rates_by_task": task_rates,
        "expected_cost_low_usd": low_calls * cost_per_call,
        "expected_cost_high_usd": high_calls * cost_per_call,
        "all_tries_cost_usd": len(train_rows) * max_tries * cost_per_call,
        "calls_within_budget": budget_usd / cost_per_call if cost_per_call > 0 else None,
    }
