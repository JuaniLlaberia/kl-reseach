import json
from collections import defaultdict
from pathlib import Path

from src.data.chem import build_prompt, get_gold
from src.data.solutions import percentile
from src.eval.extract import is_correct
from src.eval.retention import bootstrap_ci
from src.kl.sampling import build_prompt_ids, load_responses, sample_responses

def build_chem_prompts(tokenizer, rows: list[dict]) -> list[dict]:
    """
    Builds the token IDs of chemistry prompts, with the row as the prompt ID.

    Args:
        tokenizer: Tokenizer of the model.
        rows (list[dict]): Rows to build.
    Returns:
        Prompts with the fields prompt_id and prompt_ids.
    """
    return build_prompt_ids(tokenizer, [{"prompt_id": row["row_id"], "text": build_prompt(row)} for row in rows])

def score_responses(rows: list[dict], responses: list[dict]) -> tuple[dict, list[dict]]:
    """
    Scores sampled responses with the same is_correct as the GRPO reward.

    Args:
        rows (list[dict]): The rows that were sampled.
        responses (list[dict]): Response records with the fields prompt_id, text and truncated.
    Returns:
        The accuracy with its CI overall and per type and task, and one row per response.
    """
    by_id = {row["row_id"]: row for row in rows}
    if {record["prompt_id"] for record in responses} != set(by_id) or len(responses) != len(rows):
        raise ValueError(f"Expected one response for each of the {len(rows)} rows, got {len(responses)}.")

    scored = []
    groups = defaultdict(list)
    for record in responses:
        row = by_id[record["prompt_id"]]
        correct = is_correct(record["text"], get_gold(row), row["type"], record["truncated"])
        scored.append({"row_id": row["row_id"], "correct": correct, "truncated": record["truncated"], "text": record["text"]})
        for group in ("overall", f"type:{row['type']}", f"task:{row['details']['task']}"):
            groups[group].append(float(correct))

    scores = {
        group: {"accuracy": sum(values) / len(values), "ci95": bootstrap_ci(values), "n": len(values)}
        for group, values in sorted(groups.items())
    }
    return scores, scored

def measure_accuracy(model, tokenizer, rows: list[dict], out_dir: Path, checkpoint: str, seed: int, batch_size: int, max_new_tokens: int, model_info: dict) -> dict:
    """
    Measures accuracy from one sampled response per row, using the fixed sampling settings of the KL measurement.
    It resumes from its responses file, like sample_responses.

    Args:
        model: Model, already loaded in fp32.
        tokenizer: Tokenizer of the model.
        rows (list[dict]): Rows to score, for example the held-out split.
        out_dir (Path): Folder of this run's results.
        checkpoint (str): Checkpoint name, which names the files.
        seed (int): Base seed of the sampling.
        batch_size (int): Number of prompts per batch.
        max_new_tokens (int): Maximum number of tokens per response.
        model_info (dict): Model name and revision or checkpoint, saved with the responses.
    Returns:
        The accuracy with its CI overall and per type and task.
    """
    out_dir = Path(out_dir)
    responses_path = out_dir / f"{checkpoint}_new_task_responses.jsonl"
    sample_responses(model, tokenizer, build_chem_prompts(tokenizer, rows), responses_path, seed, batch_size, max_new_tokens, model_info)
    _, responses = load_responses(responses_path)
    scores, scored = score_responses(rows, responses)

    with open(out_dir / f"{checkpoint}_new_task_scores.jsonl", "w", encoding="utf-8") as file:
        for row in scored:
            file.write(json.dumps(row) + "\n")
    (out_dir / f"{checkpoint}_new_task.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores

def length_stats(lengths: list[int], truncated: list[bool], caps: tuple[int, ...] = (256, 384, 512, 768, 1024)) -> dict:
    """
    Summarizes response lengths, which feed the choice of max_new_tokens.

    Args:
        lengths (list[int]): Tokens per response.
        truncated (list[bool]): Whether each response hit the length cap without stopping.
        caps (tuple[int, ...]): Caps to report.
    Returns:
        The percentiles of the lengths, the share cut at the sampling cap, and the share longer than each cap.
    """
    return {
        "n": len(lengths),
        "p50": percentile(lengths, 0.5),
        "p90": percentile(lengths, 0.9),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "truncated_share": sum(truncated) / len(truncated),
        "share_longer_than": {str(cap): sum(length > cap for length in lengths) / len(lengths) for cap in caps},
    }
