import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.extract import FILLING, MCQ

DATASET = "hicai-zju/SciKnowEval"
REVISION = "92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2"
DATA_FILE = "data/v2/sciknoweval_test_v2.jsonl"

EXPECTED_COUNTS = {MCQ: 2100, FILLING: 300}
N_HELDOUT = 600
N_KL = 500
SPLIT_SEED = 0

REASONING = "Think step by step, then finish with a final line in exactly this form:"

def load_chem_rows() -> list[dict]:
    """
    Downloads the pinned dataset file and filters it to Chemistry L3.

    Returns:
        The Chemistry L3 rows, in file order.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATA_FILE, revision=REVISION, repo_type="dataset")
    with open(path, encoding="utf-8") as file:
        return filter_chem_rows([json.loads(line) for line in file])

def filter_chem_rows(raw_rows: list[dict]) -> list[dict]:
    """
    Keeps the Chemistry L3 rows, identified by their position in the pinned file and a hash of their question.

    Args:
        raw_rows (list[dict]): All the rows of the file, in file order.
    Returns:
        The Chemistry L3 rows, with the added fields row_id and question_hash.
    """
    return [
        {
            **row,
            "row_id": index,
            "question_hash": hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:16],
        }
        for index, row in enumerate(raw_rows)
        if row["domain"] == "Chemistry" and row["details"]["level"] == "L3"
    ]

def check_counts(rows: list[dict], expected: dict[str, int] = EXPECTED_COUNTS) -> None:
    """
    Fails if the rows aren't the ones recorded in the lab log.

    Args:
        rows (list[dict]): Chemistry L3 rows.
        expected (dict[str, int]): Expected number of rows per type.
    Returns:
        None.
    """
    counts = dict(Counter(row["type"] for row in rows))
    if counts != expected:
        raise ValueError(f"Expected {expected} rows per type, got {counts}.")

def get_gold(row: dict) -> str:
    """
    Gets the correct answer of a row.

    Args:
        row (dict): Dataset row.
    Returns:
        The correct letter (MCQ) or the balanced equation (filling).
    """
    return row["answerKey"] if row["type"] == MCQ else row["answer"]

def build_prompt(row: dict) -> str:
    """
    Builds the user message of a row. The same text goes to GPT-4o, SFT, GRPO, the eval and KL.
    It replaces the dataset's "without any explanation" instruction, as logged on 06/11/26.

    Args:
        row (dict): Dataset row.
    Returns:
        The user message, which never contains the answer.
    """
    if row["type"] == MCQ:
        choices = "\n".join(f"{label}. {text}" for label, text in zip(row["choices"]["label"], row["choices"]["text"]))
        return f"{row['question']}\n\n{choices}\n\n{REASONING}\nAnswer: <letter>"
    if row["type"] == FILLING:
        return (
            f"{row['question']}\n\n"
            f"Balance the equation and keep the reactants and products in the order given. {REASONING}\n"
            "Answer: <balanced equation>"
        )
    raise ValueError(f"Unknown row type: {row['type']}")

def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """
    Divides a total across strata in proportion to their size, using the largest remainder.

    Args:
        counts (dict[str, int]): Size of each stratum.
        total (int): Number of items to allocate.
    Returns:
        Number of items for each stratum.
    """
    size = sum(counts.values())
    quotas = {name: count * total // size for name, count in counts.items()}
    # Ties in the remainder are broken by name, so the result never depends on dictionary order
    by_remainder = sorted(counts, key=lambda name: (-(counts[name] * total % size), name))
    for name in by_remainder[:total - sum(quotas.values())]:
        quotas[name] += 1
    return quotas

def stratified_sample(rows: list[dict], total: int, seed: int) -> list[dict]:
    """
    Samples rows in proportion to the size of each subtask.

    Args:
        rows (list[dict]): Rows to sample from.
        total (int): Number of rows to sample.
        seed (int): Seed of the sampling.
    Returns:
        The sampled rows, ordered by row_id.
    """
    by_subtask = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["row_id"]):
        by_subtask[row["details"]["subtask"]].append(row)

    rng = random.Random(seed)
    quotas = allocate({name: len(group) for name, group in by_subtask.items()}, total)
    sampled = [row for name in sorted(by_subtask) for row in rng.sample(by_subtask[name], quotas[name])]
    return sorted(sampled, key=lambda row: row["row_id"])

def split_rows(rows: list[dict], seed: int = SPLIT_SEED, n_heldout: int = N_HELDOUT, n_kl: int = N_KL) -> dict:
    """
    Splits the rows into train and held-out, stratified by subtask, and draws the KL prompts from held-out.
    Rows that share a question always stay on the same side, even when their choices differ.

    Args:
        rows (list[dict]): Chemistry L3 rows.
        seed (int): Seed of the split.
        n_heldout (int): Number of held-out rows.
        n_kl (int): Number of KL prompts, drawn from the held-out rows.
    Returns:
        The split, with the row_id and question_hash of the train, held-out and KL rows.
    """
    ordered = sorted(rows, key=lambda row: row["row_id"])
    subtasks = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    for row in ordered:
        subtasks[row["details"]["subtask"]][row["question_hash"]].append(row)
        counts[row["details"]["subtask"]] += 1

    rng = random.Random(seed)
    quotas = allocate(dict(counts), n_heldout)
    heldout = []
    for name in sorted(subtasks):
        groups = list(subtasks[name].values())
        rng.shuffle(groups)
        # Takes each group that still fits, so a repeated question can't push a subtask over its quota
        taken = 0
        for group in groups:
            if taken + len(group) <= quotas[name]:
                heldout.extend(group)
                taken += len(group)
        if taken != quotas[name]:
            raise ValueError(f"Couldn't fill the held-out quota of {name}: {taken} of {quotas[name]}.")

    heldout_ids = {row["row_id"] for row in heldout}
    identify = lambda selected: [{"row_id": row["row_id"], "question_hash": row["question_hash"]} for row in sorted(selected, key=lambda row: row["row_id"])]
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "seed": seed,
        "train": identify(row for row in ordered if row["row_id"] not in heldout_ids),
        "heldout": identify(heldout),
        "kl": identify(stratified_sample(heldout, n_kl, seed)),
    }

def select_rows(rows: list[dict], split: dict, side: str) -> list[dict]:
    """
    Gets the rows of one side of a split, and fails if they don't match the rows it was made from.

    Args:
        rows (list[dict]): Chemistry L3 rows.
        split (dict): The split.
        side (str): train, heldout or kl.
    Returns:
        The rows of that side.
    """
    by_id = {row["row_id"]: row for row in rows}
    selected = []
    for item in split[side]:
        row = by_id.get(item["row_id"])
        if row is None or row["question_hash"] != item["question_hash"]:
            raise ValueError(f"Row {item['row_id']} doesn't match the split, so the dataset has changed.")
        selected.append(row)
    return selected

def save_split(split: dict, path: Path) -> None:
    """
    Saves a split in a stable format, so the same split always gives the same file.

    Args:
        split (dict): The split.
        path (Path): Path of the JSON file.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1, sort_keys=True) + "\n", encoding="utf-8")

def load_split(path: Path) -> dict:
    """
    Loads a saved split.

    Args:
        path (Path): Path of the JSON file.
    Returns:
        The split.
    """
    return json.loads(path.read_text(encoding="utf-8"))
