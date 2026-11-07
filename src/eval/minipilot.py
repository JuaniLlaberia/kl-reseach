from src.eval.retention import EVAL_RUNS

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
N_PROMPTS = 500
# Rounding in fp32 leaves differences this small when a model is compared with itself
TOLERANCE = 1e-5

def load_gsm8k_prompts(n_prompts: int = N_PROMPTS, seed: int = 0) -> list[dict]:
    """
    Gets the mini-pilot's prompts, from the pinned GSM8K test split. Each one is the question alone.

    Args:
        n_prompts (int): Number of prompts.
        seed (int): Seed of the selection.
    Returns:
        Prompts with the fields prompt_id and text.
    """
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test", revision=GSM8K_REVISION).shuffle(seed=seed).select(range(n_prompts))
    return [{"prompt_id": index, "text": row["question"]} for index, row in enumerate(rows)]

def check_kl_is_zero(scalars: dict[str, float]) -> None:
    """
    Fails unless every KL variant is zero when a model is compared with itself.

    Args:
        scalars (dict[str, float]): Every KL variant from measure_kl.
    Returns:
        None.
    """
    wrong = {name: value for name, value in scalars.items() if abs(value) > TOLERANCE}
    if wrong:
        raise ValueError(f"KL(base || base) isn't zero: {wrong}")

def check_kl_not_negative(scalars: dict[str, float]) -> None:
    """
    Fails if an exact KL or a squared estimator is negative. k1 is skipped, as it can be negative in a finite sample.

    Args:
        scalars (dict[str, float]): Every KL variant from measure_kl.
    Returns:
        None.
    """
    wrong = {name: value for name, value in scalars.items() if not name.startswith("kl_k1") and value < -TOLERANCE}
    if wrong:
        raise ValueError(f"These KL variants are negative: {wrong}")

def check_kl_rows(rows: list[dict], n_prompts: int) -> None:
    """
    Fails unless the per-sequence file covers every prompt, for both the base and the fine-tuned responses.

    Args:
        rows (list[dict]): Rows from measure_kl.
        n_prompts (int): Number of prompts.
    Returns:
        None.
    """
    for source in ("base", "ft"):
        found = {row["prompt_id"] for row in rows if row["source"] == source}
        if len(found) != n_prompts:
            raise ValueError(f"Expected {n_prompts} {source} rows, got {len(found)}.")

def check_retention_complete(results: dict, limit: int | None = None) -> None:
    """
    Fails unless a checkpoint has a score for every benchmark of the suite, with no limit on the documents.

    Args:
        results (dict): Content of the results file from run_retention.
        limit (int | None): The limit this run is expected to have.
    Returns:
        None.
    """
    missing = [run["task"] for run in EVAL_RUNS if not results["retention"].get(run["task"], {}).get("scores")]
    if missing:
        raise ValueError(f"{results['checkpoint']} has no scores for {missing}.")
    if results["limit"] != limit:
        raise ValueError(f"{results['checkpoint']} ran with limit={results['limit']}, not {limit}.")
