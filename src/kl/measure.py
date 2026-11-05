import json
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from src.kl.divergence import aggregate, exact_kl_per_token, k_estimators
from src.kl.sampling import load_responses

# Every reported KL value is computed in this precision. bf16 is only for the drift test.
REPORTED_DTYPE = torch.float32

def load_model(path: str, dtype: torch.dtype, device: str, revision: str | None = None):
    """
    Loads a model for measurement. Precision is always explicit, as Qwen's config declares bf16.

    Args:
        path (str): Hugging Face model name or local checkpoint path.
        dtype (torch.dtype): Precision of the model (fp32 for every reported KL).
        device (str): Device to load the model on.
        revision (str | None): Hugging Face commit of the model, if it has one.
    Returns:
        The model in eval mode.
    """
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, revision=revision, attn_implementation="sdpa")
    return model.to(device).eval()

def right_pad(records: list[dict]) -> tuple[Tensor, Tensor]:
    """
    Joins prompts and responses into a right-padded batch.

    Args:
        records (list[dict]): Response records with the fields prompt_ids and response_ids.
    Returns:
        The token IDs and their attention mask. Padded positions are masked, so their value doesn't matter.
    """
    sequences = [record["prompt_ids"] + record["response_ids"] for record in records]
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.tensor([sequence + [0] * (max_length - len(sequence)) for sequence in sequences])
    attention_mask = torch.tensor([[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in sequences])
    return input_ids, attention_mask

def response_positions(prompt_lengths: Tensor, total_lengths: Tensor) -> tuple[Tensor, Tensor]:
    """
    Finds the logits that predict response tokens. The token at position t is predicted by the logits at t - 1.

    Args:
        prompt_lengths (Tensor): Number of prompt tokens of each sequence.
        total_lengths (Tensor): Number of prompt and response tokens of each sequence.
    Returns:
        The positions whose logits to keep, with shape (kept), and a mask over them with shape (batch, kept).
    """
    keep = torch.arange(prompt_lengths.min() - 1, total_lengths.max() - 1)
    # A sequence uses the logits from position P - 1 to T - 2, which predict tokens P to T - 1
    mask = (keep >= (prompt_lengths - 1).unsqueeze(1)) & (keep <= (total_lengths - 2).unsqueeze(1))
    return keep, mask

def response_log_probs(model, input_ids: Tensor, attention_mask: Tensor, keep: Tensor, mask: Tensor) -> Tensor:
    """
    Computes the full-vocabulary log-probabilities that predict each response token.

    Args:
        model: Model to run.
        input_ids (Tensor): Right-padded token IDs.
        attention_mask (Tensor): Attention mask of input_ids.
        keep (Tensor): Positions whose logits to keep.
        mask (Tensor): Mask selecting the response positions among the kept ones.
    Returns:
        Log-probabilities with shape (response tokens in the batch, vocab_size), always in fp32.
    """
    logits = model(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        logits_to_keep=keep.to(model.device),
        use_cache=False,
    ).logits
    # The KL math is always fp32, even when the drift test runs the model in bf16
    return torch.log_softmax(logits[mask.to(model.device)].float(), dim=-1)

@torch.no_grad()
def measure_batch(base_model, ft_model, records: list[dict], source: str) -> dict[str, Tensor]:
    """
    Computes the per-sequence KL totals of one batch of responses.

    Args:
        base_model: Base model.
        ft_model: Fine-tuned model.
        records (list[dict]): Response records of the batch.
        source (str): Which model sampled the responses ("base" or "ft").
    Returns:
        Dictionary from variant name to the KL total of each sequence.
    """
    input_ids, attention_mask = right_pad(records)
    prompt_lengths = torch.tensor([len(record["prompt_ids"]) for record in records])
    total_lengths = prompt_lengths + torch.tensor([len(record["response_ids"]) for record in records])
    keep, mask = response_positions(prompt_lengths, total_lengths)

    log_base = response_log_probs(base_model, input_ids, attention_mask, keep, mask)
    log_ft = response_log_probs(ft_model, input_ids, attention_mask, keep, mask)

    per_token = {
        # Forward is KL(base || ft), reverse is KL(ft || base)
        f"kl_exact_fwd_{source}": exact_kl_per_token(log_base, log_ft),
        f"kl_exact_rev_{source}": exact_kl_per_token(log_ft, log_base),
    }
    if source == "ft":
        # The token predicted by the logits at position t is the one at t + 1
        tokens = input_ids[:, keep + 1][mask].to(log_base.device).unsqueeze(1)
        log_base_y = log_base.gather(1, tokens).squeeze(1)
        log_ft_y = log_ft.gather(1, tokens).squeeze(1)
        for name, values in k_estimators(log_base_y, log_ft_y).items():
            per_token[f"kl_{name}_rev_ft"] = values

    # mask.nonzero() gives the sequence that each response token belongs to
    sequence_index = mask.nonzero()[:, 0].to(log_base.device)
    return {
        name: torch.zeros(len(records), device=values.device, dtype=values.dtype).index_add_(0, sequence_index, values).cpu()
        for name, values in per_token.items()
    }

def measure_source(base_model, ft_model, records: list[dict], source: str, batch_size: int) -> dict[str, Tensor]:
    """
    Computes the per-sequence KL totals of all the responses sampled by one model.

    Args:
        base_model: Base model.
        ft_model: Fine-tuned model.
        records (list[dict]): Response records.
        source (str): Which model sampled the responses ("base" or "ft").
        batch_size (int): Number of responses per batch.
    Returns:
        Dictionary from variant name to the KL total of each sequence.
    """
    batches = [
        measure_batch(base_model, ft_model, records[start:start + batch_size], source)
        for start in range(0, len(records), batch_size)
    ]
    return {name: torch.cat([batch[name] for batch in batches]) for name in batches[0]}

def check_same_setup(base_header: dict, ft_header: dict, base_records: list[dict], ft_records: list[dict], base_model, ft_model) -> None:
    """
    Fails if the two sets of responses or the two models don't share the same setup.

    Args:
        base_header (dict): Header of the base responses.
        ft_header (dict): Header of the fine-tuned responses.
        base_records (list[dict]): Base response records.
        ft_records (list[dict]): Fine-tuned response records.
        base_model: Base model.
        ft_model: Fine-tuned model.
    Returns:
        None.
    """
    ignored = ("model_info", "seed")
    for key in base_header:
        if key not in ignored and base_header[key] != ft_header.get(key):
            raise ValueError(f"Base and fine-tuned responses differ in {key}.")
    if [(r["prompt_id"], r["prompt_ids"]) for r in base_records] != [(r["prompt_id"], r["prompt_ids"]) for r in ft_records]:
        raise ValueError("Base and fine-tuned responses don't use the same prompts.")
    if base_model.config.vocab_size != ft_model.config.vocab_size:
        raise ValueError("Base and fine-tuned models have different vocabulary sizes.")
    if base_model.dtype != ft_model.dtype:
        raise ValueError("Base and fine-tuned models have different precisions.")

def check_precision(model_dtype: torch.dtype, response_dtype: str, allow_bf16: bool) -> None:
    """
    Fails if the measurement doesn't follow the fp32 rule. Responses must always be sampled in fp32.

    Args:
        model_dtype (torch.dtype): Precision of the models.
        response_dtype (str): Precision the responses were sampled in, from their header.
        allow_bf16 (bool): Whether the models can run in bf16, only for the drift test.
    Returns:
        None.
    """
    if response_dtype != str(REPORTED_DTYPE):
        raise ValueError(f"Responses were sampled in {response_dtype}, but they must be sampled in {REPORTED_DTYPE}.")
    if model_dtype != REPORTED_DTYPE and not (allow_bf16 and model_dtype == torch.bfloat16):
        raise ValueError(f"Models are in {model_dtype}, but KL must be measured in {REPORTED_DTYPE}.")

def measure_kl(base_model, ft_model, base_path: Path, ft_path: Path, batch_size: int, allow_bf16: bool = False) -> tuple[dict[str, float], list[dict]]:
    """
    Computes every KL variant for one checkpoint.

    Args:
        base_model: Base model.
        ft_model: Fine-tuned model.
        base_path (Path): Responses sampled from the base model.
        ft_path (Path): Responses sampled from this checkpoint.
        batch_size (int): Number of responses per batch.
        allow_bf16 (bool): Whether the models can run in bf16, only for the drift test.
    Returns:
        A dictionary with every KL variant under its full name, and one row per response and source.
    """
    base_header, base_records = load_responses(base_path)
    ft_header, ft_records = load_responses(ft_path)
    check_same_setup(base_header, ft_header, base_records, ft_records, base_model, ft_model)
    check_precision(base_model.dtype, base_header["dtype"], allow_bf16)

    scalars = {}
    rows = []
    for source, records in (("base", base_records), ("ft", ft_records)):
        totals = measure_source(base_model, ft_model, records, source, batch_size)
        lengths = torch.tensor([len(record["response_ids"]) for record in records])

        for name, seq_totals in totals.items():
            for aggregation, value in aggregate(seq_totals, lengths).items():
                scalars[f"{name}_{aggregation}"] = value

        for index, record in enumerate(records):
            rows.append({
                "prompt_id": record["prompt_id"],
                "source": source,
                "n_tokens": len(record["response_ids"]),
                "truncated": record["truncated"],
                **{name: seq_totals[index].item() for name, seq_totals in totals.items()},
            })
    return scalars, rows

def save_kl_results(scalars: dict[str, float], rows: list[dict], results_path: Path, dtype: torch.dtype) -> None:
    """
    Adds the KL variants to a checkpoint's results JSON and writes the per-sequence file next to it.
    Only fp32 values can be saved, so drift test values can't overwrite reported results.

    Args:
        scalars (dict[str, float]): Every KL variant under its full name.
        rows (list[dict]): One row per response and source.
        results_path (Path): Path of the results JSON (for example results/<run>/final.json).
        dtype (torch.dtype): Precision the values were measured in.
    Returns:
        None.
    """
    if dtype != REPORTED_DTYPE:
        raise ValueError(f"Only {REPORTED_DTYPE} KL values can be saved, got {dtype}.")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results.update(scalars)
    results["kl_dtype"] = str(dtype)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with open(results_path.with_name(f"{results_path.stem}_kl_seq.jsonl"), "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
