import torch
from torch import Tensor

def exact_kl_per_token(log_p: Tensor, log_q: Tensor) -> Tensor:
    """
    Computes the exact KL(p || q) at each position, over the full vocabulary.

    Args:
        log_p (Tensor): Log-probabilities of p, with shape (..., vocab_size).
        log_q (Tensor): Log-probabilities of q, with the same shape as log_p.
    Returns:
        KL value per position, with shape (...).
    """
    # KL(p || q) = sum_v p(v) * (log p(v) - log q(v))
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)

def k_estimators(log_base_y: Tensor, log_ft_y: Tensor) -> dict[str, Tensor]:
    """
    Computes the k1, k2 and k3 estimators of KL(ft || base) for tokens y sampled from the fine-tuned model.

    Args:
        log_base_y (Tensor): Log-probability of each sampled token under the base model.
        log_ft_y (Tensor): Log-probability of each sampled token under the fine-tuned model.
    Returns:
        Dictionary with the k1, k2 and k3 values per token.
    """
    # log r = log(pi_0(y) / pi(y)) = log pi_0(y) - log pi(y)
    log_r = log_base_y - log_ft_y
    return {
        # k1 = -log r
        "k1": -log_r,
        # k2 = 1/2 * (log r)^2
        "k2": 0.5 * log_r.pow(2),
        # k3 = (r - 1) - log r, where expm1(log r) = r - 1
        "k3": torch.expm1(log_r) - log_r,
    }

def aggregate(seq_totals: Tensor, seq_lengths: Tensor) -> dict[str, float]:
    """
    Aggregates per-sequence KL totals across prompts.

    Args:
        seq_totals (Tensor): Sum of the per-token values of each response.
        seq_lengths (Tensor): Number of response tokens of each response.
    Returns:
        Dictionary with the mean_token and sum_seq aggregations.
    """
    totals = seq_totals.double()
    return {
        # mean_token = 1/N * sum_i (total_i / length_i)
        "mean_token": (totals / seq_lengths.double()).mean().item(),
        # sum_seq = 1/N * sum_i total_i
        "sum_seq": totals.mean().item(),
    }
