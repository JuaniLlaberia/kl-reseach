import math

import pytest
import torch

from src.kl.divergence import aggregate, exact_kl_per_token, k_estimators

def test_exact_kl_matches_hand_computed_values_in_both_directions():
    """Swapped arguments give the wrong value, as KL isn't symmetric."""
    log_p = torch.tensor([0.5, 0.5]).log()
    log_q = torch.tensor([0.9, 0.1]).log()

    forward = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    reverse = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)

    assert exact_kl_per_token(log_p, log_q).item() == pytest.approx(forward, abs=1e-6)
    assert exact_kl_per_token(log_q, log_p).item() == pytest.approx(reverse, abs=1e-6)

def test_exact_kl_of_a_distribution_with_itself_is_zero():
    log_p = torch.log_softmax(torch.randn(100, 50), dim=-1)
    assert exact_kl_per_token(log_p, log_p).abs().max().item() == 0

def test_exact_kl_is_never_negative():
    """Rounding can go slightly below zero, so this uses a small tolerance rather than clamping."""
    torch.manual_seed(0)
    log_p = torch.log_softmax(torch.randn(2000, 50), dim=-1)
    log_q = torch.log_softmax(log_p + torch.randn(2000, 50) * 0.01, dim=-1)
    assert exact_kl_per_token(log_p, log_q).min().item() >= -1e-6

def test_k_estimators_match_hand_computed_values():
    """With pi_0(y) = 0.1 and pi(y) = 0.4, r = 0.25."""
    values = k_estimators(torch.tensor(0.1).log(), torch.tensor(0.4).log())
    log_r = math.log(0.25)

    assert values["k1"].item() == pytest.approx(-log_r, abs=1e-5)
    assert values["k2"].item() == pytest.approx(0.5 * log_r ** 2, abs=1e-5)
    assert values["k3"].item() == pytest.approx((0.25 - 1) - log_r, abs=1e-5)

def test_k1_can_be_negative_for_a_single_token():
    """Only its average estimates KL, so the non-negative rule doesn't apply to it."""
    assert k_estimators(torch.tensor(0.4).log(), torch.tensor(0.1).log())["k1"].item() < 0

def test_aggregate_matches_hand_computed_values():
    """Totals (2, 9) over lengths (2, 3): the mean of means is (1 + 3) / 2, not pooled 11 / 5."""
    values = aggregate(torch.tensor([2.0, 9.0]), torch.tensor([2, 3]))
    assert values["mean_token"] == pytest.approx(2.0)
    assert values["sum_seq"] == pytest.approx(5.5)
