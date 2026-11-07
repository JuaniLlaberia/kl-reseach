import pytest

from src.eval.minipilot import check_kl_is_zero, check_kl_not_negative, check_kl_rows, check_retention_complete
from src.eval.retention import EVAL_RUNS
from src.kl.measure import measure_kl

def test_base_against_itself_gives_zero_on_every_variant(base_model, base_path):
    scalars, rows = measure_kl(base_model, base_model, base_path, base_path, 16)
    check_kl_is_zero(scalars)
    check_kl_not_negative(scalars)
    check_kl_rows(rows, 300)

def test_a_different_model_is_not_zero_and_is_not_negative(base_model, ft_model, base_path, ft_path):
    scalars, _ = measure_kl(base_model, ft_model, base_path, ft_path, 16)
    with pytest.raises(ValueError, match="isn't zero"):
        check_kl_is_zero(scalars)
    check_kl_not_negative(scalars)

def test_negative_exact_kl_fails_but_k1_is_allowed():
    check_kl_not_negative({"kl_k1_rev_ft_mean_token": -0.5})
    with pytest.raises(ValueError, match="negative"):
        check_kl_not_negative({"kl_exact_fwd_base_mean_token": -0.5})

def test_missing_rows_fail():
    with pytest.raises(ValueError, match="ft rows"):
        check_kl_rows([{"prompt_id": 0, "source": "base"}], 1)

def complete_results(**changes) -> dict:
    return {"checkpoint": "base", "limit": None, "retention": {run["task"]: {"scores": {"acc": {}}} for run in EVAL_RUNS}, **changes}

def test_complete_retention_passes():
    check_retention_complete(complete_results())

def test_missing_benchmark_fails():
    results = complete_results()
    del results["retention"]["mmlu"]
    with pytest.raises(ValueError, match="mmlu"):
        check_retention_complete(results)

def test_smoke_limit_fails():
    with pytest.raises(ValueError, match="limit"):
        check_retention_complete(complete_results(limit=2))
