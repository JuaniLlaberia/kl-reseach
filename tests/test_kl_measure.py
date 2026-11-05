import copy
import json

import pytest
import torch

from src.kl.measure import measure_kl, response_log_probs, response_positions, save_kl_results
from src.kl.sampling import load_responses

EXACT_SEQUENCE_TOTALS = ["kl_exact_fwd_base", "kl_exact_rev_base", "kl_exact_fwd_ft", "kl_exact_rev_ft"]
K_SEQUENCE_TOTALS = ["kl_k1_rev_ft", "kl_k2_rev_ft", "kl_k3_rev_ft"]

@pytest.fixture(scope="module")
def measured(base_model, ft_model, base_path, ft_path):
    return measure_kl(base_model, ft_model, base_path, ft_path, 32)

def test_every_variant_has_its_full_name(measured):
    scalars, _ = measured
    assert set(scalars) == {
        f"{name}_{aggregation}"
        for name in EXACT_SEQUENCE_TOTALS + K_SEQUENCE_TOTALS
        for aggregation in ("mean_token", "sum_seq")
    }

def test_kl_of_base_with_itself_is_zero(base_model, base_path):
    """PLAN: KL(base || base) = 0. Uses a separate copy, so two real forward passes are compared."""
    scalars, _ = measure_kl(base_model, copy.deepcopy(base_model), base_path, base_path, 32)
    assert max(abs(value) for value in scalars.values()) < 1e-5

def test_exact_kl_is_never_negative_on_real_responses(measured):
    """PLAN: KL is never negative. Exact KL only, as k1 can go negative."""
    _, rows = measured
    for row in rows:
        for name in EXACT_SEQUENCE_TOTALS:
            if name in row:
                assert row[name] >= -1e-5

def test_average_k1_is_close_to_exact_kl(measured):
    """
    PLAN: the average k1 is close to the exact KL. With y ~ pi, k1 estimates KL(pi || pi_0),
    so it's compared with exact reverse KL on the fine-tuned samples.
    """
    _, rows = measured
    ft_rows = [row for row in rows if row["source"] == "ft"]
    differences = torch.tensor([row["kl_k1_rev_ft"] - row["kl_exact_rev_ft"] for row in ft_rows], dtype=torch.float64)
    standard_error = differences.std() / len(differences) ** 0.5
    assert abs(differences.mean()) < 4 * standard_error

def test_average_k3_is_close_to_exact_kl(measured):
    """k3 is also unbiased, as E_pi[r - 1] = 0."""
    _, rows = measured
    ft_rows = [row for row in rows if row["source"] == "ft"]
    differences = torch.tensor([row["kl_k3_rev_ft"] - row["kl_exact_rev_ft"] for row in ft_rows], dtype=torch.float64)
    standard_error = differences.std() / len(differences) ** 0.5
    assert abs(differences.mean()) < 4 * standard_error

def test_bf16_drift_from_fp32(base_model, ft_model, base_path, ft_path, measured):
    """PLAN: record the bf16 drift. This isn't asserted; the real number needs the real model on the A100."""
    fp32_scalars, _ = measured
    bf16_base = copy.deepcopy(base_model).to(torch.bfloat16)
    bf16_ft = copy.deepcopy(ft_model).to(torch.bfloat16)
    bf16_scalars, _ = measure_kl(bf16_base, bf16_ft, base_path, ft_path, 32, allow_bf16=True)

    assert set(bf16_scalars) == set(fp32_scalars)
    primary = "kl_exact_fwd_base_mean_token"
    print(f"bf16 drift on the tiny model ({primary}): {abs(bf16_scalars[primary] - fp32_scalars[primary]):.2e}")

def test_alignment_on_a_hand_checked_case():
    """Sequences with P=3, T=5 and P=2, T=4 use the logits at positions (2, 3) and (1, 2)."""
    keep, mask = response_positions(torch.tensor([3, 2]), torch.tensor([5, 4]))
    assert keep.tolist() == [1, 2, 3]
    assert mask.tolist() == [[False, True, True], [True, True, False]]

@pytest.mark.parametrize("source", ["base", "ft"])
def test_totals_match_an_independent_recompute(base_model, ft_model, base_path, ft_path, measured, source):
    """
    Recomputes each response on its own, with no padding or logits_to_keep, and the logits at t - 1 for token t.
    This catches swapped directions, wrong alignment and prompt tokens leaking into the totals.
    """
    _, rows = measured
    _, records = load_responses(base_path if source == "base" else ft_path)
    rows_by_prompt = {row["prompt_id"]: row for row in rows if row["source"] == source}

    for record in records[:10]:
        input_ids = torch.tensor([record["prompt_ids"] + record["response_ids"]])
        prompt_length = len(record["prompt_ids"])
        with torch.no_grad():
            log_base = base_model(input_ids).logits[0, prompt_length - 1:-1].log_softmax(dim=-1)
            log_ft = ft_model(input_ids).logits[0, prompt_length - 1:-1].log_softmax(dim=-1)

        # Forward is KL(base || ft), reverse is KL(ft || base)
        forward = (log_base.exp() * (log_base - log_ft)).sum().item()
        reverse = (log_ft.exp() * (log_ft - log_base)).sum().item()

        row = rows_by_prompt[record["prompt_id"]]
        assert row["n_tokens"] == len(record["response_ids"])
        assert row[f"kl_exact_fwd_{source}"] == pytest.approx(forward, abs=1e-4)
        assert row[f"kl_exact_rev_{source}"] == pytest.approx(reverse, abs=1e-4)

        if source == "ft":
            tokens = torch.tensor(record["response_ids"]).unsqueeze(1)
            log_r = log_base.gather(1, tokens) - log_ft.gather(1, tokens)
            assert row["kl_k1_rev_ft"] == pytest.approx((-log_r).sum().item(), abs=1e-4)
            assert row["kl_k2_rev_ft"] == pytest.approx((0.5 * log_r ** 2).sum().item(), abs=1e-4)
            assert row["kl_k3_rev_ft"] == pytest.approx((log_r.exp() - 1 - log_r).sum().item(), abs=1e-4)

def test_log_probs_are_always_fp32_even_with_a_bf16_model(base_model):
    """The drift test measures bf16 forward passes, not bf16 KL math."""
    bf16_model = copy.deepcopy(base_model).to(torch.bfloat16)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    keep, mask = response_positions(torch.tensor([2]), torch.tensor([4]))
    with torch.no_grad():
        log_probs = response_log_probs(bf16_model, input_ids, torch.ones_like(input_ids), keep, mask)
    assert log_probs.dtype == torch.float32

def test_padded_and_unpadded_batches_give_the_same_kl(base_model, ft_model, base_path, ft_path, measured):
    scalars, rows = measured
    single_scalars, single_rows = measure_kl(base_model, ft_model, base_path, ft_path, 1)

    for name, value in scalars.items():
        assert single_scalars[name] == pytest.approx(value, abs=1e-4)
    for row, single_row in zip(rows, single_rows):
        for name in EXACT_SEQUENCE_TOTALS + K_SEQUENCE_TOTALS:
            if name in row:
                assert single_row[name] == pytest.approx(row[name], abs=1e-3)

def test_bf16_models_are_refused_by_default(base_model, ft_model, base_path, ft_path):
    bf16_base = copy.deepcopy(base_model).to(torch.bfloat16)
    bf16_ft = copy.deepcopy(ft_model).to(torch.bfloat16)
    with pytest.raises(ValueError):
        measure_kl(bf16_base, bf16_ft, base_path, ft_path, 32)

def test_other_precisions_are_refused_even_for_the_drift_test(base_model, ft_model, base_path, ft_path):
    half_base = copy.deepcopy(base_model).to(torch.float16)
    half_ft = copy.deepcopy(ft_model).to(torch.float16)
    with pytest.raises(ValueError):
        measure_kl(half_base, half_ft, base_path, ft_path, 32, allow_bf16=True)

def test_responses_not_sampled_in_fp32_are_refused(base_model, ft_model, base_path, ft_path, tmp_path):
    def with_dtype(path, name):
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["header"]["dtype"] = "torch.bfloat16"
        new_path = tmp_path / name
        new_path.write_text("\n".join([json.dumps(header)] + lines[1:]) + "\n", encoding="utf-8")
        return new_path

    with pytest.raises(ValueError):
        measure_kl(base_model, ft_model, with_dtype(base_path, "base.jsonl"), with_dtype(ft_path, "ft.jsonl"), 32, allow_bf16=True)

def test_responses_with_different_prompts_are_refused(base_model, ft_model, base_path, ft_path, tmp_path):
    lines = ft_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["prompt_ids"][0] += 1
    lines[1] = json.dumps(record)
    changed_path = tmp_path / "changed.jsonl"
    changed_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        measure_kl(base_model, ft_model, base_path, changed_path, 32)

def test_save_merges_results_and_writes_the_sequence_file(measured, tmp_path):
    scalars, rows = measured
    results_path = tmp_path / "run" / "final.json"
    results_path.parent.mkdir()
    results_path.write_text(json.dumps({"run": "x"}), encoding="utf-8")

    save_kl_results(scalars, rows, results_path, torch.float32)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["run"] == "x"
    assert results["kl_dtype"] == "torch.float32"
    assert results["kl_exact_fwd_base_mean_token"] == scalars["kl_exact_fwd_base_mean_token"]
    lines = (tmp_path / "run" / "final_kl_seq.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(rows)

def test_save_refuses_bf16_values(measured, tmp_path):
    scalars, rows = measured
    results_path = tmp_path / "final.json"
    with pytest.raises(ValueError):
        save_kl_results(scalars, rows, results_path, torch.bfloat16)
    assert not results_path.exists()
