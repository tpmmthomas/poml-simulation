"""Regression specifications for constrained fitting and prompt-held-out scoring."""

from itertools import product

import numpy as np
import pytest

from poml_sim.runtime_weights import (
    group_folds,
    nonnegative_ridge,
    fit_runtime_schedule,
)
from poml_sim.backends import load_schedule
from pathlib import Path


def test_nonnegative_ridge_matches_exhaustive_active_sets():
    rng = np.random.default_rng(71)
    x = rng.random((20, 5))
    y = x @ np.array([2, -3, 5, 0, 1])
    penalty = 0.1
    actual = nonnegative_ridge(x, y, penalty)
    candidates = []
    for active in product((False, True), repeat=x.shape[1]):
        mask = np.array(active)
        coefficient = np.zeros(x.shape[1])
        coefficient[mask] = np.linalg.solve(
            x[:, mask].T @ x[:, mask] / len(y) + penalty * np.eye(sum(mask)),
            x[:, mask].T @ y / len(y),
        )
        if np.all(coefficient >= 0):
            candidates.append(coefficient)

    def loss(w):
        return np.mean((x @ w - y) ** 2) + penalty * (w @ w)

    expected = min(candidates, key=loss)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert np.any(actual == 0)


def test_group_folds_never_split_prompt_variants_or_replicates():
    groups = [f"prompt-{i}" for i in range(12) for _ in range(5 if i < 4 else 2)]
    folds = group_folds(groups)
    assert set(folds) == {0, 1, 2, 3}
    for group in set(groups):
        assert len({fold for key, fold in zip(groups, folds) if key == group}) == 1
    np.testing.assert_array_equal(folds, group_folds(groups))


@pytest.mark.parametrize("penalty", [0, -1, float("nan")])
def test_nonnegative_ridge_rejects_invalid_penalty(penalty):
    with pytest.raises(ValueError):
        nonnegative_ridge([[1]], [1], penalty)


def test_fit_uses_realised_lengths_and_learns_fixed_overhead():
    schedule = load_schedule(Path("src/poml_sim/data/gpt2_reference_schedule.json"))
    records = [
        dict(
            query_id=f"p{i}-cap{k}",
            prompt_sha256=f"p{i}",
            replicate=0,
            verified=True,
            fresh_inference=True,
            fresh_proof=True,
            prompt_length=8 + 8 * (i % 4),
            output_length=k,
            # Request caps are deliberately irrelevant to realised cost.
            max_output_length=32,
            inference_proof_seconds=60 + 0.6 * (8 + 8 * (i % 4) + k),
        )
        for i in range(12)
        for k in (8, 16, 32)
    ]
    fit, predictions = fit_runtime_schedule(records, schedule)
    assert fit["held_out_weighted"]["mape_percent"] < 1
    assert fit["held_out_weighted"]["mape_percent"] < fit["held_out_uniform"]["mape_percent"]
    assert min(fit["weights"].values()) >= 0
    assert fit["scale"]["scaled_median"] == 10000
    assert fit["max_fixed_point_prediction_error_seconds"] < 0.001
    assert {row["output_length"] for row in predictions} == {8, 16, 32}
    records[0]["fresh_proof"] = False
    with pytest.raises(ValueError, match="genuine"):
        fit_runtime_schedule(records, schedule)
