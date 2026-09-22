"""Fit deterministic nonnegative reference-operation weights from archived times.

Weights predict inference-plus-proof time from realised N,K only. Prompt-grouped
nested validation keeps repeated measurements and capped variants together.
"""

from collections import Counter
import random

import numpy as np

from .gpt2_work import reference_counts, weighted_cost
from .lottery import calibrate_scale


RIDGES = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0)
FIXED_POINT = 10**15


def nonnegative_ridge(x, y, penalty):
    """Solve mean squared error plus ridge penalty with nonnegative coefficients.

    The active-set quadratic solve uses only NumPy. Positive ridge makes the
    subproblems nonsingular even when operation-count columns are identical.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(penalty)
        or penalty <= 0
    ):
        raise ValueError("finite aligned data and a positive ridge penalty required")
    h = x.T @ x / len(x) + penalty * np.eye(x.shape[1])
    b = x.T @ y / len(x)
    weights = np.zeros(x.shape[1])
    active = np.zeros(x.shape[1], dtype=bool)
    tolerance = 1e-10 * max(1.0, np.max(np.abs(b)))
    for _ in range(20 * x.shape[1] + 1):
        gradient = b - h @ weights
        if np.all(gradient[~active] <= tolerance):
            return weights
        active[np.argmax(np.where(active, -np.inf, gradient))] = True
        for _ in range(20 * x.shape[1] + 1):
            proposal = np.zeros_like(weights)
            proposal[active] = np.linalg.solve(h[np.ix_(active, active)], b[active])
            if np.all(proposal[active] > 0):
                weights = proposal
                break
            bad = active & (proposal <= 0)
            step = np.min(weights[bad] / (weights[bad] - proposal[bad]))
            weights += step * (proposal - weights)
            remove = active & (weights <= 1e-12)
            weights[remove] = 0
            active[remove] = False
        else:
            raise RuntimeError("nonnegative ridge inner solve failed to converge")
    raise RuntimeError("nonnegative ridge failed to converge")


def group_folds(groups, folds=4, seed=20260916):
    """Partition whole prompts, balancing measurement counts without using times."""
    counts = Counter(groups)
    if len(counts) < folds or folds < 2:
        raise ValueError("at least two folds with at least one prompt each required")
    ordered = sorted(counts)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group: counts[group], reverse=True)
    loads = [0] * folds
    assignment = {}
    for group in ordered:
        fold = min(range(folds), key=lambda i: loads[i])
        assignment[group] = fold
        loads[fold] += counts[group]
    return np.array([assignment[group] for group in groups])


def fit_weights(x, y, penalty, scale):
    """Return seconds per operation using fixed, public structural normalization."""
    return nonnegative_ridge(x / scale, y, penalty) / scale


def prediction_metrics(actual, predicted):
    """Report errors in seconds and relative prediction/observed-time dispersion."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ratio = predicted / actual
    return {
        "mae_seconds": float(np.mean(np.abs(actual - predicted))),
        "rmse_seconds": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape_percent": float(100 * np.mean(np.abs(ratio - 1))),
        "r_squared": float(
            1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        )
        if np.ptp(actual)
        else None,
        "prediction_per_second_cv": float(np.std(ratio) / np.mean(ratio)),
    }


def select_penalty(x, y, groups, seed, scale):
    """Select ridge strength by grouped out-of-fold mean squared error."""
    folds = group_folds(groups, min(4, len(set(groups))), seed)
    candidates = []
    for penalty in RIDGES:
        predictions = np.zeros(len(y))
        for fold in sorted(set(folds)):
            train, test = folds != fold, folds == fold
            predictions[test] = x[test] @ fit_weights(x[train], y[train], penalty, scale)
        candidates.append({"penalty": penalty, **prediction_metrics(y, predictions)})
    best = min(candidates, key=lambda row: row["rmse_seconds"])
    return best["penalty"], candidates


def fit_runtime_schedule(records, schedule, *, seed=20260916, ticket_target=10000):
    """Fit all archived pairs and estimate generalization with nested prompt folds.

    The final frozen model uses all rows; its subsequent replay is consequently
    an in-bank sensitivity experiment, distinct from the held-out error report.
    """
    if len(records) < 8:
        raise ValueError("at least eight measurements required")
    vectors, times, groups = [], [], []
    for row in records:
        if not all(row.get(key) is True for key in ("verified", "fresh_inference", "fresh_proof")):
            raise ValueError("fitting requires verified genuine measurements")
        duration = row["inference_proof_seconds"]
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("recorded inference-plus-proof time must be positive")
        vectors.append(
            reference_counts(row["prompt_length"], row["output_length"], schedule)["combined"]
        )
        times.append(duration)
        groups.append(row["prompt_sha256"])
    names = sorted(set().union(*(v.keys() for v in vectors)))
    x = np.array([[v.get(name, 0) for name in names] for v in vectors], dtype=float)
    # A rare MSM-size count can be constant in a training fold but jump 196x
    # in a held-out shape. Scaling by that fold's mean treats its coefficient
    # as fixed overhead and badly extrapolates. Bounds from the entire public
    # supported shape domain regularize such coefficients without using times
    # or measurement-bank statistics from any validation fold.
    domain = [
        reference_counts(n, k, schedule)["combined"]
        for n in range(2, schedule["setup_max"])
        for k in range(1, schedule["setup_max"] - n + 1)
    ]
    normalization = np.array([max(v.get(name, 0) for v in domain) for name in names], dtype=float)
    y = np.array(times)
    if len(set(groups)) < 4:
        raise ValueError("at least four distinct prompts required for nested validation")
    folds = group_folds(groups, seed=seed)
    held_out, baseline = np.zeros(len(y)), np.zeros(len(y))
    fold_details = []
    raw = x.sum(axis=1)
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        train_groups = [g for g, keep in zip(groups, train) if keep]
        penalty, candidates = select_penalty(
            x[train], y[train], train_groups, seed + int(fold) + 1, normalization
        )
        held_out[test] = x[test] @ fit_weights(x[train], y[train], penalty, normalization)
        # A fitted global multiplier makes the uniform-weight comparison fair.
        slope = float(raw[train] @ y[train] / (raw[train] @ raw[train]))
        baseline[test] = raw[test] * slope
        fold_details.append({"fold": int(fold), "penalty": penalty, "inner_candidates": candidates})
    penalty, candidates = select_penalty(x, y, groups, seed, normalization)
    coefficients = fit_weights(x, y, penalty, normalization)
    weights = {
        name: int(round(float(value) * FIXED_POINT)) for name, value in zip(names, coefficients)
    }
    costs = [weighted_cost(v, weights) for v in vectors]
    if min(costs) <= 0:
        raise ValueError("fitted weights assign no work to a recorded shape")
    final_prediction = np.array(costs, dtype=float) / FIXED_POINT
    fit = {
        "schema": "poml-runtime-weights-1",
        "schedule_sha256": schedule["sha256"],
        "seed": seed,
        "weights": weights,
        "fixed_point_units_per_second": FIXED_POINT,
        "scale": calibrate_scale(costs, ticket_target),
        "records": len(records),
        "distinct_prompts": len(set(groups)),
        "distinct_shapes": len(
            set(
                zip(
                    [r["prompt_length"] for r in records],
                    [r["output_length"] for r in records],
                )
            )
        ),
        "operation_columns": len(names),
        "normalized_feature_rank": int(np.linalg.matrix_rank(x / np.maximum(x.mean(axis=0), 1))),
        "penalty": penalty,
        "candidate_validation": candidates,
        "nested_folds": fold_details,
        "normalization": dict(zip(names, map(int, normalization))),
        "held_out_uniform": prediction_metrics(y, baseline),
        "held_out_weighted": prediction_metrics(y, held_out),
        "training_weighted": prediction_metrics(y, final_prediction),
        "max_fixed_point_prediction_error_seconds": float(
            np.max(np.abs(final_prediction - x @ coefficients))
        ),
        "method": "nonnegative ridge; fixed column maxima over public supported N,K domain; no extra intercept; 4 outer prompt folds and up to 4 inner prompt folds",
        "scope": "Final weights fit all existing pairs. Replay uses the same bank; held-out prediction errors use nested prompt-grouped validation. Collinear weights are not individually identified physical costs.",
    }
    predictions = [
        {
            "query_id": row["query_id"],
            "replicate": row["replicate"],
            "prompt_sha256": groups[i],
            "prompt_length": row["prompt_length"],
            "output_length": row["output_length"],
            "outer_fold": int(folds[i]),
            "actual_seconds": float(y[i]),
            "uniform_held_out_seconds": float(baseline[i]),
            "weighted_held_out_seconds": float(held_out[i]),
            "weighted_fitted_seconds": float(final_prediction[i]),
        }
        for i, row in enumerate(records)
    ]
    return fit, predictions
