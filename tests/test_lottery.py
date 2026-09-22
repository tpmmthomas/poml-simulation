"""Scaled ticket counts and literal SHA-256 checks against independent digests."""

import hashlib
from fractions import Fraction

import pytest

from poml_sim.lottery import (
    LIMIT,
    calibrate_scale,
    scaled_complexity,
    evaluate_tickets,
    ticket_hash,
)


def test_calibration_scales_all_unit_weights_identically():
    counts = [10_000_000_000, 20_000_000_000, 30_000_000_000]
    scale = calibrate_scale(counts)
    assert scale["scaled_range"] == [5000, 15000]
    assert scale["scaled_median"] == 10000
    for raw in [1_000_000_003, *counts]:
        exact = Fraction(raw * scale["numerator"], scale["denominator"])
        assert abs(Fraction(scaled_complexity(raw, scale)) - exact) <= Fraction(1, 2)


def test_rounds_total_once_with_half_ties_upward():
    scale = {"numerator": 1, "denominator": 10}
    assert scaled_complexity(25, scale) == 3
    assert scaled_complexity(24, scale) == 2
    assert scaled_complexity(1, scale) == 1


@pytest.mark.parametrize("size", [0, 47, 48, 55, 56, 63, 64, 10000])
def test_evaluates_every_real_ticket_and_retains_last_success(size):
    prefix = b"x" * size
    difficulty = LIMIT // 3
    winners = [
        j
        for j in range(1, 501)
        if int.from_bytes(hashlib.sha256(prefix + j.to_bytes(8, "big")).digest(), "big")
        < difficulty
    ]
    row = evaluate_tickets(prefix, 500, difficulty)
    assert row["tickets_evaluated"] == 500
    assert row["winning_ticket"] == winners[-1]
    assert row["winning_hash"] == ticket_hash(prefix, winners[-1]).hex()


def test_threshold_extremes():
    assert evaluate_tickets(b"prefix", 100, 1)["winning_ticket"] is None
    assert evaluate_tickets(b"prefix", 100, LIMIT)["winning_ticket"] == 100
    with pytest.raises(ValueError):
        evaluate_tickets(b"", 0, LIMIT)
