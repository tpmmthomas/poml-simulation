"""Statistical experiment specifications for first-completion and exhaustion."""

import random
import pytest
from poml_sim.experiments import LazyPermutation, calibrate_difficulty, simulate_race
from poml_sim.lottery import LIMIT


def test_permutations_are_complete_without_replacement():
    order = LazyPermutation(100, random.Random(4))
    assert set(order.draw() for _ in range(100)) == set(range(100))
    assert order.draw() is None


def test_exhausted_race_keeps_all_completed_duplicates():
    result = simulate_race([(1, 1)], miners=3, queries=2, difficulty=1, seed=42)
    assert not result["adopted"]
    assert result["termination_reason"] == "query_pool_exhausted"
    assert result["completed_pairs"] == 6
    assert result["first_completed_complexity"] == 2
    assert result["wasted_work_fraction"] == 4 / 6


def test_one_winning_completion_cancels_tied_events():
    result = simulate_race([(1, 1)], miners=10, queries=10, difficulty=LIMIT, seed=1)
    assert result["completed_pairs"] == 1 and result["canceled_attempts"] == 9
    assert result["wasted_work_fraction"] == 0


def test_complexity_weights_waste_instead_of_counting_pairs():
    result = simulate_race([(1, 1), (2, 100)], miners=5, queries=1, difficulty=1, seed=7)
    assert result["completed_pairs"] == 5
    assert (
        result["wasted_work_fraction"]
        == (result["completed_complexity"] - result["first_completed_complexity"])
        / result["completed_complexity"]
    )
    assert result["wasted_work_fraction"] != 4 / 5


def test_calibration_uses_ratio_of_sums_and_scales_by_miners():
    a = calibrate_difficulty([(1, 10), (9, 10)], 1, 300)
    b = calibrate_difficulty([(1, 10), (9, 10)], 2, 300)
    assert abs(a - 2 * b) <= 1
    assert abs(a / LIMIT - 1 / 600) < 1e-15


def test_races_are_reproducible_and_replenishment_preserves_elapsed_time():
    kwargs = dict(
        samples=[(1, 1)], miners=1, queries=1, difficulty=LIMIT // 10, seed=4, replenish=True
    )
    first = simulate_race(**kwargs)
    assert first == simulate_race(**kwargs)
    assert first["adopted"] and first["block_time"] > 1
    assert first["completed_pairs"] == first["block_time"]


@pytest.mark.parametrize("samples", [[], [(0, 1)], [(float("nan"), 1)], [(1, 0)], [(1, 2.2)]])
def test_invalid_measurements_are_rejected(samples):
    with pytest.raises(ValueError):
        calibrate_difficulty(samples, 1, 300)
