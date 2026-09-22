"""Independent exact arithmetic checks for the current paper lottery."""

from fractions import Fraction
import pytest
from poml_sim.crypto import canonical, sha256
from poml_sim.lottery import LIMIT, complexity_threshold, evaluate_lottery


@pytest.mark.parametrize("difficulty", [1, 2**64, LIMIT // 100, LIMIT // 2, LIMIT - 1, LIMIT])
@pytest.mark.parametrize("complexity", [1, 2, 5, 9, 33, 100])
def test_threshold_matches_exact_rational_probability(difficulty, complexity):
    exact = LIMIT * (1 - (1 - Fraction(difficulty, LIMIT)) ** complexity)
    assert complexity_threshold(difficulty, complexity) == exact.numerator // exact.denominator


def test_tiny_probability_does_not_disappear_by_float_cancellation():
    assert complexity_threshold(1, 10**12) == 10**12 - 1


def test_lottery_hashes_whole_prefix_once_with_last_pair_complexity():
    base, prefix = b"b" * 32, (b"one", b"two")
    digest, won = evaluate_lottery(base, prefix, LIMIT, 1)
    assert digest == sha256(base, canonical(prefix)) and won
    assert digest != evaluate_lottery(base, (b"two",), LIMIT, 1)[0]


@pytest.mark.parametrize("difficulty,complexity", [(0, 1), (LIMIT + 1, 1), (1, 0), (1, 1.5)])
def test_invalid_lottery_inputs_are_rejected(difficulty, complexity):
    with pytest.raises(ValueError):
        complexity_threshold(difficulty, complexity)
