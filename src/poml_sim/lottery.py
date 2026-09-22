"""The paper's single SHA-256 lottery with an integer complexity threshold."""

from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from functools import lru_cache
from fractions import Fraction
from statistics import median

from .crypto import canonical, sha256

LIMIT = 1 << 256


@lru_cache(maxsize=8192)
def complexity_threshold(difficulty: int, complexity: int) -> int:
    """Compute floor(2**256 * (1 - (1 - D/2**256)**C)).

    Directed decimal bounds avoid float cancellation at tiny per-unit D and
    large C. Precision is increased until both bounds give the same integer.
    """
    if type(difficulty) is not int or not 1 <= difficulty <= LIMIT:
        raise ValueError("difficulty must be an integer in [1, 2**256]")
    if type(complexity) is not int or complexity < 1:
        raise ValueError("complexity must be a positive integer")
    if complexity == 1 or difficulty == LIMIT:
        return difficulty
    if complexity <= 8:
        # Small cases also give an exact independent check of the decimal path.
        return LIMIT - (
            (LIMIT - difficulty) ** complexity + LIMIT ** (complexity - 1) - 1
        ) // LIMIT ** (complexity - 1)
    for precision in (100, 200, 400, 800, 1600):
        bounds = []
        for rounding in (ROUND_CEILING, ROUND_FLOOR):
            with localcontext() as context:
                context.prec = precision
                context.rounding = rounding
                survival = (Decimal(LIMIT - difficulty) / Decimal(LIMIT)) ** complexity
                # Subtract integers only: subtracting a tiny survival probability
                # from decimal 1 would incorrectly round a near-certain win to 1.
                remaining = Decimal(LIMIT) * survival
                ceil_remaining = max(1, int(remaining.to_integral_value(rounding=ROUND_CEILING)))
                bounds.append(LIMIT - ceil_remaining)
        if bounds[0] == bounds[1]:
            return bounds[0]
    raise ArithmeticError("could not resolve the integer lottery threshold")


def evaluate_lottery(
    binding: bytes, ciphertexts: tuple[bytes, ...], difficulty: int, complexity: int
) -> tuple[bytes, bool]:
    """Hash G(s,tx) and the whole ciphertext prefix exactly once."""
    value = sha256(binding, canonical(ciphertexts))
    return value, int.from_bytes(value, "big") < complexity_threshold(difficulty, complexity)


def calibrate_scale(counts, target=10000):
    """Return a frozen rational rescaling for fitted integer operation weights."""
    if not counts or min(counts) <= 0 or target < 1:
        raise ValueError("positive counts and target required")
    scale = Fraction(target, 1) / Fraction(median(counts))
    result = {"numerator": scale.numerator, "denominator": scale.denominator}
    scaled = [scaled_complexity(value, result) for value in counts]
    return {**result, "scaled_median": median(scaled), "scaled_range": [min(scaled), max(scaled)]}


def scaled_complexity(raw: int, scale: dict) -> int:
    """Round one total, with ties upward, under a public rational unit scale."""
    numerator, denominator = scale["numerator"], scale["denominator"]
    if min(raw, numerator, denominator) < 1:
        raise ValueError("positive complexity and scale required")
    return max(1, (2 * raw * numerator + denominator) // (2 * denominator))
