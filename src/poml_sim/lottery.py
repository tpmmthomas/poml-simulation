"""Scaled appendix complexity and literal SHA-256 virtual-ticket evaluation."""

import hashlib
from fractions import Fraction
import statistics
import time

LIMIT = 1 << 256


def calibrate_scale(raw_counts: list[int], target: int = 10_000) -> dict:
    """Freeze one common multiplicative weight scale at the calibration median."""
    if not raw_counts or min(raw_counts) < 1 or target < 1:
        raise ValueError("positive calibration counts and target required")
    ordered = sorted(raw_counts)
    n = len(ordered)
    median = (
        Fraction(ordered[n // 2])
        if n % 2
        else Fraction(ordered[n // 2 - 1] + ordered[n // 2], 2)
    )
    scale = Fraction(target, 1) / median
    result = {
        "numerator": scale.numerator,
        "denominator": scale.denominator,
        "target_median": target,
        "raw_median": float(median),
        "rounding": "nearest integer; half ties upward; minimum one",
    }
    scaled = [scaled_complexity(c, result) for c in raw_counts]
    result.update(
        raw_range=[min(raw_counts), max(raw_counts)],
        scaled_range=[min(scaled), max(scaled)],
        scaled_median=statistics.median(scaled),
        max_absolute_rounding_error=float(
            max(abs(Fraction(c) - raw * scale) for raw, c in zip(raw_counts, scaled))
        ),
        max_relative_rounding_error=float(
            max(
                abs(Fraction(c) / (raw * scale) - 1)
                for raw, c in zip(raw_counts, scaled)
            )
        ),
    )
    return result


def scaled_complexity(raw: int, scale: dict) -> int:
    """Apply the same rational scale to every unit weight, then round the total."""
    numerator, denominator = scale["numerator"], scale["denominator"]
    if raw < 1 or numerator < 1 or denominator < 1:
        raise ValueError("complexity and scale must be positive")
    return max(1, (2 * raw * numerator + denominator) // (2 * denominator))


def ticket_hash(prefix: bytes, ticket: int) -> bytes:
    """Hash G(s,tx) || canonical ciphertext sequence || uint64_be(j)."""
    if not 1 <= ticket < 1 << 64:
        raise ValueError("ticket index must fit a positive uint64")
    return hashlib.sha256(prefix + ticket.to_bytes(8, "big")).digest()


def evaluate_tickets(prefix: bytes, count: int, difficulty: int) -> dict:
    """Actually hash every j=1..C; retain the largest success, as Algorithm 1 does."""
    if not 1 <= count < 1 << 64 or not 0 < difficulty <= LIMIT:
        raise ValueError("invalid complexity or difficulty")
    start = time.perf_counter()
    common = hashlib.sha256(prefix)
    winner, winning_hash = None, None
    # Copying SHA's shared-prefix state preserves the exact message digest.
    # A typical 10,000-ticket attempt needs no probability approximation.
    for j in range(1, count + 1):
        state = common.copy()
        state.update(j.to_bytes(8, "big"))
        digest = state.digest()
        if int.from_bytes(digest, "big") < difficulty:
            winner, winning_hash = j, digest.hex()
    return {
        "tickets_evaluated": count,
        "winning_ticket": winner,
        "winning_hash": winning_hash,
        "hash_seconds": time.perf_counter() - start,
        "prefix_sha256": common.hexdigest(),
    }
