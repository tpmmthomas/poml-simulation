"""Measured-service-time experiments for liveness and duplicate wasted work.

These event experiments resample complete (duration, complexity) pairs. They do
not claim that a replayed pair is a fresh proof for a virtual miner's challenge.
"""

from decimal import Decimal, localcontext
import heapq
import json
import math
import random
import statistics
from .lottery import LIMIT, complexity_threshold


class LazyPermutation:
    """Draw an unbiased permutation prefix without allocating M times Q cells."""

    def __init__(self, size, rng):
        if size < 1:
            raise ValueError("permutation size must be positive")
        self.size, self.rng, self.position, self.swaps = size, rng, 0, {}

    def draw(self):
        """Return a new query index or None once all queries were consumed."""
        if self.position == self.size:
            return None
        index = self.position
        other = self.rng.randrange(index, self.size)
        value = self.swaps.get(other, other)
        self.swaps[other] = self.swaps.get(index, index)
        self.swaps.pop(index, None)
        self.position += 1
        return value


def load_measurements(path):
    """Require positive, verified, fresh measurements with explicit provenance."""
    data = json.loads(path.read_text())
    if data.get("schema") != "poml-measurements-1" or not data.get("records"):
        raise ValueError("expected a nonempty poml-measurements-1 bank")
    rows = data["records"]
    for row in rows:
        if any(
            row.get(field) is not True for field in ("verified", "fresh_inference", "fresh_proof")
        ):
            raise ValueError("measurement bank must contain verified fresh model proofs")
        if not row.get("proof_sha256"):
            raise ValueError("measurement lacks proof provenance")
    samples = [(r["duration"], r["complexity"]) for r in rows]
    validate_samples(samples)
    return data, samples


def validate_samples(samples):
    """Reject invalid work/time values before calibration or event scheduling."""
    if not samples or any(
        not math.isfinite(t) or t <= 0 or type(c) is not int or c < 1 for t, c in samples
    ):
        raise ValueError("samples need finite positive times and integer complexities")


def calibrate_difficulty(samples, miners, target):
    """Calibrate D using E[T]/(M target E[C]); report achieved intervals separately."""
    validate_samples(samples)
    if miners < 1 or not math.isfinite(target) or target <= 0:
        raise ValueError("positive miners and finite target required")
    with localcontext() as context:
        context.prec = 100
        probability = sum(Decimal(str(t)) for t, _ in samples) / (
            Decimal(sum(c for _, c in samples)) * miners * Decimal(str(target))
        )
        difficulty = int(probability * LIMIT)
    if not 0 < difficulty < LIMIT:
        raise ValueError("target is outside the supported calibration range")
    return difficulty


def simulate_race(
    samples, *, miners, queries, difficulty, seed, replenish=False, max_events=10_000_000
):
    """Count only completed duplicate pairs through the first winning completion.

    Ties are randomized reproducibly; events after the chosen winner are not
    completed work. Each miner consumes a uniform permutation without replacement.
    The wasted-work definition credits the *first completion*, regardless of
    which miner later wins; this differs from actual on-chain fee settlement.
    """
    validate_samples(samples)
    if miners < 1 or queries < 1 or max_events < 1:
        raise ValueError("positive miner, query and event limits required")
    rng, lottery_rng = random.Random(seed), random.Random(f"lottery:{seed}")
    orders = [
        LazyPermutation(queries, random.Random(f"selection:{seed}:{m}")) for m in range(miners)
    ]
    generation = [0] * miners
    thresholds = {c: complexity_threshold(difficulty, c) for _, c in samples}
    queue, sequence = [], 0

    def start(miner, time):
        nonlocal sequence
        query = orders[miner].draw()
        if query is None and replenish:
            generation[miner] += 1
            orders[miner] = LazyPermutation(
                queries, random.Random(f"selection:{seed}:{miner}:{generation[miner]}")
            )
            query = orders[miner].draw()
        if query is None:
            return
        duration, complexity = rng.choice(samples)
        sequence += 1
        heapq.heappush(
            queue,
            (
                time + duration,
                rng.getrandbits(64),
                sequence,
                miner,
                (generation[miner], query),
                complexity,
                duration,
            ),
        )

    for miner in range(miners):
        start(miner, 0)
    seen, first, total, count, finish, winner = set(), 0, 0, 0, 0, None
    while queue and count < max_events:
        finish, _, _, miner, query, complexity, duration = heapq.heappop(queue)
        count += 1
        total += complexity
        if query not in seen:
            first += complexity
            seen.add(query)
        # Equivalent uniform threshold trial for the statistical experiment.
        # The full protocol simulator computes the actual ciphertext hash.
        if lottery_rng.getrandbits(256) < thresholds[complexity]:
            winner = miner
            break
        start(miner, finish)
    return {
        "adopted": winner is not None,
        "termination_reason": "adopted"
        if winner is not None
        else "event_limit"
        if queue
        else "query_pool_exhausted",
        "block_time": finish,
        "winner": winner,
        "completed_pairs": count,
        "unique_queries": len(seen),
        "completed_complexity": total,
        "first_completed_complexity": first,
        "duplicate_complexity": total - first,
        "wasted_work_fraction": (total - first) / total,
        "canceled_attempts": len(queue),
    }


def summarize(values):
    """Report sample size, mean and sample standard deviation without hiding zeros."""
    values = list(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }
