"""Event-driven simulation primitives for the revised LLM PoML evaluation.

The simulator models the protocol's public complexity function directly.  A
query has a prompt length and a stochastic output-length trace; a completed
inference--proof pair contributes one virtual lottery attempt per complexity
unit.  It intentionally does not model cryptographic verification: those
properties are assumptions of the reduction and are tested separately in the
paper appendix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .gpt2_work import reference_counts, weighted_cost


@dataclass(frozen=True)
class QuerySpec:
    """Public query metadata; the realised output length is not public."""

    query_id: str
    prompt_length: int
    max_output_length: int
    category: str = "synthetic"
    fee_limit: float = 0.0
    profile_mean_output_length: float | None = None
    profile_trials: int = 0
    trace_query_id: str | None = None


@dataclass(frozen=True)
class QueryTrace:
    """One private challenge outcome for a query attempt."""

    output_length: int
    complexity: int
    duration: float


class LazyPermutation:
    """Draw a random permutation prefix without allocating all pool entries."""

    def __init__(self, size: int, rng: random.Random) -> None:
        if size < 0:
            raise ValueError("permutation size cannot be negative")
        self.size = size
        self.rng = rng
        self.position = 0
        self.swaps: dict[int, int] = {}

    def draw(self) -> int | None:
        """Return the next unused index, or None after pool exhaustion."""
        if self.position >= self.size:
            return None
        index = self.position
        swap_index = self.rng.randrange(index, self.size)
        index_value = self.swaps.get(index, index)
        swap_value = self.swaps.get(swap_index, swap_index)
        self.swaps[swap_index] = index_value
        self.swaps.pop(index, None)
        self.position += 1
        return swap_value


@dataclass
class Attempt:
    """Completed work produced by one miner during one race."""

    miner_id: int
    query_id: str
    trace: QueryTrace
    completion_time: float
    won: bool = False


@dataclass
class RaceResult:
    """All metrics needed to analyse one block race."""

    adopted: bool
    winner: int | None
    block_time: float
    attempts: list[Attempt]
    winning_complexity: int
    total_completed_complexity: int
    settled_complexity: int = 0
    discarded_complexity: int = 0
    settled_queries: int = 0
    unfinished_complexity: int = 0
    termination_reason: str = "adopted"

    def as_rows(self, race_id: int) -> list[dict]:
        rows = []
        for attempt in self.attempts:
            row = asdict(attempt)
            row["race_id"] = race_id
            row["trace"] = asdict(attempt.trace)
            rows.append(row)
        return rows


@dataclass(frozen=True)
class SimulationConfig:
    """Explicit manifest for a reproducible simulation campaign."""

    seed: int = 20260915
    context_limit: int = 64
    difficulty_probability: float = 1e-4
    miner_count: int = 16
    miner_work_rates: tuple[float, ...] = ()
    target_block_time: float = 300.0
    response_inclusion_probability: float = 1.0
    network_delay: float = 0.0


def load_schedule(path: str | Path) -> dict:
    """Load and validate an audited reference-gas schedule."""

    schedule = json.loads(Path(path).read_text())
    # reference_counts performs the full digest and version validation.
    reference_counts(2, 1, schedule)
    return schedule


def complexity_for(
    query: QuerySpec,
    output_length: int,
    schedule: Mapping,
    weights: Mapping[str, int] | None = None,
) -> int:
    """Return the public complexity for one realised ``(N, K)`` pair."""

    if output_length < 1 or output_length > query.max_output_length:
        raise ValueError("output length is outside the query's allowed range")
    counts = reference_counts(query.prompt_length, output_length, dict(schedule))
    if weights is None:
        # A deterministic integer work unit when no public gas weights have
        # been selected yet: total logical operation count.
        return int(sum(counts["combined"].values()))
    return int(weighted_cost(counts["combined"], weights))


def make_query_pool(
    count: int,
    *,
    seed: int,
    schedule: Mapping,
    max_output_length: int = 16,
    prompt_lengths: Sequence[int] = (2, 4, 8, 16, 24, 32),
    fee_rate: float = 1.0,
) -> list[QuerySpec]:
    """Create synthetic fixtures for explicit smoke tests, without fake profiles."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    usable = [n for n in prompt_lengths if 2 <= n < schedule["setup_max"]]
    if not usable:
        raise ValueError("prompt_lengths contains no supported lengths")
    pool: list[QuerySpec] = []
    upper_by_length: dict[int, int] = {}
    for index in range(count):
        n = usable[index % len(usable)] if index < len(usable) else rng.choice(usable)
        # Set the fee high enough for the configured maximum length.  This is
        # the demand-sufficient primary experiment; fee rejection is separate.
        # The fee bound depends on shape, not identifier. Cache the few shapes
        # so constructing a large simulated pool does not repeat gas audits.
        if n not in upper_by_length:
            upper_by_length[n] = complexity_for(
                QuerySpec(f"tmp-{index}", n, max_output_length),
                max_output_length,
                schedule,
            )
        upper = upper_by_length[n]
        pool.append(
            QuerySpec(
                query_id=f"q-{index:06d}",
                prompt_length=n,
                max_output_length=max_output_length,
                fee_limit=fee_rate * upper,
            )
        )
    return pool


def sample_trace(
    query: QuerySpec,
    *,
    rng: random.Random,
    schedule: Mapping,
    work_rate: float,
    output_shape: str = "categorical",
    weights: Mapping[str, int] | None = None,
    trace_bank: Mapping[str, Sequence[QueryTrace]] | None = None,
    trace_duration: bool = False,
    require_trace_bank: bool = False,
    trace_distribution: Sequence[QueryTrace] | None = None,
) -> QueryTrace:
    """Sample a challenge-specific ``K`` and convert it to work and time."""

    if work_rate <= 0:
        raise ValueError("work_rate must be positive")
    if trace_distribution is not None:
        if not trace_distribution:
            raise ValueError("trace distribution is empty")
        template = trace_distribution[rng.randrange(len(trace_distribution))]
        complexity = complexity_for(query, template.output_length, schedule, weights)
        duration = template.duration if trace_duration else complexity / work_rate
        return QueryTrace(template.output_length, complexity, duration)

    trace_choices = None
    if trace_bank is not None:
        trace_choices = trace_bank.get(query.trace_query_id or query.query_id)
        if trace_choices is None:
            trace_choices = trace_bank.get(f"shape:{query.prompt_length}")
    if trace_choices is not None:
        choices = tuple(trace_choices)
        if not choices:
            raise ValueError(f"trace bank has no traces for {query.query_id}")
        chosen = choices[rng.randrange(len(choices))]
        duration = chosen.duration if trace_duration else chosen.complexity / work_rate
        return QueryTrace(chosen.output_length, chosen.complexity, duration)
    if require_trace_bank:
        raise ValueError(f"missing real trace for query {query.query_id}")
    if output_shape == "categorical":
        # Paper campaigns require benchmark traces. This is only a fixture.
        k = rng.randint(1, query.max_output_length)
    else:
        raise ValueError(f"unknown output_shape: {output_shape}")
    complexity = complexity_for(query, k, schedule, weights)
    # A deterministic work-rate model is appropriate for the discrete-event
    # simulation; optional measured residuals can be applied by callers.
    duration = complexity / work_rate
    return QueryTrace(output_length=k, complexity=complexity, duration=duration)


def _select_query(
    pool: Sequence[QuerySpec],
    used: set[str],
    rng: random.Random,
    strategy: str,
    permutation: LazyPermutation | None = None,
) -> QuerySpec | None:
    if permutation is not None:
        index = permutation.draw()
        return None if index is None else pool[index]
    available = [query for query in pool if query.query_id not in used]
    if not available:
        return None
    if strategy == "uniform":
        return rng.choice(available)
    if strategy in {"profiled-short", "profiled-long"}:
        if any(
            q.profile_mean_output_length is None or q.profile_trials < 2
            for q in available
        ):
            raise ValueError(
                "profiled selection requires independent measured profiles"
            )
        direction = 1 if strategy == "profiled-short" else -1
        # Seeded tie-breaking avoids converting equal K predictions into a
        # hidden prompt-length policy. No held-out trace is inspected here.
        best = min(direction * q.profile_mean_output_length for q in available)
        return rng.choice(
            [q for q in available if direction * q.profile_mean_output_length == best]
        )
    if strategy == "shortest-prompt":
        return min(available, key=lambda q: (q.prompt_length, q.query_id))
    raise ValueError(f"unknown selection strategy: {strategy}")


def simulate_race(
    pool: Sequence[QuerySpec],
    *,
    miner_count: int,
    work_rates: Sequence[float] | None = None,
    difficulty_probability: float,
    seed: int,
    schedule: Mapping,
    strategies: Sequence[str] | None = None,
    weights: Mapping[str, int] | None = None,
    response_inclusion_probability: float = 0.0,
    trace_bank: Mapping[str, Sequence[QueryTrace]] | None = None,
    trace_duration: bool = False,
    require_trace_bank: bool = False,
    trace_distribution: Sequence[QueryTrace] | None = None,
    max_events: int | None = 100_000,
    replenish_pool: bool = False,
    literal_lottery: bool = False,
) -> RaceResult:
    """Simulate a block interval, optionally replenishing exhausted demand.

    Replenishment publishes another generation of benchmark-backed requests;
    miners independently finish their current generation before advancing.
    """

    if not pool:
        raise ValueError("pool must not be empty")
    if miner_count < 1:
        raise ValueError("miner_count must be positive")
    if not 0 < difficulty_probability < 1:
        raise ValueError("difficulty_probability must be in (0, 1)")
    if not 0 <= response_inclusion_probability <= 1:
        raise ValueError("response_inclusion_probability must be in [0, 1]")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive or None")
    rates = tuple(work_rates or (1.0,) * miner_count)
    if len(rates) != miner_count or any(rate <= 0 for rate in rates):
        raise ValueError("work_rates must contain one positive rate per miner")
    policies = tuple(strategies or ("uniform",) * miner_count)
    if len(policies) != miner_count:
        raise ValueError("strategies must contain one policy per miner")
    rng = random.Random(seed)
    lazy_permutations = (
        [
            LazyPermutation(len(pool), random.Random(rng.randrange(1 << 63)))
            for _ in range(miner_count)
        ]
        if all(policy == "uniform" for policy in policies)
        else [None] * miner_count
    )
    queue: list[tuple[float, int, int, QuerySpec, QueryTrace]] = []
    used: list[set[str]] = [set() for _ in range(miner_count)]
    generations = [0] * miner_count
    serial = 0
    for miner in range(miner_count):
        query = _select_query(
            pool, used[miner], rng, policies[miner], lazy_permutations[miner]
        )
        assert query is not None
        trace = sample_trace(
            query,
            rng=rng,
            schedule=schedule,
            work_rate=rates[miner],
            weights=weights,
            trace_bank=trace_bank,
            trace_duration=trace_duration,
            require_trace_bank=require_trace_bank,
            trace_distribution=trace_distribution,
        )
        heapq.heappush(queue, (trace.duration, serial, miner, query, trace))
        serial += 1

    attempts: list[Attempt] = []
    winner: int | None = None
    block_time = 0.0
    while queue and (max_events is None or len(attempts) < max_events):
        completion, _, miner, query, trace = heapq.heappop(queue)
        # Exhausted trials still consume time, which matters for reward rates.
        block_time = completion
        miner_used = used[miner]
        miner_used.add(query.query_id)
        request_id = (
            f"generation-{generations[miner]}:{query.query_id}"
            if replenish_pool
            else query.query_id
        )
        attempt = Attempt(miner, request_id, trace, completion)
        attempts.append(attempt)
        win_probability = -math.expm1(
            trace.complexity * math.log1p(-difficulty_probability)
        )
        if literal_lottery:
            from .lottery import evaluate_tickets, LIMIT
            # Experiment 3 substitutes seeded bindings for ciphertext/proof bytes
            # alongside its empirical work replay, but still hashes every ticket.
            prefix = hashlib.sha256(
                f"poml-empirical-lottery:{seed}:{miner}:{len(attempts)}:{request_id}".encode()
            ).digest()
            won = evaluate_tickets(prefix, trace.complexity,
                                   max(1, int(difficulty_probability * LIMIT)))["winning_ticket"] is not None
        else:
            won = rng.random() < win_probability
        if won:
            attempt.won = True
            winner = miner
            block_time = completion
            break
        next_query = _select_query(
            pool, miner_used, rng, policies[miner], lazy_permutations[miner]
        )
        if next_query is None and replenish_pool:
            generations[miner] += 1
            miner_used.clear()
            if lazy_permutations[miner] is not None:
                lazy_permutations[miner] = LazyPermutation(
                    len(pool), random.Random(rng.randrange(1 << 63))
                )
            next_query = _select_query(
                pool, miner_used, rng, policies[miner], lazy_permutations[miner]
            )
        if next_query is not None:
            next_trace = sample_trace(
                next_query,
                rng=rng,
                schedule=schedule,
                work_rate=rates[miner],
                weights=weights,
                trace_bank=trace_bank,
                trace_duration=trace_duration,
                require_trace_bank=require_trace_bank,
                trace_distribution=trace_distribution,
            )
            heapq.heappush(
                queue,
                (
                    completion + next_trace.duration,
                    serial,
                    miner,
                    next_query,
                    next_trace,
                ),
            )
            serial += 1

    adopted = winner is not None
    # The block contains every pair in the winning miner's proof prefix,
    # including earlier pairs whose own lottery trials did not win.
    canonical = [a for a in attempts if adopted and a.miner_id == winner]
    winning_complexity = sum(a.trace.complexity for a in canonical)
    total = sum(a.trace.complexity for a in attempts)
    settled = 0
    settled_queries = 0
    if adopted and response_inclusion_probability:
        winning_queries = {a.query_id for a in canonical}
        settled_ids: set[str] = set()
        for attempt in attempts:
            if (
                attempt.won
                or attempt.query_id in winning_queries
                or attempt.query_id in settled_ids
            ):
                continue
            # A probability of one is ideal response eligibility accounting,
            # not a separate randomized inclusion/fee/network model.
            if (
                response_inclusion_probability == 1
                or rng.random() < response_inclusion_probability
            ):
                settled += attempt.trace.complexity
                settled_queries += 1
                settled_ids.add(attempt.query_id)
    discarded = max(0, total - winning_complexity - settled)
    unfinished = sum(event[4].complexity for event in queue) if adopted else 0
    return RaceResult(
        adopted,
        winner,
        block_time,
        attempts,
        winning_complexity,
        total,
        settled,
        discarded,
        settled_queries,
        unfinished,
        "adopted" if adopted else "event_limit" if queue else "query_pool_exhausted",
    )


def run_liveness(
    *,
    races: int,
    miner_count: int,
    schedule: Mapping,
    target_block_time: float,
    seed: int,
    pool_size: int = 256,
    work_rate: float = 1.0,
    strategies: Sequence[str] | None = None,
    weights: Mapping[str, int] | None = None,
) -> list[RaceResult]:
    """Run independent races while keeping demand continuously available."""

    pool = make_query_pool(pool_size, seed=seed, schedule=schedule)
    # Calibrate p from the expected ticket rate of a representative miner
    # population.  One work unit is one second in the normalized simulator.
    ticket_rate = miner_count * work_rate
    p = min(0.25, 1.0 / max(1.0, ticket_rate * target_block_time))
    results = []
    for index in range(races):
        results.append(
            simulate_race(
                pool,
                miner_count=miner_count,
                work_rates=(work_rate,) * miner_count,
                difficulty_probability=p,
                seed=seed + index,
                schedule=schedule,
                strategies=strategies,
                weights=weights,
            )
        )
    return results


def trace_from_jsonl(path: str | Path) -> dict[str, list[QueryTrace]]:
    """Load GPT-2 rollout traces keyed by query id and prompt shape.

    Shape keys let a measured trace bank cover large simulated pools without
    pretending that every synthetic query identifier was individually run.
    """

    out: dict[str, list[QueryTrace]] = {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                trace = QueryTrace(
                    int(row["output_length"]),
                    int(row["complexity"]),
                    float(row["duration"]),
                )
                query_id = str(row["query_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid trace at line {line_number}") from error
            if trace.output_length < 1 or trace.complexity < 1 or trace.duration <= 0:
                raise ValueError(f"invalid trace values at line {line_number}")
            out.setdefault(query_id, []).append(trace)
            if "prompt_length" in row:
                try:
                    prompt_length = int(row["prompt_length"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid prompt_length at line {line_number}"
                    ) from error
                out.setdefault(f"shape:{prompt_length}", []).append(trace)
    if not out:
        raise ValueError("trace file is empty")
    return out


def manifest(
    config: SimulationConfig, *, schedule_path: str | Path, pool: Sequence[QuerySpec]
) -> dict:
    """Return a JSON-serialisable provenance manifest."""

    schedule_path = Path(schedule_path)
    return {
        "schema": "poml-llm-experiments-1",
        "config": asdict(config),
        "schedule_path": str(schedule_path),
        "schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        "query_pool": [asdict(query) for query in pool],
    }
