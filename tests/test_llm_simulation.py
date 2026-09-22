"""Tests for the revised event-driven LLM PoML experiments."""

from pathlib import Path
import random

import pytest

from poml_sim.llm_simulation import (
    QuerySpec,
    QueryTrace,
    LazyPermutation,
    complexity_for,
    load_schedule,
    make_query_pool,
    sample_trace,
    simulate_race,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = load_schedule(ROOT / "config/gpt2_reference_schedule.json")


def test_query_pool_does_not_fix_output_length():
    pool = make_query_pool(12, seed=7, schedule=SCHEDULE, max_output_length=16)
    assert len(pool) == 12
    assert all(query.prompt_length >= 2 for query in pool)
    assert all(query.max_output_length == 16 for query in pool)
    assert all(query.profile_mean_output_length is None for query in pool)
    assert all(query.profile_trials == 0 for query in pool)


def test_complexity_is_deterministic_for_realised_output_length():
    query = QuerySpec("q", 8, 16)
    assert complexity_for(query, 4, SCHEDULE) == complexity_for(query, 4, SCHEDULE)
    assert complexity_for(query, 4, SCHEDULE) != complexity_for(query, 12, SCHEDULE)


def test_trace_samples_k_only_after_a_challenge():
    query = QuerySpec("q", 8, 16, category="short")
    first = sample_trace(
        query, rng=__import__("random").Random(1), schedule=SCHEDULE, work_rate=1e9
    )
    second = sample_trace(
        query, rng=__import__("random").Random(2), schedule=SCHEDULE, work_rate=1e9
    )
    assert 1 <= first.output_length <= 16
    assert 1 <= second.output_length <= 16
    assert first.complexity > 0 and first.duration > 0


def test_race_is_reproducible_and_has_consistent_accounting():
    pool = make_query_pool(128, seed=3, schedule=SCHEDULE, max_output_length=8)
    kwargs = dict(
        pool=pool,
        miner_count=4,
        work_rates=(1e9,) * 4,
        difficulty_probability=1e-10,
        seed=99,
        schedule=SCHEDULE,
    )
    first = simulate_race(**kwargs)
    second = simulate_race(**kwargs)
    assert first.block_time == second.block_time
    assert first.total_completed_complexity == second.total_completed_complexity
    assert [a.query_id for a in first.attempts] == [a.query_id for a in second.attempts]
    assert (
        first.discarded_complexity + first.settled_complexity + first.winning_complexity
        == first.total_completed_complexity
    )


def test_response_settlement_reduces_discarded_work():
    pool = make_query_pool(256, seed=5, schedule=SCHEDULE, max_output_length=8)
    base = dict(
        pool=pool,
        miner_count=4,
        work_rates=(1e9,) * 4,
        difficulty_probability=1e-10,
        seed=123,
        schedule=SCHEDULE,
    )
    without = simulate_race(**base, response_inclusion_probability=0.0)
    with_response = simulate_race(**base, response_inclusion_probability=1.0)
    assert with_response.discarded_complexity <= without.discarded_complexity


def _response_race(seed, **overrides):
    arguments = dict(
        pool=[QuerySpec("a", 2, 4), QuerySpec("b", 4, 4), QuerySpec("c", 8, 4)],
        miner_count=3,
        difficulty_probability=0.01,
        seed=seed,
        schedule=SCHEDULE,
        trace_bank={
            "a": (QueryTrace(2, 10, 1), QueryTrace(3, 30, 1.2)),
            "b": (QueryTrace(2, 20, 1),),
            "c": (QueryTrace(2, 50, 1),),
        },
        trace_duration=True,
        response_inclusion_probability=1,
        max_events=None,
    )
    arguments.update(overrides)
    return simulate_race(**arguments)


def test_unique_losing_queries_are_useful_responses_not_waste():
    result = _response_race(0)
    assert result.adopted
    assert len(result.attempts) == 3
    assert result.winning_complexity == 30
    assert result.settled_complexity == 70
    assert result.settled_queries == 2
    assert result.discarded_complexity == 0


def test_only_extra_completions_are_wasted_using_complexity_units():
    result = _response_race(1)
    assert result.adopted
    assert result.total_completed_complexity == 170
    assert result.winning_complexity == 80
    assert result.settled_complexity == 0
    assert result.discarded_complexity == 90
    assert (
        result.discarded_complexity / 170 != 4 / 7
    )  # Query counts are a different metric.


def test_response_accounting_preserves_the_event_stream():
    without = _response_race(1, response_inclusion_probability=0)
    with_response = _response_race(1)
    assert without.attempts == with_response.attempts
    assert without.winner == with_response.winner
    assert without.block_time == with_response.block_time
    assert without.unfinished_complexity == with_response.unfinished_complexity
    assert (
        without.discarded_complexity
        == with_response.discarded_complexity + with_response.settled_complexity
    )


def test_canonical_completion_takes_priority_over_duplicate_response():
    result = _response_race(18)
    assert [(a.query_id, a.trace.complexity) for a in result.attempts] == [
        ("a", 10),
        ("b", 20),
        ("a", 30),
    ]
    assert result.winning_complexity == 30
    assert result.settled_complexity == 20
    assert result.discarded_complexity == 10


def test_entire_winning_prefix_takes_priority_when_duplicate_lengths_differ():
    result = _response_race(32)
    assert result.winner == 2
    # Miner 2's earlier a=30 must be credited over miner 1's a=10 response.
    assert result.winning_complexity == 30 + 20 + 50
    assert result.settled_complexity == 0
    assert result.discarded_complexity == 160


def test_event_limit_is_distinguished_from_pool_exhaustion():
    capped = _response_race(1, max_events=1)
    completed = _response_race(1)
    exhausted = _response_race(2)
    assert capped.termination_reason == "event_limit"
    assert not capped.adopted
    assert capped.attempts == completed.attempts[:1]
    assert completed.termination_reason == "adopted"
    assert exhausted.termination_reason == "query_pool_exhausted"
    assert not exhausted.adopted


def test_lazy_permutation_has_no_repeated_identifiers():
    permutation = LazyPermutation(100, random.Random(42))
    assert set(permutation.draw() for _ in range(100)) == set(range(100))
    assert permutation.draw() is None


def test_invalid_trace_and_pool_inputs_fail():
    with pytest.raises(ValueError):
        make_query_pool(0, seed=1, schedule=SCHEDULE)
    with pytest.raises(ValueError):
        complexity_for(QuerySpec("q", 1, 4), 2, SCHEDULE)


def test_real_trace_mode_uses_measured_duration_and_rejects_missing_queries():
    query = QuerySpec("q", 8, 16)
    trace = sample_trace(
        query,
        rng=__import__("random").Random(1),
        schedule=SCHEDULE,
        work_rate=1e9,
        trace_bank={"q": [QueryTrace(4, 123, 7.5)]},
        trace_duration=True,
        require_trace_bank=True,
    )
    assert trace.duration == 7.5
    shaped = sample_trace(
        QuerySpec("generated-id", 8, 16),
        rng=__import__("random").Random(1),
        schedule=SCHEDULE,
        work_rate=1e9,
        trace_bank={"shape:8": [QueryTrace(4, 123, 7.5)]},
        trace_duration=True,
        require_trace_bank=True,
    )
    assert shaped.duration == 7.5
    distributed = sample_trace(
        QuerySpec("distributed", 8, 16),
        rng=__import__("random").Random(1),
        schedule=SCHEDULE,
        work_rate=1e9,
        trace_distribution=[QueryTrace(4, 999, 8.25)],
        trace_duration=True,
        require_trace_bank=True,
    )
    assert distributed.output_length == 4
    assert distributed.duration == 8.25
    assert distributed.complexity != 999
    with pytest.raises(ValueError, match="missing real trace"):
        sample_trace(
            QuerySpec("missing", 8, 16),
            rng=__import__("random").Random(1),
            schedule=SCHEDULE,
            work_rate=1e9,
            trace_bank={"q": [trace]},
            trace_duration=True,
            require_trace_bank=True,
        )


def test_sufficient_demand_replenishes_until_adoption_without_losing_elapsed_time():
    query = QuerySpec("q", 8, 8)
    trace = QueryTrace(1, 1, 10.0)
    result = simulate_race(
        [query],
        miner_count=1,
        difficulty_probability=0.1,
        seed=4,
        schedule=SCHEDULE,
        trace_bank={"q": [trace]},
        trace_duration=True,
        require_trace_bank=True,
        max_events=None,
        replenish_pool=True,
        response_inclusion_probability=1.0,
    )
    assert result.adopted
    assert len(result.attempts) > 1
    assert result.block_time == 10 * len(result.attempts)
    assert len({a.query_id for a in result.attempts}) == len(result.attempts)
    assert result.discarded_complexity == 0


def test_exhausted_query_pool_preserves_time_spent_on_failed_attempts():
    query = QuerySpec("q", 8, 8)
    result = simulate_race(
        [query],
        miner_count=1,
        difficulty_probability=1e-20,
        seed=4,
        schedule=SCHEDULE,
        trace_bank={"q": [QueryTrace(1, 1, 10.0)]},
        trace_duration=True,
        require_trace_bank=True,
    )
    assert not result.adopted
    assert result.block_time == 10.0
