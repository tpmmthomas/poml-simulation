"""Specifications for distinct benchmark queries and independent profiling."""

from dataclasses import replace
import random

import pytest

from poml_sim.llm_benchmark import (
    challenge_seed,
    prediction_diagnostics,
    profile_pool,
    wikitext_prompts,
)
from poml_sim.llm_simulation import QuerySpec, _select_query


def _prompts():
    return [
        dict(
            query_id=f"q{i}", prompt_sha256=str(i), prompt_length=2, max_output_length=8
        )
        for i in range(2)
    ]


def _observations(phase, lengths):
    return [
        dict(
            query_id=f"q{i}",
            prompt_sha256=str(i),
            phase=phase,
            challenge_seed=challenge_seed(7, phase, f"q{i}", j),
            output_length=k,
            gpt2_duration=0.1,
            stop_reason="eos",
        )
        for i, values in enumerate(lengths)
        for j, k in enumerate(values)
    ]


def test_benchmark_windows_are_distinct_real_source_slices():
    rows = [{"text": " ".join(str(i * 10 + j) for j in range(8))} for i in range(10)]
    arguments = dict(count=6, lengths=(2, 4), max_output=8, seed=1)

    def encode(text):
        return list(map(int, text.split()))

    prompts = wikitext_prompts(rows, encode, **arguments)
    assert prompts == wikitext_prompts(rows, encode, **arguments)
    assert len({p["source_row"] for p in prompts}) == 6
    assert len({p["prompt_sha256"] for p in prompts}) == 6
    for prompt in prompts:
        source = encode(rows[prompt["source_row"]]["text"])
        offset = prompt["token_offset"]
        assert (
            prompt["prompt_tokens"] == source[offset : offset + prompt["prompt_length"]]
        )


def test_insufficient_benchmark_data_is_not_repeated_or_padded():
    with pytest.raises(ValueError, match="distinct eligible"):
        wikitext_prompts(
            [{"text": "abc"}],
            lambda _: [1, 2],
            count=2,
            lengths=(2,),
            max_output=4,
            seed=1,
        )


def test_profile_means_do_not_use_held_out_lengths():
    prompts = _prompts()
    profile = _observations("profile", [[1, 3], [7, 7]])
    frozen = profile_pool(prompts, profile)
    assert [p["profile_mean_output_length"] for p in frozen] == [2, 7]
    evaluation = _observations("evaluation", [[8, 8], [1, 1]])
    diagnostics = prediction_diagnostics(frozen, profile, evaluation)
    assert diagnostics["spearman_rank_correlation"] == pytest.approx(-1)
    assert [p["profile_mean_output_length"] for p in frozen] == [2, 7]
    with pytest.raises(ValueError, match="cannot be used"):
        profile_pool(prompts, evaluation)


def test_overlapping_randomness_and_duplicate_profiles_are_rejected():
    profile = _observations("profile", [[1, 3], [7, 7]])
    evaluation = _observations("evaluation", [[2, 2], [8, 8]])
    frozen = profile_pool(_prompts(), profile)
    evaluation[0]["challenge_seed"] = profile[0]["challenge_seed"]
    with pytest.raises(ValueError, match="distinct"):
        prediction_diagnostics(frozen, profile, evaluation)
    with pytest.raises(ValueError, match="duplicate"):
        profile_pool(_prompts(), profile + [profile[0]])


def test_equal_capped_profiles_do_not_claim_an_identifiable_ranking():
    profile = _observations("profile", [[8, 8], [8, 8]])
    evaluation = _observations("evaluation", [[8, 8], [8, 8]])
    diagnostics = prediction_diagnostics(
        profile_pool(_prompts(), profile), profile, evaluation
    )
    assert diagnostics["ranking_identifiable"] is False
    assert diagnostics["spearman_rank_correlation"] is None


def test_profiled_short_and_long_select_only_by_observed_profile_mean():
    short = QuerySpec("short", 32, 8, profile_mean_output_length=2, profile_trials=16)
    long = QuerySpec("long", 2, 8, profile_mean_output_length=7, profile_trials=16)
    assert (
        _select_query([short, long], set(), random.Random(1), "profiled-short") == short
    )
    assert (
        _select_query([short, long], set(), random.Random(1), "profiled-long") == long
    )
    assert (
        _select_query([short, long], {"short"}, random.Random(1), "profiled-short")
        == long
    )


def test_profile_ties_are_random_instead_of_a_hidden_prompt_length_policy():
    first = QuerySpec("a", 2, 8, profile_mean_output_length=8, profile_trials=16)
    second = replace(first, query_id="b", prompt_length=32)
    choices = {
        _select_query(
            [first, second], set(), random.Random(seed), "profiled-short"
        ).query_id
        for seed in range(20)
    }
    assert choices == {"a", "b"}


def test_profiled_selection_rejects_unmeasured_length_labels():
    with pytest.raises(ValueError, match="independent measured profiles"):
        _select_query([QuerySpec("q", 2, 8)], set(), random.Random(1), "profiled-long")
