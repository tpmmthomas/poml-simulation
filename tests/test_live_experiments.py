"""Driver checks for checkpoint recovery, literal replay, and honest reporting."""

import argparse
from pathlib import Path
import sys

from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_live_llm_experiments import (  # noqa: E402
    chain_state,
    restore_chain,
    collision_cell,
    parser,
)  # noqa: E402
from poml_sim.llm_simulation import QueryTrace  # noqa: E402
from poml_sim.live_mining import LiveChain  # noqa: E402
from poml_sim.lottery import evaluate_tickets  # noqa: E402


def test_chain_checkpoint_restores_selection_and_parent():
    work = SimpleNamespace(prover=SimpleNamespace(ready={}), scale={})
    pool = [{"query_id": "q1"}, {"query_id": "q2"}]
    chain = LiveChain(work, pool, 2, 1, 41)
    chain.rng.random()
    chain.height = 4
    chain.pending.pop()
    state = chain_state(chain)
    copy = LiveChain(work, pool, 2, 1, 41)
    restore_chain(copy, state)
    assert copy.parent == chain.parent
    assert copy.pending == chain.pending
    assert copy.rng.random() == chain.rng.random()


def test_literal_empirical_cell_retains_response_collision_partition(monkeypatch):
    import poml_sim.lottery

    calls = []

    def observed(prefix, count, difficulty):
        calls.append(count)
        return evaluate_tickets(prefix, count, difficulty)

    monkeypatch.setattr(poml_sim.lottery, "evaluate_tickets", observed)
    args = argparse.Namespace(seed=11, collision_seeds=3, collision_lottery="literal")
    templates = [{"query_id": "a", "prompt_length": 2, "max_output_length": 1}]
    bank = {"a": [QueryTrace(1, 10, 1)]}
    result = collision_cell((args, {}, bank, templates, 10, 1, 2, 5))
    assert calls and set(calls) == {10}
    assert len(result["runs"]) == 3
    for row in result["runs"]:
        assert 0 <= row["collision_complexity"] <= row["completed_complexity"]
        assert row["response_complexity"] <= row["completed_complexity"]


def test_default_empirical_cell_samples_probability_without_hashing(monkeypatch):
    def reject_hashes(*args, **kwargs):
        raise AssertionError("aggregate Experiment 3 must not enumerate tickets")

    monkeypatch.setattr("poml_sim.lottery.evaluate_tickets", reject_hashes)
    args = parser().parse_args(["all", "--output", "unused", "--collision-seeds", "3"])
    templates = [{"query_id": "a", "prompt_length": 2, "max_output_length": 1}]
    bank = {"a": [QueryTrace(1, 10, 1)]}
    job = (args, {}, bank, templates, 10, 1, 2, 5)
    result = collision_cell(job)
    assert result == collision_cell(job)
    assert result["cell"]["adopted_races"] > 0
    for row in result["runs"]:
        assert row["lottery"] == "aggregate"
        assert row["per_ticket_probability"] == int(row["difficulty"]) / 2**256
        assert (
            0
            <= row["collision_complexity"] + row["response_complexity"]
            <= row["completed_complexity"]
        )


def test_genesis_ignores_setup_elapsed_time():
    first = SimpleNamespace(
        prover=SimpleNamespace(ready={"setup_sha256": "fixed", "setup_seconds": 1}),
        scale={},
    )
    second = SimpleNamespace(
        prover=SimpleNamespace(ready={"setup_sha256": "fixed", "setup_seconds": 999}),
        scale={},
    )
    assert (
        LiveChain(first, [{"query_id": "q"}], 1, 1, 0).parent
        == LiveChain(second, [{"query_id": "q"}], 1, 1, 0).parent
    )


def test_pooled_yield_keeps_zero_win_baseline_seeds():
    from experiments.run_live_llm_experiments import pooled_relative_yield

    baseline = [
        dict(seed=i, attacker_block_share=w, blocks=1, virtual_chain_seconds=10)
        for i, w in enumerate([0, 1])
    ]
    selected = [
        dict(seed=i, attacker_block_share=1, blocks=1, virtual_chain_seconds=10)
        for i in range(2)
    ]
    result = pooled_relative_yield(selected, baseline, 1)
    assert result["relative_block_yield"] == 2
    assert result["relative_yield_undefined_bootstrap_fraction"] > 0.2
    assert "relative_block_yield_ci95_low" not in result
