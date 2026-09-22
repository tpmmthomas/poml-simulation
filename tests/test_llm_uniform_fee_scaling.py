"""Regressions for the paper's homogeneous, complexity-weighted grid replay."""

from argparse import Namespace
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import run_llm_uniform_fee_scaling as scaling  # noqa: E402


def test_constant_complexity_reduces_to_original_query_count_metric():
    result = scaling.simulate_uniform_race(
        [(1.0, 10)], miners=5, queries=2, difficulty=scaling.LIMIT // 40, seed=8
    )
    assert result["adopted"]
    assert result["wasted_work_pct"] == pytest.approx(
        100
        * (result["completed_pairs"] - result["unique_queries"])
        / result["completed_pairs"]
    )
    assert result["completed_complexity"] == 10 * result["completed_pairs"]


def test_exhaustion_processes_every_identifier_once_per_miner():
    result = scaling.simulate_uniform_race(
        [(1.0, 10), (2.0, 30)], miners=4, queries=3, difficulty=1, seed=4
    )
    assert not result["adopted"]
    assert result["termination_reason"] == "query_pool_exhausted"
    assert result["completed_pairs"] == 12
    assert result["unique_queries"] == 3
    assert result["duplicate_pairs"] == 9
    assert result["canceled_attempts"] == 0
    assert result["wasted_work_pct"] is None
    assert (
        result["first_completed_complexity"] + result["collision_complexity"]
        == result["completed_complexity"]
    )


def test_first_pair_keeps_credit_even_when_later_duplicate_wins(monkeypatch):
    original = random.Random
    draws = iter([1.0, 0.0])

    class Lottery:
        def random(self):
            return next(draws)

    seed = 5
    monkeypatch.setattr(
        scaling.random,
        "Random",
        lambda s: (
            Lottery() if s == scaling.derive_seed(seed, "lottery") else original(s)
        ),
    )
    result = scaling.simulate_uniform_race(
        [(1.0, 10), (2.0, 30)],
        miners=2,
        queries=1,
        difficulty=scaling.LIMIT // 100,
        seed=seed,
    )
    assert result["adopted"]
    assert result["completed_pairs"] == 2
    assert result["first_completed_complexity"] == 10
    assert result["collision_complexity"] == 30
    assert result["wasted_work_pct"] == 75


def test_adoption_excludes_pending_attempts_and_cells_are_reproducible():
    job = ([(1.0, 10)], 0.025, 5, 2, 3, 8)
    first = scaling.run_cell(job)
    assert first == scaling.run_cell(job)
    for run in first["runs"]:
        assert run["completed_pairs"] == 1
        assert run["completed_complexity"] == 10
        assert run["canceled_attempts"] == 4
        assert run["wasted_work_pct"] == 0
    assert first["cell"]["stdev_wasted_work_pct"] == 0


def test_calibration_generalizes_original_threshold_per_complexity_unit():
    samples = [(10.0, 100), (20.0, 200)]
    assert scaling.calibrated_difficulty(samples, 10, 300) == scaling.LIMIT // 30000


@pytest.mark.parametrize("samples", [[(0.0, 1)], [(float("nan"), 1)], [(1.0, 0)]])
def test_invalid_empirical_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="samples require"):
        scaling.simulate_uniform_race(
            samples, miners=1, queries=1, difficulty=1, seed=0
        )


def measurement():
    """Return a minimal verified measurement-bank record."""
    return dict(
        duration=1.0,
        raw_complexity=10,
        verified=True,
        fresh_inference=True,
        fresh_proof=True,
        reference_counts={"combined": {"op": 10}},
    )


def test_measurement_bank_preserves_paired_raw_complexity_and_duration(tmp_path):
    path = tmp_path / "timings.jsonl"
    row = measurement()
    row["complexity"] = 999  # The old rounded ticket scale must not enter the metric.
    path.write_text(json.dumps(row) + "\n")
    assert scaling.load_measurements(path) == [(1.0, 10)]
    row["raw_complexity"] = 11
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="reference counts"):
        scaling.load_measurements(path)
    row = measurement()
    row["verified"] = False
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="verified fresh"):
        scaling.load_measurements(path)


def test_campaign_fills_diagonal_and_lower_triangle_and_resumes(tmp_path):
    source = tmp_path / "source"
    preparation = source / "preparation"
    preparation.mkdir(parents=True)
    (preparation / "COMPLETED").write_text("complete\n")
    bank = preparation / "timings.jsonl"
    bank.write_text((json.dumps(measurement()) + "\n") * 2)
    args = Namespace(
        input=source,
        output=tmp_path / "run",
        targets=[1.0],
        miner_grid=[2, 3],
        pool_grid=[2, 4],
        replicates=3,
        workers=1,
        seed=41,
        resume=False,
    )
    report = scaling.run_campaign(args)
    assert report["1"]["cells"] == 4
    assert report["1"]["trials"] == 12
    assert (args.output / "COMPLETED").exists()
    assert (args.output / "figures/llm_collision_1s.pdf").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        scaling.run_campaign(args)
    args.resume = True
    assert scaling.run_campaign(args) == report
    bank.write_text(bank.read_text() + json.dumps(measurement()) + "\n")
    with pytest.raises(ValueError, match="identical inputs"):
        scaling.run_campaign(args)
