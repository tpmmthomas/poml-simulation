"""Counterfactual reports must hold historical selections and work time fixed."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.report_runtime_selection import historical_reweighting  # noqa: E402
from experiments.report_runtime_selection import effective_work_rates  # noqa: E402


def test_historical_reweighting_excludes_canceled_and_other_miners(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "experiments.report_runtime_selection.reference_counts",
        lambda n, k, schedule: {"combined": {"op": n * k}},
    )
    completed = [
        dict(
            miner=0,
            logical_status="completed",
            duration=t,
            prompt_length=3,
            output_length=k,
            complexity=3 * k,
        )
        for t, k in ((3, 2), (7, 4))
    ]
    rows = [
        *completed,
        {**completed[0], "logical_status": "canceled", "complexity": 9999},
        {**completed[0], "miner": 1, "complexity": 9999},
    ]
    for policy in ("uniform", "profiled-short", "profiled-long", "shortest-prompt"):
        directory = tmp_path / "cherry_pick" / f"000-{policy}"
        directory.mkdir(parents=True)
        (directory / "attempts.json").write_text(json.dumps(rows))
    fit = {"weights": {"op": 2}, "scale": {"numerator": 1, "denominator": 1}}
    for row in historical_reweighting(tmp_path, fit, {}):
        assert row["completed_pairs"] == 2
        assert row["old_rate"] == 1.8
        assert row["reweighted_same_attempts_rate"] == 3.6
        assert row["old_relative_uniform"] == row["reweighted_relative_uniform"] == 1


def test_effective_rates_pool_clock_time_instead_of_averaging_seed_rates(tmp_path):
    for policy in ("uniform", "profiled-short", "profiled-long", "shortest-prompt"):
        for repeat, seconds, rate in ((0, 10, 0.8), (1, 30, 1.6)):
            directory = tmp_path / "cherry_pick" / f"{repeat:03d}-{policy}"
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(
                json.dumps(
                    {
                        "virtual_chain_seconds": seconds,
                        "attacker_tickets_per_chain_second": rate,
                        "attacker_tickets_per_busy_second": 2,
                    }
                )
            )
    for row in effective_work_rates(tmp_path):
        assert row["completed_tickets_per_chain_second"] == 1.4
        assert row["completed_time_fraction"] == 0.7
        assert abs(row["canceled_time_fraction"] - 0.3) < 1e-12
