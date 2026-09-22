"""Regression tests for the Experiment 5c regular M-by-Q heatmap driver."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.exp5c_m_q_heatmap as experiment


def test_design_grid_filters_infeasible_cells_and_orders_them():
    cells = experiment.design_grid()

    assert len(cells) == 75
    assert all(cell["query_pool_size"] > cell["miners"] for cell in cells)
    assert {cell["miners"] for cell in cells} == set(experiment.MINER_GRID)
    feasible_queries = {cell["query_pool_size"] for cell in cells}
    assert feasible_queries == {20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000}
    assert all(cell["q_over_m"] == cell["query_pool_size"] / cell["miners"] for cell in cells)
    indices = [(cell["query_pool_size"], cell["miners"]) for cell in cells]
    assert indices == sorted(indices)


def test_design_grid_full_grid_includes_q_le_m_cells():
    cells = experiment.design_grid(full_grid=True)

    assert len(cells) == 10 * 12
    assert all(cell["query_pool_size"] > 0 and cell["miners"] > 0 for cell in cells)
    assert sum(cell["query_pool_size"] > cell["miners"] for cell in cells) == 75
    assert sum(cell["query_pool_size"] == cell["miners"] for cell in cells) == 9
    assert sum(cell["query_pool_size"] < cell["miners"] for cell in cells) == 36
    assert all(cell["q_over_m"] == cell["query_pool_size"] / cell["miners"] for cell in cells)
    indices = [(cell["query_pool_size"], cell["miners"]) for cell in cells]
    assert indices == sorted(indices)


def test_summary_includes_adopted_zero_waste_runs():
    rows = [
        {
            "status": "adopted",
            "percentage_wasted_work": "0.0",
            "target_block_time_s": "300.0",
            "miners": "10",
            "query_pool_size": "100",
        },
        {
            "status": "adopted",
            "percentage_wasted_work": "50.0",
            "target_block_time_s": "300.0",
            "miners": "10",
            "query_pool_size": "100",
        },
    ]

    [summary] = experiment.summarize_cells(rows)

    assert summary["replicates"] == 2
    assert summary["adopted_runs"] == 2
    assert summary["mean_percentage_wasted_work"] == 25.0


def test_campaign_writes_runs_summary_heatmaps_and_deterministic_seeds(tmp_path, monkeypatch):
    timing_path = tmp_path / "timings.json"
    timing_path.write_text(
        json.dumps(
            {
                "artifact_kind": "ezkl_attempt_timing_calibration",
                "artifact_id": "synthetic-timing",
                "duration_samples_s": [1.0, 1.25, 1.5],
            }
        )
    )

    def fake_simulate_race(**arguments):
        del arguments
        return (
            {
                "status": "adopted",
                "adoption_time_s": 2.0,
                "winner_miner_id": 0,
                "winner_query_id": 1,
                "completed_pairs": 8,
                "collision_count": 2,
                "collision_ratio": 0.25,
                "partial_attempts_discarded": 2,
                "tied_completions_cancelled": 0,
                "unique_queries_completed": 6,
            },
            [],
        )

    monkeypatch.setattr(experiment, "simulate_race", fake_simulate_race)
    output_dir = tmp_path / "exp5c"
    campaign = experiment.run_campaign(
        timing_path=timing_path,
        output_dir=output_dir,
        targets_s=(300.0, 600.0),
        replicates=2,
        miner_values=(10, 50, 200),
        query_values=(50, 100, 500, 1_000),
        seed_root="test-exp5c-campaign",
    )

    assert campaign["campaign_status"] == "complete"
    # 2 targets * 2 replicates * 9 feasible cells (M=10/Q=50..1000, M=50/Q=100..1000, M=200/Q=500..1000)
    assert campaign["run_count"] == 36
    assert campaign["design"]["feasible_cells"] == 9

    with (output_dir / "runs.csv").open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert len(rows) == campaign["run_count"]
    assert all(float(row["percentage_wasted_work"]) == 25.0 for row in rows)

    with (output_dir / "summary.csv").open(newline="") as file_handle:
        summary_rows = list(csv.DictReader(file_handle))
    assert len(summary_rows) == 2 * 9
    for row in summary_rows:
        assert int(row["query_pool_size"]) > int(row["miners"])
        assert float(row["mean_percentage_wasted_work"]) == 25.0
        assert float(row["std_percentage_wasted_work"]) == 0.0
        assert int(row["replicates"]) == 2
        assert int(row["adopted_runs"]) == 2

    seeds = [row["run_seed"] for row in rows]
    assert len(set(seeds)) == len(seeds)

    for target in (300, 600):
        png = output_dir / "figures" / f"wasted_work_heatmap_m_q_target_{target}s.png"
        pdf = output_dir / "figures" / f"wasted_work_heatmap_m_q_target_{target}s.pdf"
        assert png.is_file() and png.stat().st_size > 0
        assert pdf.is_file() and pdf.stat().st_size > 0


def test_heatmap_regeneration_from_existing_runs(tmp_path, monkeypatch):
    timing_path = tmp_path / "timings.json"
    timing_path.write_text(
        json.dumps(
            {
                "artifact_kind": "ezkl_attempt_timing_calibration",
                "artifact_id": "synthetic-timing",
                "duration_samples_s": [1.0, 1.25, 1.5],
            }
        )
    )

    def fake_simulate_race(**arguments):
        del arguments
        return (
            {
                "status": "adopted",
                "adoption_time_s": 1.0,
                "winner_miner_id": 0,
                "winner_query_id": 1,
                "completed_pairs": 4,
                "collision_count": 1,
                "collision_ratio": 0.5,
                "partial_attempts_discarded": 0,
                "tied_completions_cancelled": 0,
                "unique_queries_completed": 3,
            },
            [],
        )

    monkeypatch.setattr(experiment, "simulate_race", fake_simulate_race)
    campaign_dir = tmp_path / "campaign"
    experiment.run_campaign(
        timing_path=timing_path,
        output_dir=campaign_dir,
        targets_s=(300.0,),
        replicates=1,
        miner_values=(10, 100),
        query_values=(50, 500),
        seed_root="test-exp5c-plot",
    )

    figures_dir = tmp_path / "regen"
    outputs = experiment.generate_heatmaps(
        campaign_dir / "runs.csv", figures_dir
    )

    assert len(outputs) == 2
    assert (figures_dir / "wasted_work_heatmap_m_q_target_300s.png").is_file()
    assert (figures_dir / "wasted_work_heatmap_m_q_target_300s.pdf").is_file()
