from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.exp5b_ratio_scatter as experiment


def test_sample_pairs_are_deterministic_unique_and_balanced():
    arguments = {
        "count": 100,
        "ratio_bins": 10,
        "max_value": 1_000,
        "seed_root": "test-exp5b-pairs",
    }

    first = experiment.sample_pairs(**arguments)
    second = experiment.sample_pairs(**arguments)

    assert first == second
    assert len(first) == 100
    assert len({(pair["query_pool_size"], pair["miners"]) for pair in first}) == 100
    assert Counter(pair["ratio_bin"] for pair in first) == {index: 10 for index in range(10)}
    assert all(
        1 <= pair["miners"] < pair["query_pool_size"] <= 1_000
        for pair in first
    )
    assert all(
        pair["q_over_m"] == pair["query_pool_size"] / pair["miners"]
        for pair in first
    )


def test_campaign_reuses_pairs_and_writes_percentage_and_heatmaps(tmp_path, monkeypatch):
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
                "collision_count": 1,
                "collision_ratio": 0.125,
                "partial_attempts_discarded": 2,
                "tied_completions_cancelled": 0,
                "unique_queries_completed": 7,
            },
            [],
        )

    monkeypatch.setattr(experiment, "simulate_race", fake_simulate_race)
    output_dir = tmp_path / "campaign"
    campaign = experiment.run_campaign(
        timing_path=timing_path,
        output_dir=output_dir,
        pair_count=6,
        ratio_bins=3,
        max_value=100,
        targets_s=(300.0, 600.0),
        seed_root="test-exp5b-campaign",
    )

    with (output_dir / "runs.csv").open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    pairs_by_target = {
        target: {
            (row["pair_index"], row["miners"], row["query_pool_size"], row["q_over_m"])
            for row in rows
            if float(row["target_block_time_s"]) == target
        }
        for target in (300.0, 600.0)
    }

    assert campaign["run_count"] == 12
    assert campaign["status_counts"] == {"adopted": 12}
    assert len(rows) == 12
    assert pairs_by_target[300.0] == pairs_by_target[600.0]
    assert all(float(row["percentage_wasted_work"]) == 12.5 for row in rows)
    assert not (output_dir / "details").exists()
    for target in (300, 600):
        assert (output_dir / "figures" / f"wasted_work_heatmap_target_{target}s.png").is_file()
        assert (output_dir / "figures" / f"wasted_work_heatmap_target_{target}s.pdf").is_file()