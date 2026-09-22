"""Paper statistics must preserve exposure pooling and collision-only credit."""

import copy
import csv
import json
from pathlib import Path
import statistics
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.plot_llm_paper_results import (  # noqa: E402
    POLICIES,
    collision_plots,
    collision_statistics,
    resolve_collision_source,
    selection_statistics,
    write_collision_report,
)


def write_csv(path, rows):
    """Write the small archived-report fixtures with the production column names."""
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def selection_archive(tmp_path):
    directory = tmp_path / "cherry_pick"
    directory.mkdir()
    (tmp_path / "campaign.json").write_text(
        json.dumps({"seeds": 2, "selection_blocks": 3})
    )
    rows, intervals = [], []
    for index, policy in enumerate(POLICIES, 1):
        for seed, seconds, rate, wins in ((0, 10, 0.8, 1), (1, 30, 1.6, 2)):
            rows.append(
                dict(
                    policy=policy,
                    seed=seed,
                    blocks=3,
                    virtual_chain_seconds=seconds,
                    attacker_tickets_per_chain_second=rate * index,
                    attacker_tickets_per_busy_second=2 * index,
                    attacker_blocks=wins,
                )
            )
        intervals.append(
            dict(
                policy=policy,
                relative_block_yield=1,
                relative_block_yield_ci95_low=0.8,
                relative_block_yield_ci95_high=1.2,
                attacker_block_share_ci95_low=0.3,
                attacker_block_share_ci95_high=0.7,
            )
        )
    write_csv(directory / "per_seed_metrics.csv", rows)
    write_csv(directory / "policy_metrics.csv", intervals)
    return tmp_path, rows


def test_selection_pools_work_and_clock_exposure_before_forming_ratios(
    selection_archive,
):
    path, _ = selection_archive
    result = selection_statistics(path)
    for index, policy in enumerate(POLICIES, 1):
        row = result[policy]
        assert row["tickets_per_chain_second"] == pytest.approx(1.4 * index)
        assert row["tickets_per_completed_second"] == 2 * index
        assert row["canceled_time_fraction"] == pytest.approx(0.3)
        assert row["blocks"] == 6
        assert row["wins"] == 3
        assert row["share"] == 0.5
        assert row["blocks_per_hour"] == 270
        assert row["ticket_rate_ratio"] == index
        assert row["relative_yield_ci95"] == [0.8, 1.2]


def test_selection_rejects_repeated_seed_even_when_row_count_matches(selection_archive):
    path, rows = selection_archive
    rows[1]["seed"] = rows[0]["seed"]
    write_csv(path / "cherry_pick/per_seed_metrics.csv", rows)
    with pytest.raises(ValueError, match="duplicated independent selection seeds"):
        selection_statistics(path)


def test_selection_rejects_chain_shorter_than_manifest(selection_archive):
    path, rows = selection_archive
    rows[1]["blocks"] = 2
    write_csv(path / "cherry_pick/per_seed_metrics.csv", rows)
    with pytest.raises(ValueError, match="chain length"):
        selection_statistics(path)


@pytest.fixture
def collision_archive():
    config = dict(targets=[300], miner_grid=[2], pool_grid=[3], collision_seeds=3)
    cell = dict(
        target_block_time="300",
        miners="2",
        pool_size="3",
        seeds="3",
        adopted_races="2",
        failed_races="1",
        mean_wasted_work_pct="15",
        stdev_wasted_work_pct=str(statistics.stdev([10, 20])),
    )
    trials = [
        dict(
            target_block_time="300",
            miners="2",
            pool_size="3",
            seed=str(seed),
            lottery="aggregate",
            adopted="True" if seed < 2 else "False",
            completed_complexity="100",
            collision_complexity=str(10 * (seed + 1)),
            response_complexity="20",
            wasted_work_pct=str(10 * (seed + 1)) if seed < 2 else "",
        )
        for seed in range(3)
    ]
    return [cell], trials, config


def test_collision_means_exclude_exhaustion_and_credit_responses(collision_archive):
    row = collision_statistics(*collision_archive)["300"]
    assert row["trials"] == 3
    assert row["adopted"] == 2
    assert row["failed"] == 1
    assert row["mean_of_cell_means_pct"] == 15
    assert len(row["failed_cells"]) == 1


def test_collision_rejects_waste_that_includes_eligible_responses(collision_archive):
    cells, trials, config = copy.deepcopy(collision_archive)
    trials[0]["wasted_work_pct"] = "30"
    with pytest.raises(ValueError, match="other than completed collisions"):
        collision_statistics(cells, trials, config)


def test_collision_rejects_missing_feasible_grid_cell(collision_archive):
    cells, trials, config = copy.deepcopy(collision_archive)
    config["pool_grid"].append(4)
    with pytest.raises(ValueError, match="missing or duplicate cells"):
        collision_statistics(cells, trials, config)


def test_collision_rejects_duplicated_trial_seed(collision_archive):
    cells, trials, config = copy.deepcopy(collision_archive)
    trials[1]["seed"] = trials[0]["seed"]
    with pytest.raises(ValueError, match="adoption counts disagree"):
        collision_statistics(cells, trials, config)


def test_collision_full_grid_checks_equal_and_smaller_pools(collision_archive):
    cells, trials, config = copy.deepcopy(collision_archive)
    config.update(full_grid=True, miner_grid=[3])
    cells[0]["miners"] = "3"
    for trial in trials:
        trial["miners"] = "3"
    assert collision_statistics(cells, trials, config)["300"]["cells"] == 1
    config["pool_grid"].append(2)
    with pytest.raises(ValueError, match="missing or duplicate cells"):
        collision_statistics(cells, trials, config)


def test_collision_cell_with_no_adoption_has_no_invented_mean(collision_archive):
    cells, trials, config = copy.deepcopy(collision_archive)
    cells[0].update(
        adopted_races="0",
        failed_races="3",
        mean_wasted_work_pct="",
        stdev_wasted_work_pct="",
    )
    for trial in trials:
        trial.update(adopted="False", wasted_work_pct="")
    result = collision_statistics(cells, trials, config)["300"]
    assert result["adopted"] == 0
    assert result["mean_of_cell_means_pct"] is None


def test_collision_plot_shows_mean_and_sd_in_original_color_scale(
    collision_archive, monkeypatch, tmp_path
):
    import experiments.plot_llm_paper_results as plotting

    cells, _, config = collision_archive
    figures = []
    monkeypatch.setattr(plotting, "save", lambda fig, *_: figures.append(fig))
    collision_plots(cells, config, tmp_path)
    axis = figures[0].axes[0]
    assert axis.images[0].get_cmap().name == "RdYlGn_r"
    assert axis.images[0].get_clim() == (0, 100)
    assert "15%\n±7%" in [t.get_text() for t in axis.texts]
    assert "*" in [t.get_text() for t in axis.texts]
    plotting.plt.close(figures[0])


def test_rebuild_reuses_saved_full_grid_when_collisions_argument_is_omitted(tmp_path):
    baseline = tmp_path / "baseline"
    old_source = baseline / "wasted_work"
    old_source.mkdir(parents=True)
    (old_source / "COMPLETED").touch()
    source = tmp_path / "uniform-fee-scaling"
    source.mkdir()
    (source / "COMPLETED").touch()
    output = tmp_path / "plots"
    output.mkdir()
    (output / "llm_uniform_fee_statistics.json").write_text(
        json.dumps({"source": str(source), "config": {"full_grid": True}})
    )
    assert resolve_collision_source(None, baseline, output) == source
    assert resolve_collision_source(old_source, baseline, output) == old_source
    (source / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="unfinished collision source"):
        resolve_collision_source(None, baseline, output)


def test_fresh_output_can_still_reproduce_baseline_archive(tmp_path):
    source = tmp_path / "baseline/wasted_work"
    source.mkdir(parents=True)
    (source / "COMPLETED").touch()
    assert resolve_collision_source(None, source.parent, tmp_path / "plots") == source


def test_collision_only_rebuild_updates_combined_provenance_without_other_results(
    tmp_path,
):
    source = tmp_path / "uniform-fee-scaling"
    source.mkdir()
    for name in ("campaign.json", "cells.csv", "runs.csv"):
        (source / name).write_text("fixture\n")
    old_source = tmp_path / "baseline/wasted_work"
    report = tmp_path / "llm_paper_figure_statistics.json"
    untouched_input = str(tmp_path / "liveness/blocks.json")
    report.write_text(
        json.dumps(
            {
                "liveness": {"mean": 300},
                "collision_source": str(old_source),
                "collisions": {"300": {"cells": 75}},
                "input_sha256": {
                    str(old_source / "cells.csv"): "old",
                    untouched_input: "retained",
                },
            }
        )
    )
    stats = {"300": {"cells": 120}}
    config = {"full_grid": True, "collision_metric": "first_completion"}
    write_collision_report(source, config, stats, tmp_path)
    updated = json.loads(report.read_text())
    assert updated["liveness"] == {"mean": 300}
    assert updated["collision_source"] == str(source)
    assert updated["collisions"] == stats
    assert str(old_source / "cells.csv") not in updated["input_sha256"]
    assert updated["input_sha256"][untouched_input] == "retained"
    assert str(source / "runs.csv") in updated["input_sha256"]
    assert "raw unit weights" in updated["complexity_scope"]


def test_full_grid_plot_leaves_no_diagonal_or_lower_triangle_cell_masked(
    collision_archive, monkeypatch, tmp_path
):
    import numpy as np
    import experiments.plot_llm_paper_results as plotting

    cells, _, config = collision_archive
    config.update(full_grid=True, miner_grid=[2, 3], pool_grid=[2, 4])
    cells = [
        dict(cells[0], miners=str(m), pool_size=str(q))
        for m in config["miner_grid"]
        for q in config["pool_grid"]
    ]
    figures = []
    monkeypatch.setattr(plotting, "save", lambda fig, *_: figures.append(fig))
    collision_plots(cells, config, tmp_path)
    matrix = figures[0].axes[0].images[0].get_array()
    assert matrix.shape == (2, 2)
    assert not np.ma.getmaskarray(matrix).any()
    assert np.all(matrix == 15)
    plotting.plt.close(figures[0])
