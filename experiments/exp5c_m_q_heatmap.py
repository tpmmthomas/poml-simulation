#!/usr/bin/env python3
"""Experiment 5c: wasted work on a regular miner-count by query-pool grid.

Experiment 5b sampled random (Q, M) pairs and summarized them in coarse
banded heatmaps. Experiment 5c re-runs the Experiment 5 race engine on a
regular, denser log-spaced (M, Q) grid so each heatmap cell is a single
exact configuration, replicated across several seeds.

Grid design (derived from the Experiment 5b scatter results):

* The dynamic regime is Q/M in roughly [1, 100]: median wasted work falls
  from ~85% near the Q = M boundary to ~5% by Q/M ~ 100, and single runs
  vary widely (run-to-run sigma up to ~30 percentage points).
* For Q/M above ~300 the wasted work is saturated at ~0%, so only a few
  columns are needed to show the plateau.
* Both axes therefore use 1-2-5 log steps: M in 10..10,000 and Q in
  20..100,000, keeping the Experiment 5b envelope (Q <= 100,000). Feasible
  cells require Q > M, giving 75 cells whose Q/M ratios span 2..10,000.
* Each cell is replicated (default 100 seeds); cells report the mean
  percentage of wasted work with the across-seed standard deviation.

By default only the 75 feasible cells with Q > M are simulated (cells with
Q <= M are masked in the heatmaps). Passing ``--full-grid`` additionally
simulates the 45 cells with Q <= M, including the Q = M diagonal, so every
grid position is filled.

Heatmaps put miners M on the x-axis and query pool Q on the y-axis, both
increasing in the conventional direction (large values at the top right).
Infeasible cells (Q <= M) are masked. Red means more wasted work, green
means less.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_uniform_fee_collisions import (  # noqa: E402
    SCHEMA_VERSION,
    SIMULATION_MODE,
    _git_revision,
    analytical_difficulty,
    content_id,
    derive_seed,
    host_environment,
    simulate_race,
)


DEFAULT_TIMINGS = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "exp5_run_all"
    / "timing_calibration.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "exp5c_m_q_heatmap_100rep"
DEFAULT_TARGETS_S = (300.0, 600.0, 900.0)
DEFAULT_REPLICATES = 100
DEFAULT_SEED_ROOT = "poml-exp5c-m-q-heatmap-v1"
MINER_GRID = (10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000)
QUERY_GRID = (20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
RUN_FIELDS = (
    "miners",
    "query_pool_size",
    "q_over_m",
    "target_block_time_s",
    "replicate",
    "run_id",
    "config_id",
    "run_seed",
    "fee",
    "difficulty_hex",
    "status",
    "adoption_time_s",
    "winner_miner_id",
    "winner_query_id",
    "completed_pairs",
    "collision_count",
    "collision_ratio",
    "percentage_wasted_work",
    "partial_attempts_discarded",
    "tied_completions_cancelled",
    "unique_queries_completed",
)
SUMMARY_FIELDS = (
    "target_block_time_s",
    "miners",
    "query_pool_size",
    "q_over_m",
    "replicates",
    "adopted_runs",
    "mean_percentage_wasted_work",
    "std_percentage_wasted_work",
    "min_percentage_wasted_work",
    "max_percentage_wasted_work",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_timing_samples(path: Path) -> tuple[dict[str, Any], list[float]]:
    timing = json.loads(path.read_text())
    if not isinstance(timing, dict) or timing.get("artifact_kind") != "ezkl_attempt_timing_calibration":
        raise ValueError(f"not an Experiment 5 timing calibration: {path}")
    samples = [float(value) for value in timing.get("duration_samples_s", [])]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing calibration must contain finite positive duration samples")
    return timing, samples


def design_grid(
    miner_values: tuple[int, ...] = MINER_GRID,
    query_values: tuple[int, ...] = QUERY_GRID,
    *,
    full_grid: bool = False,
) -> list[dict[str, int | float]]:
    """Return grid cells in row-major (Q, M) order.

    By default only feasible cells with Q > M are returned. With
    ``full_grid=True`` every (M, Q) combination is returned, including the
    Q = M diagonal and the Q < M corner.
    """
    if any(value <= 0 for value in (*miner_values, *query_values)):
        raise ValueError("grid values must be positive")
    if tuple(sorted(miner_values)) != tuple(miner_values) or tuple(sorted(query_values)) != tuple(query_values):
        raise ValueError("grid values must be sorted ascending")
    cells = [
        {
            "miners": miners,
            "query_pool_size": queries,
            "q_over_m": queries / miners,
        }
        for queries in query_values
        for miners in miner_values
        if full_grid or queries > miners
    ]
    if not cells:
        raise ValueError("grid has no cells")
    return cells


def _cell_aggregates(
    rows: list[dict[str, str]],
) -> dict[tuple[float, int, int], dict[str, float]]:
    """Aggregate adopted runs into per-(target, M, Q) wasted-work statistics."""
    values: dict[tuple[float, int, int], list[float]] = {}
    for row in rows:
        percentage = row.get("percentage_wasted_work")
        if row["status"] != "adopted" or percentage is None or percentage == "":
            continue
        key = (
            float(row["target_block_time_s"]),
            int(row["miners"]),
            int(row["query_pool_size"]),
        )
        values.setdefault(key, []).append(float(percentage))

    aggregates: dict[tuple[float, int, int], dict[str, float]] = {}
    for key, cell_values in values.items():
        aggregates[key] = {
            "adopted_runs": len(cell_values),
            "mean": statistics.fmean(cell_values),
            "std": statistics.stdev(cell_values) if len(cell_values) > 1 else 0.0,
            "min": min(cell_values),
            "max": max(cell_values),
        }
    return aggregates


def summarize_cells(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build one summary row per (target, Q, M) grid cell."""
    aggregates = _cell_aggregates(rows)
    summary_rows: list[dict[str, Any]] = []
    for (target, miners, queries) in sorted(aggregates):
        stats = aggregates[(target, miners, queries)]
        summary_rows.append(
            {
                "target_block_time_s": target,
                "miners": miners,
                "query_pool_size": queries,
                "q_over_m": queries / miners,
                "replicates": sum(
                    1
                    for row in rows
                    if float(row["target_block_time_s"]) == target
                    and int(row["miners"]) == miners
                    and int(row["query_pool_size"]) == queries
                ),
                "adopted_runs": stats["adopted_runs"],
                "mean_percentage_wasted_work": stats["mean"],
                "std_percentage_wasted_work": stats["std"],
                "min_percentage_wasted_work": stats["min"],
                "max_percentage_wasted_work": stats["max"],
            }
        )
    return summary_rows


def generate_heatmaps(runs_path: Path, output_dir: Path) -> list[Path]:
    """Generate one annotated M-by-Q wasted-work heatmap per target block time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with runs_path.open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        raise ValueError(f"runs CSV has no rows: {runs_path}")

    aggregates = _cell_aggregates(rows)
    if not aggregates:
        raise ValueError(f"runs CSV has no adopted runs with wasted-work values: {runs_path}")

    miner_values = sorted({key[1] for key in aggregates})
    query_values = sorted({key[2] for key in aggregates})
    targets = sorted({key[0] for key in aggregates})
    miner_index = {value: index for index, value in enumerate(miner_values)}
    query_index = {value: index for index, value in enumerate(query_values)}
    replicates = Counter(
        (float(row["target_block_time_s"]), int(row["miners"]), int(row["query_pool_size"]))
        for row in rows
    )

    def compact(value: int) -> str:
        if value >= 1_000:
            return f"{value // 1_000}k" if value % 1_000 == 0 else f"{value / 1_000:g}k"
        return str(value)

    color_map = plt.get_cmap("RdYlGn_r").copy()
    color_map.set_bad("#dddddd")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for target in targets:
        means = np.full((len(query_values), len(miner_values)), np.nan)
        stds = np.full_like(means, np.nan)
        counts = np.zeros_like(means, dtype=int)
        for (cell_target, miners, queries), stats in aggregates.items():
            if cell_target != target:
                continue
            row_index = query_index[queries]
            column_index = miner_index[miners]
            means[row_index, column_index] = stats["mean"]
            stds[row_index, column_index] = stats["std"]
            counts[row_index, column_index] = replicates[(target, miners, queries)]

        masked = np.ma.masked_invalid(means)
        figure, axis = plt.subplots(figsize=(13.0, 8.2), constrained_layout=True)
        image = axis.imshow(
            masked,
            cmap=color_map,
            vmin=0.0,
            vmax=100.0,
            aspect="auto",
            origin="lower",
        )
        axis.set_xticks(
            range(len(miner_values)),
            [compact(value) for value in miner_values],
            fontsize=13,
        )
        axis.set_yticks(
            range(len(query_values)),
            [compact(value) for value in query_values],
            fontsize=13,
        )
        axis.set_xlabel("Miners M", fontsize=17, labelpad=10)
        axis.set_ylabel("Query pool size Q", fontsize=17, labelpad=10)
        axis.set_title(
            f"Percentage of wasted work, target {target:g} s",
            fontsize=19,
            pad=14,
        )
        for row_index in range(len(query_values)):
            for column_index in range(len(miner_values)):
                if np.isnan(means[row_index, column_index]):
                    continue
                mean = means[row_index, column_index]
                std = stds[row_index, column_index]
                red, green, blue, _ = color_map(mean / 100.0)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column_index,
                    row_index,
                    f"{mean:.0f}%\n\u00b1{std:.0f}%",
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.55 else "white",
                    fontsize=10.5,
                    fontweight="semibold",
                )
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label("Mean percentage of wasted work (%)", fontsize=15)
        colorbar.ax.tick_params(labelsize=13)
        stem = f"wasted_work_heatmap_m_q_target_{target:g}s"
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=300)
            outputs.append(path)
        plt.close(figure)
    return outputs


def run_campaign(
    *,
    timing_path: Path,
    output_dir: Path,
    targets_s: tuple[float, ...] = DEFAULT_TARGETS_S,
    replicates: int = DEFAULT_REPLICATES,
    seed_root: str = DEFAULT_SEED_ROOT,
    miner_values: tuple[int, ...] = MINER_GRID,
    query_values: tuple[int, ...] = QUERY_GRID,
    full_grid: bool = False,
) -> dict[str, Any]:
    """Run Experiment 5c and write campaign, run, summary, and figure artifacts."""
    if not targets_s or any(not math.isfinite(target) or target <= 0 for target in targets_s):
        raise ValueError("target block times must be finite and positive")
    if len(set(targets_s)) != len(targets_s):
        raise ValueError("target block times must be unique")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timing, durations = _load_timing_samples(timing_path)
    mean_attempt_s = statistics.fmean(durations)
    cells = design_grid(miner_values, query_values, full_grid=full_grid)
    rows: list[dict[str, Any]] = []

    total = len(targets_s) * len(cells) * replicates
    completed = 0
    progress_step = max(1, total // 20)
    for target in targets_s:
        for cell in cells:
            miners = int(cell["miners"])
            query_pool_size = int(cell["query_pool_size"])
            difficulty = analytical_difficulty(mean_attempt_s, miners, target)
            for replicate in range(replicates):
                config = {
                    "schema_version": SCHEMA_VERSION,
                    "simulation_mode": SIMULATION_MODE,
                    "experiment": "5c",
                    "miners": miners,
                    "query_pool_size": query_pool_size,
                    "target_block_time_s": target,
                    "replicate": replicate,
                    "uniform_fee": 1,
                    "difficulty_hex": f"0x{difficulty:064x}",
                }
                config_id = content_id(config)
                run_seed = derive_seed(
                    seed_root,
                    "race",
                    miners,
                    query_pool_size,
                    target,
                    replicate,
                )
                run_id = content_id({"config_id": config_id, "seed": run_seed})
                run, _ = simulate_race(
                    miners=miners,
                    query_pool_size=query_pool_size,
                    difficulty=difficulty,
                    duration_samples_s=durations,
                    run_seed=run_seed,
                    config_id=config_id,
                    run_id=run_id,
                    fee=1,
                    retain_details=False,
                )
                collision_ratio = run["collision_ratio"]
                rows.append(
                    {
                        "miners": miners,
                        "query_pool_size": query_pool_size,
                        "q_over_m": cell["q_over_m"],
                        "target_block_time_s": target,
                        "replicate": replicate,
                        "run_id": run_id,
                        "config_id": config_id,
                        "run_seed": run_seed,
                        "fee": 1,
                        "difficulty_hex": f"0x{difficulty:064x}",
                        "status": run["status"],
                        "adoption_time_s": run["adoption_time_s"],
                        "winner_miner_id": run["winner_miner_id"],
                        "winner_query_id": run["winner_query_id"],
                        "completed_pairs": run["completed_pairs"],
                        "collision_count": run["collision_count"],
                        "collision_ratio": collision_ratio,
                        "percentage_wasted_work": (
                            100.0 * float(collision_ratio) if collision_ratio is not None else None
                        ),
                        "partial_attempts_discarded": run["partial_attempts_discarded"],
                        "tied_completions_cancelled": run["tied_completions_cancelled"],
                        "unique_queries_completed": run["unique_queries_completed"],
                    }
                )
                completed += 1
                if completed % progress_step == 0 or completed == total:
                    print(f"[exp5c] {completed:,}/{total:,} races complete", flush=True)

    runs_path = output_dir / "runs.csv"
    with runs_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = summarize_cells(rows)
    with (output_dir / "summary.csv").open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    status_counts = Counter(row["status"] for row in rows)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "uniform_fee_m_q_heatmap_campaign",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_status": "complete" if status_counts.get("no_winner", 0) == 0 else "complete_with_failures",
        "simulation_mode": SIMULATION_MODE,
        "simulator_revision": _git_revision(),
        "environment": host_environment(),
        "design": {
            "miner_grid": list(miner_values),
            "query_grid": list(query_values),
            "feasible_cells": len(cells),
            "constraint": (
                "full grid: all Q x M combinations simulated"
                if full_grid
                else "Q > M; infeasible cells masked in heatmaps"
            ),
            "grid_basis": (
                "Experiment 5b scatter: dynamic regime Q/M in [1, 100]; "
                "wasted work saturated near 0% above Q/M ~ 300"
            ),
            "replicates_per_cell": replicates,
            "cell_statistic": "mean percentage of wasted work with across-seed standard deviation",
            "target_block_times_s": list(targets_s),
            "uniform_fee": 1,
            "attempt_details_retained": False,
            "failed_race_policy": "record in runs.csv and omit from cell aggregates",
            "seed_root": seed_root,
        },
        "timing_artifact": {
            "path": str(timing_path),
            "artifact_id": timing.get("artifact_id"),
            "sha256": _file_sha256(timing_path),
            "sample_count": len(durations),
            "mean_attempt_s": mean_attempt_s,
        },
        "grid": cells,
        "run_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
    }
    campaign["campaign_id"] = content_id(campaign)
    (output_dir / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    generate_heatmaps(runs_path, output_dir / "figures")
    return campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-runs", type=Path, help="regenerate heatmaps from an existing runs CSV")
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS_S))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed-root", default=DEFAULT_SEED_ROOT)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="also simulate cells with Q <= M (Q = M diagonal and Q < M corner)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.plot_runs:
        outputs = generate_heatmaps(args.plot_runs, args.output_dir / "figures")
        print(f"[exp5c] wrote {len(outputs)} heatmap files to {args.output_dir / 'figures'}")
        return
    campaign = run_campaign(
        timing_path=args.timings,
        output_dir=args.output_dir,
        targets_s=tuple(args.targets),
        replicates=args.replicates,
        seed_root=args.seed_root,
        full_grid=args.full_grid,
    )
    print(
        f"[exp5c] complete: {campaign['run_count']:,} races; outputs: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
