#!/usr/bin/env python3
"""Experiment 5b: wasted work across query-to-miner proportions.

Sample shared (Q, M) pairs across logarithmic Q/M bins, run one Experiment 5
race per pair and target block time, and summarize wasted work in heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "exp5b_ratio_scatter"
DEFAULT_TARGETS_S = (300.0, 600.0, 900.0)
DEFAULT_SEED_ROOT = "poml-exp5b-ratio-scatter-v1"
HEATMAP_RATIO_BINS = 6
HEATMAP_MINER_EDGES = (1, 10, 100, 1_000, 100_001)
RUN_FIELDS = (
    "pair_index",
    "ratio_bin",
    "run_id",
    "config_id",
    "run_seed",
    "miners",
    "query_pool_size",
    "q_over_m",
    "target_block_time_s",
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


def _log_uniform_int(rng: random.Random, upper: int) -> int:
    if upper <= 1:
        return 1
    return min(upper, int(math.exp(rng.uniform(0.0, math.log(upper + 1)))))


def sample_pairs(
    *,
    count: int,
    ratio_bins: int,
    max_value: int,
    seed_root: str,
) -> list[dict[str, int | float]]:
    """Sample unique integer (Q, M) pairs, stratified by log(Q/M)."""
    if count <= 0 or ratio_bins <= 0 or ratio_bins > count:
        raise ValueError("pair count must be positive and at least the number of bins")
    if max_value < 2:
        raise ValueError("max value must be at least 2")

    rng = random.Random(derive_seed(seed_root, "pair-design"))
    minimum_ratio = max_value / (max_value - 1)
    log_minimum = math.log(minimum_ratio)
    log_span = math.log(max_value) - log_minimum
    base_count, extra = divmod(count, ratio_bins)
    used: set[tuple[int, int]] = set()
    pairs: list[dict[str, int | float]] = []

    for ratio_bin in range(ratio_bins):
        lower = math.exp(log_minimum + log_span * ratio_bin / ratio_bins)
        upper = math.exp(log_minimum + log_span * (ratio_bin + 1) / ratio_bins)
        needed = base_count + int(ratio_bin < extra)
        accepted = 0
        attempts = 0
        while accepted < needed:
            attempts += 1
            if attempts > max(10_000, needed * 1_000):
                raise RuntimeError(
                    f"could not sample {needed} unique pairs in ratio bin {ratio_bin}"
                )
            target_ratio = math.exp(rng.uniform(math.log(lower), math.log(upper)))
            miners = _log_uniform_int(rng, max(1, int(max_value / target_ratio)))
            query_pool_size = min(
                max_value,
                max(miners + 1, int(round(target_ratio * miners))),
            )
            actual_ratio = query_pool_size / miners
            pair = (query_pool_size, miners)
            if pair in used or not lower <= actual_ratio <= upper:
                continue
            used.add(pair)
            pairs.append(
                {
                    "ratio_bin": ratio_bin,
                    "miners": miners,
                    "query_pool_size": query_pool_size,
                    "q_over_m": actual_ratio,
                }
            )
            accepted += 1

    rng.shuffle(pairs)
    for pair_index, pair in enumerate(pairs):
        pair["pair_index"] = pair_index
    return pairs


def generate_heatmaps(runs_path: Path, output_dir: Path) -> list[Path]:
    """Generate one annotated wasted-work heatmap per target block time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with runs_path.open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        raise ValueError(f"runs CSV has no rows: {runs_path}")

    adopted = [
        row
        for row in rows
        if row["status"] == "adopted" and row["percentage_wasted_work"]
    ]
    if not adopted:
        raise ValueError(f"runs CSV has no adopted runs with wasted-work values: {runs_path}")

    targets = sorted({float(row["target_block_time_s"]) for row in adopted})
    maximum_ratio = max(float(row["q_over_m"]) for row in adopted)
    ratio_upper = 10 ** math.ceil(math.log10(maximum_ratio))
    ratio_edges = np.geomspace(1.0, ratio_upper, HEATMAP_RATIO_BINS + 1)

    def compact(value: float) -> str:
        scale, suffix = (1_000.0, "k") if value >= 1_000 else (1.0, "")
        return f"{value / scale:.3g}{suffix}"

    ratio_labels = [
        f"{compact(lower)}-{compact(upper)}"
        for lower, upper in zip(ratio_edges[:-1], ratio_edges[1:])
    ]
    miner_labels = ("1-9", "10-99", "100-999", "1,000-100,000")
    color_map = plt.get_cmap("RdYlGn_r").copy()
    color_map.set_bad("#dddddd")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for target in targets:
        cell_values: dict[tuple[int, int], list[float]] = {}
        for row in adopted:
            if float(row["target_block_time_s"]) != target:
                continue
            ratio_index = min(
                HEATMAP_RATIO_BINS - 1,
                int(np.searchsorted(ratio_edges, float(row["q_over_m"]), side="right")) - 1,
            )
            miner_index = min(
                len(HEATMAP_MINER_EDGES) - 2,
                int(np.searchsorted(HEATMAP_MINER_EDGES, int(row["miners"]), side="right")) - 1,
            )
            cell_values.setdefault((miner_index, ratio_index), []).append(
                float(row["percentage_wasted_work"])
            )

        matrix = np.full((len(miner_labels), HEATMAP_RATIO_BINS), np.nan)
        for (miner_index, ratio_index), values in cell_values.items():
            matrix[miner_index, ratio_index] = statistics.median(values)

        figure, axis = plt.subplots(figsize=(12.0, 6.4), constrained_layout=True)
        image = axis.imshow(
            matrix,
            cmap=color_map,
            vmin=0.0,
            vmax=100.0,
            aspect="auto",
        )
        axis.set_xticks(range(HEATMAP_RATIO_BINS), ratio_labels, fontsize=12)
        axis.set_yticks(range(len(miner_labels)), miner_labels, fontsize=13)
        axis.set_xlabel("Query-to-miner proportion Q/M", fontsize=16, labelpad=10)
        axis.set_ylabel("Miner count M", fontsize=16, labelpad=10)
        axis.set_title(
            f"Percentage of wasted work, target {target:g} s",
            fontsize=18,
            pad=14,
        )
        for (miner_index, ratio_index), values in cell_values.items():
            value = matrix[miner_index, ratio_index]
            red, green, blue, _ = color_map(value / 100.0)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                ratio_index,
                miner_index,
                f"{value:.1f}%\n(n={len(values)})",
                ha="center",
                va="center",
                color="black" if luminance > 0.55 else "white",
                fontsize=12,
                fontweight="semibold",
            )
        colorbar = figure.colorbar(image, ax=axis, pad=0.025)
        colorbar.set_label("Median percentage of wasted work (%)", fontsize=14)
        colorbar.ax.tick_params(labelsize=12)
        stem = f"wasted_work_heatmap_target_{target:g}s"
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
    pair_count: int = 1000,
    ratio_bins: int = 50,
    max_value: int = 100_000,
    targets_s: tuple[float, ...] = DEFAULT_TARGETS_S,
    seed_root: str = DEFAULT_SEED_ROOT,
) -> dict[str, Any]:
    """Run Experiment 5b and write campaign, run, and figure artifacts."""
    if not targets_s or any(not math.isfinite(target) or target <= 0 for target in targets_s):
        raise ValueError("target block times must be finite and positive")
    if len(set(targets_s)) != len(targets_s):
        raise ValueError("target block times must be unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timing, durations = _load_timing_samples(timing_path)
    mean_attempt_s = statistics.fmean(durations)
    pairs = sample_pairs(
        count=pair_count,
        ratio_bins=ratio_bins,
        max_value=max_value,
        seed_root=seed_root,
    )
    rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets_s, start=1):
        print(
            f"[exp5b] target {target_index}/{len(targets_s)}: {target:g}s; "
            f"running {len(pairs):,} pairs",
            flush=True,
        )
        progress_step = max(1, len(pairs) // 10)
        for pair in pairs:
            pair_index = int(pair["pair_index"])
            miners = int(pair["miners"])
            query_pool_size = int(pair["query_pool_size"])
            difficulty = analytical_difficulty(mean_attempt_s, miners, target)
            config = {
                "schema_version": SCHEMA_VERSION,
                "simulation_mode": SIMULATION_MODE,
                "experiment": "5b",
                "pair_index": pair_index,
                "miners": miners,
                "query_pool_size": query_pool_size,
                "target_block_time_s": target,
                "uniform_fee": 1,
                "difficulty_hex": f"0x{difficulty:064x}",
            }
            config_id = content_id(config)
            run_seed = derive_seed(
                seed_root,
                "race",
                pair_index,
                miners,
                query_pool_size,
                target,
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
            row = {
                "pair_index": pair_index,
                "ratio_bin": pair["ratio_bin"],
                "run_id": run_id,
                "config_id": config_id,
                "run_seed": run_seed,
                "miners": miners,
                "query_pool_size": query_pool_size,
                "q_over_m": pair["q_over_m"],
                "target_block_time_s": target,
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
            rows.append(row)
            if (pair_index + 1) % progress_step == 0 or pair_index + 1 == len(pairs):
                print(
                    f"[exp5b] target {target:g}s: {pair_index + 1:,}/{len(pairs):,} complete",
                    flush=True,
                )

    runs_path = output_dir / "runs.csv"
    with runs_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["status"] for row in rows)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "uniform_fee_ratio_scatter_campaign",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_status": "complete" if status_counts.get("no_winner", 0) == 0 else "complete_with_failures",
        "simulation_mode": SIMULATION_MODE,
        "simulator_revision": _git_revision(),
        "environment": host_environment(),
        "design": {
            "pair_count": pair_count,
            "ratio_bins": ratio_bins,
            "pairs_per_bin": dict(sorted(Counter(int(pair["ratio_bin"]) for pair in pairs).items())),
            "pair_reuse": "same sampled pairs for every target block time",
            "miner_sampling": "log-uniform within each feasible logarithmic Q/M bin",
            "minimum_value": 1,
            "maximum_value": max_value,
            "constraint": "1 <= M < Q <= maximum_value",
            "target_block_times_s": list(targets_s),
            "uniform_fee": 1,
            "attempt_details_retained": False,
            "failed_race_policy": "record in runs.csv and omit from heatmap summaries",
            "seed_root": seed_root,
        },
        "timing_artifact": {
            "path": str(timing_path),
            "artifact_id": timing.get("artifact_id"),
            "sha256": _file_sha256(timing_path),
            "sample_count": len(durations),
            "mean_attempt_s": mean_attempt_s,
        },
        "pair_design": pairs,
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
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--ratio-bins", type=int, default=50)
    parser.add_argument("--max-value", type=int, default=100_000)
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS_S))
    parser.add_argument("--seed-root", default=DEFAULT_SEED_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.plot_runs:
        outputs = generate_heatmaps(args.plot_runs, args.output_dir / "figures")
        print(f"[exp5b] wrote {len(outputs)} heatmap files to {args.output_dir / 'figures'}")
        return
    campaign = run_campaign(
        timing_path=args.timings,
        output_dir=args.output_dir,
        pair_count=args.pairs,
        ratio_bins=args.ratio_bins,
        max_value=args.max_value,
        targets_s=tuple(args.targets),
        seed_root=args.seed_root,
    )
    print(
        f"[exp5b] complete: {campaign['run_count']:,} races; outputs: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()