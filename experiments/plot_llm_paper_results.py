#!/usr/bin/env python3
"""Render the paper's three experiments from completed archived runs only.

Experiments 1 and 3 use the varied-K unit-weight campaign; Experiment 2 compares
that campaign with its runtime-weighted rerun. No mining or proving is invoked.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICIES = ("uniform", "profiled-short", "profiled-long", "shortest-prompt")
LABELS = ("Uniform", "Profiled short", "Profiled long", "Short prompt")


def read_json(path):
    """Read one archived JSON object."""
    return json.loads(path.read_text())


def read_csv(path):
    """Read a CSV while preserving explicit type conversion at its consumer."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def distribution(values):
    """Summarize observations with a sample, rather than population, SD."""
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "cv": statistics.stdev(values) / statistics.fmean(values),
    }


def selection_statistics(source):
    """Pool exposure across seeds and preserve the archived paired-yield intervals."""
    rows = read_csv(source / "cherry_pick/per_seed_metrics.csv")
    intervals = {
        r["policy"]: r for r in read_csv(source / "cherry_pick/policy_metrics.csv")
    }
    expected = read_json(source / "campaign.json")
    result = {}
    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        if len(selected) != expected["seeds"] or len(
            {r["seed"] for r in selected}
        ) != len(selected):
            raise ValueError("missing or duplicated independent selection seeds")
        if any(int(r["blocks"]) != expected["selection_blocks"] for r in selected):
            raise ValueError("selection chain length differs from manifest")
        seconds = sum(float(r["virtual_chain_seconds"]) for r in selected)
        tickets = sum(
            float(r["attacker_tickets_per_chain_second"])
            * float(r["virtual_chain_seconds"])
            for r in selected
        )
        completed_seconds = sum(
            float(r["attacker_tickets_per_chain_second"])
            * float(r["virtual_chain_seconds"])
            / float(r["attacker_tickets_per_busy_second"])
            for r in selected
        )
        blocks = sum(int(r["blocks"]) for r in selected)
        wins = sum(int(r["attacker_blocks"]) for r in selected)
        item = intervals[policy]
        result[policy] = {
            "blocks": blocks,
            "wins": wins,
            "share": wins / blocks,
            "tickets_per_completed_second": tickets / completed_seconds,
            "tickets_per_chain_second": tickets / seconds,
            "canceled_time_fraction": 1 - completed_seconds / seconds,
            "blocks_per_hour": 3600 * wins / seconds,
            "relative_yield": float(item["relative_block_yield"]),
            "relative_yield_ci95": [
                float(item["relative_block_yield_ci95_low"]),
                float(item["relative_block_yield_ci95_high"]),
            ],
            "block_share_ci95": [
                float(item["attacker_block_share_ci95_low"]),
                float(item["attacker_block_share_ci95_high"]),
            ],
        }
    baseline = result["uniform"]
    for item in result.values():
        item["ticket_rate_ratio"] = (
            item["tickets_per_completed_second"]
            / baseline["tickets_per_completed_second"]
        )
        item["effective_rate_ratio"] = (
            item["tickets_per_chain_second"] / baseline["tickets_per_chain_second"]
        )
        if not np.isclose(
            item["blocks_per_hour"] / baseline["blocks_per_hour"],
            item["relative_yield"],
        ):
            raise ValueError("pooled yield disagrees with archived result")
    return result


def collision_statistics(cells, runs, config):
    """Verify all grid cells, including failed trials, before computing cell means."""
    expected = {
        (float(t), m, q)
        for t in config["targets"]
        for m in config["miner_grid"]
        for q in config["pool_grid"]
        if config.get("full_grid", False) or q > m
    }

    def key(row):
        return (
            float(row["target_block_time"]),
            int(row["miners"]),
            int(row["pool_size"]),
        )

    actual = [key(r) for r in cells]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("collision grid has missing or duplicate cells")
    grouped = defaultdict(list)
    for row in runs:
        if row["lottery"] != "aggregate" or key(row) not in expected:
            raise ValueError("unexpected collision lottery or grid cell")
        grouped[key(row)].append(row)
    for cell in cells:
        trials = grouped[key(cell)]
        adopted = [r for r in trials if r["adopted"] == "True"]
        if (
            len(trials) != config["collision_seeds"]
            or len(adopted) != int(cell["adopted_races"])
            or len(trials) - len(adopted) != int(cell["failed_races"])
            or len({r["seed"] for r in trials}) != len(trials)
            or int(cell["seeds"]) != len(trials)
        ):
            raise ValueError("collision adoption counts disagree")
        if any(r["adopted"] not in ("True", "False") for r in trials):
            raise ValueError("invalid collision adoption status")
        values = []
        for row in adopted:
            total, collision = (
                int(row[k])
                for k in (
                    "completed_complexity",
                    "collision_complexity",
                )
            )
            response = int(row.get("response_complexity", 0))
            if (
                not 0 <= collision <= total
                or not 0 <= response <= total - collision
                or total <= 0
            ):
                raise ValueError("invalid response/collision accounting")
            if config.get("collision_metric") == "first_completion":
                if int(row["first_completed_complexity"]) + collision != total:
                    raise ValueError("first/duplicate complexity partition disagrees")
                if int(row["unique_queries"]) + int(row["duplicate_pairs"]) != int(
                    row["completed_pairs"]
                ):
                    raise ValueError("first/duplicate query counts disagree")
            value = 100 * collision / total
            if not np.isclose(value, float(row["wasted_work_pct"])):
                raise ValueError("waste includes work other than completed collisions")
            values.append(value)
        if not values:
            if (
                cell["mean_wasted_work_pct"] != ""
                or cell["stdev_wasted_work_pct"] != ""
            ):
                raise ValueError("exhausted cell must have no mean or SD")
            continue
        if not np.isclose(
            statistics.fmean(values), float(cell["mean_wasted_work_pct"])
        ):
            raise ValueError("collision mean disagrees with adopted trials")
        if len(values) == 1:
            if cell["stdev_wasted_work_pct"] != "":
                raise ValueError("one adopted trial has no sample SD")
        elif not np.isclose(
            statistics.stdev(values), float(cell["stdev_wasted_work_pct"])
        ):
            raise ValueError("collision SD disagrees with adopted trials")
    result = {}
    for target in config["targets"]:
        selected = [r for r in cells if float(r["target_block_time"]) == target]
        means = [
            float(r["mean_wasted_work_pct"])
            for r in selected
            if int(r["adopted_races"])
        ]
        result[f"{target:g}"] = {
            "cells": len(selected),
            "trials": sum(int(r["seeds"]) for r in selected),
            "adopted": sum(int(r["adopted_races"]) for r in selected),
            "failed": sum(int(r["failed_races"]) for r in selected),
            "mean_of_cell_means_pct": statistics.fmean(means) if means else None,
            "failed_cells": [r for r in selected if int(r["failed_races"])],
        }
    return result


def save(fig, output, name):
    """Keep vector figures for TeX and PNG previews from the same render."""
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def liveness_plot(blocks, pow_times, attempts, output):
    """Draw block intervals and complexity-duration correlation panels."""
    completed = [r for r in attempts if r["logical_status"] == "completed"]
    complexity = np.asarray([float(r["complexity"]) for r in completed])
    duration = np.asarray([float(r["duration"]) for r in completed])
    slope, intercept = np.polyfit(complexity, duration, 1)
    predicted = slope * complexity + intercept
    residuals = duration - predicted
    mae = float(np.mean(np.abs(residuals)))

    options = dict(
        showfliers=False,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#d5e2f0", "edgecolor": "#355b80"},
        medianprops={"color": "#182f45", "linewidth": 1.3},
        meanprops={"color": "#be552b", "linestyle": "--"},
    )
    fig, ax = plt.subplots(figsize=(2.45, 2.35), layout="constrained")
    artists = ax.boxplot([blocks, pow_times], tick_labels=["PoML", "PoW"], **options)
    ax.axhline(300, color="gray", linestyle=":", linewidth=1, label="300 s target")
    # With outlier points omitted, scale to the visible whiskers; the table
    # retains the complete sample extrema.
    maximum_whisker = max(max(line.get_ydata()) for line in artists["whiskers"])
    ax.set(ylabel="Block interval (s)", ylim=(0, max(300, maximum_whisker) * 1.08))
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.15)
    save(fig, output, "llm_replay_boxplots")

    fig, ax = plt.subplots(figsize=(2.45, 2.35), layout="constrained")
    ax.scatter(complexity, duration, s=8, alpha=0.45, color="#285d8f")
    order = np.argsort(complexity)
    ax.plot(
        complexity[order],
        predicted[order],
        color="#be552b",
        linewidth=1.2,
        label=f"OLS fit; MAE={mae:.2f} s",
    )
    ax.set(
        xlabel="Calculated complexity $C(N,K)$",
        ylabel="Inference-proof time (s)",
    )
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(axis="y", alpha=0.15)
    save(fig, output, "llm_replay_complexity_time")


def selection_plot(before, after, output):
    """Compare work-rate calibration and block yield without hiding residual effects."""
    fig, axes = plt.subplots(
        1, 2, figsize=(5.0, 2.65), layout="constrained", sharey=True
    )
    positions = np.arange(4)
    for values, color, marker, offset, label in (
        (before, "#ad5830", "s", -0.13, "Unit weights"),
        (after, "#285d8f", "o", 0.13, "Fitted weights"),
    ):
        axes[0].plot(
            [values[p]["ticket_rate_ratio"] for p in POLICIES],
            positions + offset,
            linestyle="none",
            marker=marker,
            markersize=4,
            color=color,
            label=label,
        )
        for i, policy in enumerate(POLICIES):
            item = values[policy]
            low, high = item["relative_yield_ci95"]
            axes[1].errorbar(
                item["relative_yield"],
                i + offset,
                xerr=[[item["relative_yield"] - low], [high - item["relative_yield"]]],
                color=color,
                marker=marker,
                markersize=4,
                capsize=2,
            )
    axes[0].set(
        yticks=positions,
        yticklabels=LABELS,
        xlabel="Completed-ticket rate\nrelative to uniform",
        xlim=(0.80, 1.13),
    )
    axes[0].invert_yaxis()
    # Keep the legend outside the data area, especially the shortest-prompt row.
    fig.legend(frameon=False, fontsize=7, loc="outside upper center", ncol=2)
    axes[1].set(
        xlabel="Blocks/hour relative to uniform\n(paired-seed bootstrap 95% CI)",
        xlim=(0.45, 2.05),
    )
    for ax in axes:
        ax.axvline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.15)
    save(fig, output, "llm_runtime_selection")


def fit_plot(rows, output):
    """Show all nested held-out predictions, with no clipped low-cost observations."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7), layout="constrained")
    actual = [float(r["actual_seconds"]) for r in rows]
    for field, color, marker, label in (
        ("uniform_held_out_seconds", "#ad5830", "s", "Unit weights"),
        ("weighted_held_out_seconds", "#285d8f", "o", "Fitted weights"),
    ):
        ax.scatter(
            actual,
            [float(r[field]) for r in rows],
            color=color,
            marker=marker,
            s=12,
            alpha=0.7,
            label=label,
        )
    ax.plot([35, 125], [35, 125], color="gray", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="Measured inference + proof (s)",
        ylabel="Held-out prediction (s)",
        xlim=(60, 115),
        ylim=(35, 125),
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, output, "llm_runtime_fit")


def collision_plots(cells, config, output):
    """Match the original red/yellow/green grid with mean and sample SD labels."""
    miners, pools = config["miner_grid"], config["pool_grid"]
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#dddddd")
    for target in config["targets"]:
        rows = [r for r in cells if float(r["target_block_time"]) == target]
        matrix = np.full((len(pools), len(miners)), np.nan)
        for row in rows:
            matrix[
                pools.index(int(row["pool_size"])), miners.index(int(row["miners"]))
            ] = (
                float(row["mean_wasted_work_pct"])
                if int(row["adopted_races"])
                else np.nan
            )
        fig, ax = plt.subplots(figsize=(13.0, 8.2), layout="constrained")
        heatmap = ax.imshow(
            np.ma.masked_invalid(matrix),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=100,
        )
        miner_labels = [str(v) if v < 1000 else f"{v // 1000}k" for v in miners]
        pool_labels = [str(v) if v < 1000 else f"{v // 1000}k" for v in pools]
        ax.set_xticks(range(len(miners)), miner_labels, fontsize=13)
        ax.set_yticks(range(len(pools)), pool_labels, fontsize=13)
        ax.set_xlabel("Miners M", fontsize=17, labelpad=10)
        ax.set_ylabel("Query pool size Q", fontsize=17, labelpad=10)
        ax.set_title(
            f"Percentage of wasted complexity, target {target:g} s", fontsize=19, pad=14
        )
        for row in rows:
            x, y = miners.index(int(row["miners"])), pools.index(int(row["pool_size"]))
            if not int(row["adopted_races"]):
                ax.text(x, y, "N/A*", ha="center", va="center", fontsize=10.5)
                continue
            value = float(row["mean_wasted_work_pct"])
            sd = row["stdev_wasted_work_pct"]
            sd_label = f"±{float(sd):.0f}%" if sd not in (None, "") else "SD N/A"
            red, green, blue, _ = cmap(value / 100)
            color = (
                "black"
                if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.55
                else "white"
            )
            ax.text(
                x,
                y,
                f"{value:.0f}%\n{sd_label}",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="semibold",
                color=color,
            )
            if int(row["failed_races"]):
                ax.text(
                    x + 0.38,
                    y + 0.25,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                )
        bar = fig.colorbar(heatmap, ax=ax, pad=0.02)
        bar.set_label("Mean percentage of wasted complexity (%)", fontsize=15)
        bar.ax.tick_params(labelsize=13)
        save(fig, output, f"llm_collision_{target:g}s")


def resolve_collision_source(requested, baseline, output):
    """Retain the output's selected collision campaign across figure rebuilds."""
    saved_report = output / "llm_uniform_fee_statistics.json"
    if requested is not None:
        source = requested.resolve()
    elif saved_report.exists():
        # A liveness-only change must not silently replace the full grid with
        # the baseline's older Q > M experiment when --collisions is omitted.
        source = Path(read_json(saved_report)["source"]).resolve()
    else:
        source = baseline / "wasted_work"
    if not (source / "COMPLETED").exists():
        raise ValueError(f"unfinished collision source: {source}")
    return source


def write_collision_report(source, config, stats, output):
    """Persist the selected source and keep combined figure metadata consistent."""
    manifest = "campaign.json" if (source / "campaign.json").exists() else "method.json"
    result = {
        "source": str(source),
        "config": config,
        "collisions": stats,
        "input_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (manifest, "cells.csv", "runs.csv")
        },
        "figure_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "llm_uniform_fee_statistics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    combined_path = output / "llm_paper_figure_statistics.json"
    if combined_path.exists():
        combined = read_json(combined_path)
        previous_source = combined.get("collision_source")
        hashes = combined.get("input_sha256", {})
        if previous_source:
            hashes = {
                name: value
                for name, value in hashes.items()
                if Path(name).parent != Path(previous_source)
            }
        hashes.update(
            {
                str(source / name): value
                for name, value in result["input_sha256"].items()
            }
        )
        combined.update(
            collision_source=str(source),
            collisions=stats,
            input_sha256=hashes,
            collision_figure_provenance="llm_uniform_fee_statistics.json",
            collision_figure_source_sha256=result["figure_source_sha256"],
            complexity_scope=(
                "E1: scaled unit weights; uniform-fee grid: raw unit weights; E2: unit and fitted weights"
                if config.get("collision_metric") == "first_completion"
                else "E1/E3: scaled unit weights; E2: unit and fitted weights"
            ),
        )
        combined_path.write_text(json.dumps(combined, indent=2) + "\n")


def main(argv=None):
    """Validate archived statistics, draw paper figures and save their provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--weighted", type=Path)
    parser.add_argument("--liveness-source", type=Path)
    parser.add_argument(
        "--collisions",
        type=Path,
        help="collision campaign; defaults to the output's saved source, then the baseline archive",
    )
    parser.add_argument(
        "--collisions-only",
        action="store_true",
        help="update only collision plots and their statistics",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.collisions_only:
        if not args.collisions:
            parser.error("--collisions-only requires --collisions")
        source = resolve_collision_source(args.collisions, None, args.output)
        config = read_json(source / "campaign.json")
        cells = read_csv(source / "cells.csv")
        stats = collision_statistics(cells, read_csv(source / "runs.csv"), config)
        args.output.mkdir(parents=True, exist_ok=True)
        collision_plots(cells, config, args.output)
        write_collision_report(source, config, stats, args.output)
        print(json.dumps(stats, indent=2))
        return
    if not args.baseline or not args.weighted:
        parser.error("--baseline and --weighted are required unless --collisions-only")
    baseline, weighted, output = (
        args.baseline.resolve(),
        args.weighted.resolve(),
        args.output.resolve(),
    )
    liveness_source = (
        args.liveness_source.resolve() if args.liveness_source else baseline
    )
    collision_source = resolve_collision_source(args.collisions, baseline, output)
    for source, stages in (
        (baseline, ("", "cherry_pick")),
        (weighted, ("", "cherry_pick")),
    ):
        for stage in stages:
            if not (source / stage / "COMPLETED").exists():
                raise ValueError(f"unfinished source: {source / stage}")
    if not (liveness_source / "liveness" / "COMPLETED").exists():
        raise ValueError(f"unfinished liveness source: {liveness_source / 'liveness'}")
    config = read_json(baseline / "campaign.json")
    blocks = [
        r["block_time"] for r in read_json(liveness_source / "liveness/blocks.json")
    ]
    attempts = read_json(liveness_source / "liveness/attempts.json")
    old_pow = read_csv(Path(__file__).parent / "results/exp1_pow_blocks.csv")
    pow_times = [float(r["time_since_last_block_s"]) for r in old_pow]
    summary = read_json(liveness_source / "liveness/summary.json")
    if (
        len(blocks) != summary["blocks"]
        or len([r for r in attempts if r["logical_status"] == "completed"])
        != summary["logical_completed_attempts"]
    ):
        raise ValueError("liveness sample sizes disagree")
    if not np.isclose(statistics.mean(blocks), summary["block_intervals"]["mean"]):
        raise ValueError("liveness mean disagrees")
    before, after = selection_statistics(baseline), selection_statistics(weighted)
    collision_manifest = collision_source / "campaign.json"
    collision_config = (
        read_json(collision_manifest)
        if collision_manifest.exists()
        else read_json(collision_source.parent / "campaign.json")
    )
    cells = read_csv(collision_source / "cells.csv")
    collisions = collision_statistics(
        cells, read_csv(collision_source / "runs.csv"), collision_config
    )
    timings = [
        json.loads(line)
        for line in (baseline / "preparation/timings.jsonl").read_text().splitlines()
    ]
    pool = read_json(baseline / "preparation/pool.json")
    fit = read_json(weighted / "runtime_weights.json")
    result = {
        "baseline": str(baseline),
        "weighted": str(weighted),
        "collision_source": str(collision_source),
        "complexity_scope": (
            "E1: scaled unit weights; uniform-fee grid: raw unit weights; E2: unit and fitted weights"
            if collision_config.get("collision_metric") == "first_completion"
            else "E1/E3: scaled unit weights; E2: unit and fitted weights"
        ),
        "pool": {
            "queries": len(pool),
            "distinct_prompts": len({r["prompt_sha256"] for r in pool}),
            "pairs": len(timings),
            "realised_K_counts": dict(Counter(r["output_length"] for r in timings)),
            "pair_time": distribution([r["duration"] for r in timings]),
        },
        "liveness": {
            "poml": distribution(blocks),
            "pow_model": distribution(pow_times),
            "completed_pairs": distribution(
                [r["duration"] for r in attempts if r["logical_status"] == "completed"]
            ),
        },
        "selection": {"unit": before, "fitted": after},
        "fit": fit,
        "collisions": collisions,
        "figure_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    provenance = [
        baseline / name
        for name in (
            "campaign.json",
            "environment.json",
            "preparation/timings.jsonl",
            "preparation/pool.json",
            "preparation/calibration.json",
            "liveness/pow_simulated_intervals.json",
        )
    ]
    if liveness_source != baseline:
        provenance += [
            liveness_source / name
            for name in (
                "preparation/calibration.json",
                "liveness/blocks.json",
                "liveness/attempts.json",
                "liveness/summary.json",
            )
        ]
    provenance += [collision_source / name for name in ("cells.csv", "runs.csv")]
    provenance.append(
        collision_source
        / ("campaign.json" if collision_manifest.exists() else "method.json")
    )
    provenance += [
        source / "cherry_pick" / name
        for source in (baseline, weighted)
        for name in ("per_seed_metrics.csv", "policy_metrics.csv")
    ]
    provenance += [
        weighted / name
        for name in (
            "campaign.json",
            "runtime_weights.json",
            "validation_predictions.csv",
            "preparation/calibration.json",
        )
    ]
    result["input_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance
    }
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    liveness_plot(blocks, pow_times, attempts, output)
    selection_plot(before, after, output)
    fit_plot(read_csv(weighted / "validation_predictions.csv"), output)
    collision_plots(cells, collision_config, output)
    (output / "llm_paper_figure_statistics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_collision_report(collision_source, collision_config, collisions, output)
    print(
        json.dumps({"liveness": result["liveness"], "collisions": collisions}, indent=2)
    )


if __name__ == "__main__":
    main()
