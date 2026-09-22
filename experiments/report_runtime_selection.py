#!/usr/bin/env python3
"""Report runtime-fit errors and isolate reweighting on identical archived work."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poml_sim.gpt2_work import reference_counts, weighted_cost  # noqa: E402
from poml_sim.lottery import scaled_complexity  # noqa: E402


def historical_reweighting(source, fit, schedule):
    """Change only C on the old completed attempts, holding selections fixed."""
    table = []
    for policy in ("uniform", "profiled-short", "profiled-long", "shortest-prompt"):
        rows = [
            r
            for directory in (source / "cherry_pick").glob("*-" + policy)
            for r in json.loads((directory / "attempts.json").read_text())
            if r["miner"] == 0 and r["logical_status"] == "completed"
        ]
        seconds = sum(r["duration"] for r in rows)
        if not seconds:
            raise ValueError(f"no completed attacker work for {policy}")
        counts = [
            scaled_complexity(
                weighted_cost(
                    reference_counts(r["prompt_length"], r["output_length"], schedule)[
                        "combined"
                    ],
                    fit["weights"],
                ),
                fit["scale"],
            )
            for r in rows
        ]
        table.append(
            {
                "policy": policy,
                "completed_pairs": len(rows),
                "old_rate": sum(r["complexity"] for r in rows) / seconds,
                "reweighted_same_attempts_rate": sum(counts) / seconds,
            }
        )
    for row in table:
        row["old_relative_uniform"] = row["old_rate"] / table[0]["old_rate"]
        row["reweighted_relative_uniform"] = (
            row["reweighted_same_attempts_rate"]
            / table[0]["reweighted_same_attempts_rate"]
        )
    return table


def plot_full_predictions(rows, directory):
    """Show every held-out prediction with common axes and an explicit K key."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("uniform_held_out_seconds", "weighted_held_out_seconds")
    values = [float(r[k]) for r in rows for k in ("actual_seconds", *keys)]
    low, high = min(values) - 5, max(values) + 5
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.4), sharex=True, sharey=True, layout="constrained"
    )
    actual = [float(r["actual_seconds"]) for r in rows]
    for ax, key, title in zip(
        axes, keys, ("Uniform operation weights", "Fitted operation weights")
    ):
        scatter = ax.scatter(
            actual,
            [float(r[key]) for r in rows],
            c=[int(r["output_length"]) for r in rows],
            cmap="viridis",
            s=18,
            alpha=0.8,
        )
        ax.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=1)
        ax.set(
            title=title,
            xlabel="Measured inference + proof (s)",
            xlim=(low, high),
            ylim=(low, high),
        )
    axes[0].set_ylabel("Held-out prediction (s)")
    fig.colorbar(
        scatter,
        ax=axes,
        label="Realised output length K",
        ticks=[8, 16, 24, 32],
        shrink=0.9,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"runtime_predictions.{suffix}", dpi=180)
    plt.close(fig)


def effective_work_rates(directory):
    """Include time spent on interrupted attempts in each miner's exposure."""
    table = []
    for policy in ("uniform", "profiled-short", "profiled-long", "shortest-prompt"):
        summaries = [
            json.loads(path.read_text())
            for path in (directory / "cherry_pick").glob(f"*-{policy}/summary.json")
        ]
        seconds = sum(row["virtual_chain_seconds"] for row in summaries)
        # Each miner is continuously busy in this zero-delay event model. The
        # chain clock therefore includes its time lost to block-boundary cancellation.
        tickets = sum(
            row["attacker_tickets_per_chain_second"] * row["virtual_chain_seconds"]
            for row in summaries
        )
        completed_seconds = sum(
            row["attacker_tickets_per_chain_second"]
            * row["virtual_chain_seconds"]
            / row["attacker_tickets_per_busy_second"]
            for row in summaries
        )
        table.append(
            {
                "policy": policy,
                "completed_tickets_per_chain_second": tickets / seconds,
                "completed_time_fraction": completed_seconds / seconds,
                "canceled_time_fraction": 1 - completed_seconds / seconds,
            }
        )
    for row in table:
        row["effective_ticket_rate_relative_to_uniform"] = (
            row["completed_tickets_per_chain_second"]
            / table[0]["completed_tickets_per_chain_second"]
        )
    return table


def main(argv=None):
    """Write supplementary shape/control diagnostics without rerunning chains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    directory = args.input.resolve()
    manifest = json.loads((directory / "campaign.json").read_text())
    fit = json.loads((directory / "runtime_weights.json").read_text())
    schedule = json.loads(
        (directory / "source_snapshot/reference_schedule.json").read_text()
    )
    with (directory / "validation_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_length"]), int(row["output_length"])].append(row)
    shapes = []
    for (n, k), items in sorted(grouped.items()):
        shapes.append(
            {
                "N": n,
                "K": k,
                "count": len(items),
                "mean_measured_seconds": sum(float(r["actual_seconds"]) for r in items)
                / len(items),
                "mean_fitted_seconds": sum(
                    float(r["weighted_fitted_seconds"]) for r in items
                )
                / len(items),
                "held_out_MAPE_percent": sum(
                    abs(
                        float(r["weighted_held_out_seconds"])
                        / float(r["actual_seconds"])
                        - 1
                    )
                    for r in items
                )
                * 100
                / len(items),
            }
        )
    table = historical_reweighting(Path(manifest["source"]), fit, schedule)
    for name, data in (
        ("shape_diagnostics", shapes),
        (
            "historical_attempt_reweighting",
            {
                "notice": "Counterfactual costs on fixed historical attempts, not new mining outcomes.",
                "rows": table,
            },
        ),
    ):
        (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    plot_full_predictions(rows, directory)
    if (directory / "COMPLETED").exists():
        rates = {
            "uniform_schedule": effective_work_rates(Path(manifest["source"])),
            "fitted_schedule": effective_work_rates(directory),
        }
        (directory / "effective_work_rates.json").write_text(
            json.dumps(rates, indent=2) + "\n"
        )
        comparison = json.loads((directory / "comparison.json").read_text())
        selected = {(r["schedule"], r["policy"]): r for r in comparison}
        old_long = selected["uniform", "profiled-long"]
        new_long = selected["fitted", "profiled-long"]
        short = selected["fitted", "profiled-short"]
        by_policy = {r["policy"]: r for r in rates["fitted_schedule"]}
        section = "\n## Interpretation\n"
        report = (directory / "report.md").read_text().split(section)[0]
        report += section + "\n"
        report += (
            "Reweighting removes most of the completed-work throughput imbalance. "
            f"Profiled-long's excess ticket rate falls from {100 * (old_long['ticket_rate_relative_to_uniform'] - 1):.2f}% "
            f"to {100 * (new_long['ticket_rate_relative_to_uniform'] - 1):.2f}%. "
            "Holding the historical selections fixed gives a separate counterfactual "
            "in `historical_attempt_reweighting.json`.\n\n"
            "Including time spent on canceled attempts, the fitted effective ticket-rate "
            f"differences relative to uniform are {100 * (by_policy['profiled-short']['effective_ticket_rate_relative_to_uniform'] - 1):+.2f}% "
            f"for profiled-short, {100 * (by_policy['profiled-long']['effective_ticket_rate_relative_to_uniform'] - 1):+.2f}% "
            f"for profiled-long, and {100 * (by_policy['shortest-prompt']['effective_ticket_rate_relative_to_uniform'] - 1):+.2f}% "
            "for shortest-prompt. Short jobs lose less time to interruption; weighting alone "
            "does not remove completion-time or cancellation effects.\n\n"
            f"Profiled-short still wins {100 * short['attacker_block_share']:.2f}% of blocks. "
            f"Its pooled blocks/hour ratio to uniform is {short['block_yield_relative_to_uniform']:.4f}; "
            "the seed-bootstrap interval is retained in `cherry_pick/policy_metrics.csv`. "
            "Do not conclude that all selection advantages vanish. Only three seed-level chains "
            "per policy were run; block outcomes remain noisy, and the bootstrap has very few independent units. "
            "The result supports better empirical work calibration for this workload, not universal policy independence.\n"
        )
        (directory / "report.md").write_text(report)
    snapshot = directory / "source_snapshot/postprocessing"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), snapshot / Path(__file__).name)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
