"""Paper duplicate-work grid, with first-completion complexity credit."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.experiments import calibrate_difficulty, load_measurements, simulate_race, summarize

MINERS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
QUERIES = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)


def main(argv=None):
    """Save all trials, including exhaustion, and conditional mean/SD heatmaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--measurements", type=Path)
    source.add_argument("--smoke", action="store_true")
    parser.add_argument("--miners", default=",".join(map(str, MINERS)))
    parser.add_argument("--queries", default=",".join(map(str, QUERIES)))
    parser.add_argument("--targets", default="300,600,900")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/wasted-work"))
    args = parser.parse_args(argv)
    miners, queries, targets = (
        [int(x) for x in value.split(",")] for value in (args.miners, args.queries, args.targets)
    )
    if min(*miners, *queries, *targets, args.repeats) < 1:
        parser.error("grid values and repeats must be positive")
    data, samples = (
        ({"backend": "synthetic-smoke"}, [(60.0, 100), (90.0, 160)])
        if args.smoke
        else load_measurements(args.measurements)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists() or (args.output / "runs.csv").exists():
        parser.error("choose a fresh output directory")
    rows, cells = [], []
    for target in targets:
        for m in miners:
            difficulty = calibrate_difficulty(samples, m, target)
            for q in queries:
                runs = []
                for repeat in range(args.repeats):
                    seed = f"{args.seed}:{target}:{m}:{q}:{repeat}"
                    row = {
                        "target": target,
                        "miners": m,
                        "queries": q,
                        "repeat": repeat,
                        "seed": seed,
                        "difficulty": str(difficulty),
                        **simulate_race(
                            samples, miners=m, queries=q, difficulty=difficulty, seed=seed
                        ),
                    }
                    runs.append(row)
                    rows.append(row)
                stats = summarize(r["wasted_work_fraction"] for r in runs if r["adopted"])
                cells.append(
                    {
                        "target": target,
                        "miners": m,
                        "queries": q,
                        **stats,
                        "exhausted": sum(not r["adopted"] for r in runs),
                    }
                )
        print(f"Completed target {target}s", flush=True)
    for name, records in (("runs", rows), ("cells", cells)):
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "poml-wasted-work-1",
                "source": data["backend"],
                "mode": "synthetic-smoke" if args.smoke else "measured-pair-replay",
                "miners": miners,
                "queries": queries,
                "targets": targets,
                "repeats": args.repeats,
                "seed": args.seed,
                "metric": "(all completed C - first completion C per qid) / all completed C",
                "exhaustion": "retained in runs; excluded from adopted-race means",
            },
            indent=2,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for target in targets:
        matrix = np.full((len(miners), len(queries)), np.nan)
        for cell in cells:
            if cell["target"] == target and cell["mean"] is not None:
                matrix[miners.index(cell["miners"]), queries.index(cell["queries"])] = (
                    100 * cell["mean"]
                )
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto")
        ax.set(
            xticks=range(len(queries)),
            xticklabels=queries,
            yticks=range(len(miners)),
            yticklabels=miners,
            xlabel="Query pool Q",
            ylabel="Miners M",
            title=f"Completed duplicate work, target {target}s",
        )
        for cell in cells:
            if cell["target"] == target and cell["mean"] is not None:
                ax.text(
                    queries.index(cell["queries"]),
                    miners.index(cell["miners"]),
                    f"{100 * cell['mean']:.1f}\n±{100 * cell['stdev']:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
        fig.colorbar(im, ax=ax, label="Wasted work (%)")
        fig.tight_layout()
        fig.savefig(args.output / f"waste-{target}s.pdf")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
