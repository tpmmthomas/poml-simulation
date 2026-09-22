"""Appendix GPT-2 utility and independently decoded activation separation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.benchmark_data import benchmark_examples
from poml_sim.gpt2_compatibility import (
    MODEL_REVISION,
    load_model,
    embedding_std,
    utility,
    utility_summary,
    compare_trajectories,
)


def main(argv=None):
    """Run only the two GPT-2 compatibility experiments present in the appendix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("utility", "trace"))
    parser.add_argument("--manifest", type=Path, help="reuse exact tokenized benchmark examples")
    parser.add_argument("--examples", type=int, default=100, help="examples per task")
    parser.add_argument("--alphas", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--steps", default="0,1,4,8,16,32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.examples, args.pairs, args.replicates) < 1:
        parser.error("positive sample, pair and replicate counts required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        parser.error("choose a fresh output directory")
    model, tokenizer = load_model(args.device)
    alphas = [float(v) for v in args.alphas.split(",")]
    steps = tuple(int(v) for v in args.steps.split(","))
    tasks = ["wikitext2", "hellaswag", "piqa", "arc_easy"]
    if args.mode == "trace":
        tasks.extend(["lambada", "resisting_correction"])
    examples = (
        json.loads(args.manifest.read_text())
        if args.manifest
        else [
            row
            for task in tasks
            for row in benchmark_examples(tokenizer, task, count=args.examples, seed=args.seed)
        ]
    )
    (args.output / "prompts.json").write_text(json.dumps(examples, indent=2) + "\n")
    rows = []
    for index, example in enumerate(examples):
        if args.mode == "utility":
            new = utility(
                model, [example], alphas=alphas, replicates=args.replicates, seed=args.seed
            )
        else:
            new = [
                row
                for alpha in alphas
                for pair in range(args.pairs)
                for row in compare_trajectories(
                    model, example, alpha=alpha, pair=pair, steps=steps, seed=args.seed
                )
            ]
        rows.extend(new)
        with (args.output / "results.jsonl").open("a") as stream:
            for row in new:
                stream.write(json.dumps(row) + "\n")
        print(f"Completed {index + 1}/{len(examples)}", flush=True)
    if args.mode == "utility":
        summary = utility_summary(rows)
    else:
        groups = {}
        for row in rows:
            groups.setdefault((row["alpha"], row["boundary"]), []).append(row["linf"])
        summary = [
            {
                "alpha": alpha,
                "boundary": boundary,
                "minimum_linf": min(values),
                "comparisons": len(values),
            }
            for (alpha, boundary), values in groups.items()
        ]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        boundaries = [
            "embedding",
            *[f"{i}{kind}" for i in range(1, model.config.n_layer + 1) for kind in ("A", "F")],
            "norm",
            "logits",
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        for alpha in alphas:
            lookup = {r["boundary"]: r["minimum_linf"] for r in summary if r["alpha"] == alpha}
            ax.semilogy(
                range(len(boundaries)),
                [lookup[b] for b in boundaries],
                marker=".",
                label=f"α={alpha}",
            )
        ax.axhline(1e-3, color="black", linestyle="--", label="comparison bin Δ=0.001")
        ax.set_xticks(range(len(boundaries)), boundaries, rotation=60)
        ax.set_ylabel("Minimum L∞ distance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "layer-distances.pdf")
        plt.close(fig)
    metadata = {
        "mode": args.mode,
        "revision": MODEL_REVISION,
        "embedding_std": embedding_std(model),
        "alphas": alphas,
        "pairs": args.pairs,
        "replicates": args.replicates,
        "steps": steps,
        "seed": args.seed,
        "examples": len(examples),
        "trace_method": "independent autoregressive continuations, whole context matrices, EOS-aware",
        "summary": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
