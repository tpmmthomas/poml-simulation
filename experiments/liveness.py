"""Paper liveness experiment: four miners, 256 queries, 50 blocks, 300s target."""

import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.cli import parser, make_backend
from poml_sim.experiments import calibrate_difficulty, load_measurements, simulate_race, summarize
from poml_sim.simulation import run_chain
from poml_sim.system import PoMLSystem


def main(argv=None):
    """Compare fresh or replayed PoML against actual hashing or a Poisson control."""
    argument_parser = parser()
    argument_parser.description = __doc__
    argument_parser.set_defaults(
        blocks=50,
        miners=4,
        queries=256,
        backend="gpt2",
        output=Path("experiments/results/liveness"),
    )
    source = argument_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--measurements", type=Path)
    source.add_argument("--smoke", action="store_true")
    argument_parser.add_argument("--execution", choices=("fresh", "replay"), default="replay")
    argument_parser.add_argument("--pow-mode", choices=("hash", "poisson"), default="hash")
    argument_parser.add_argument("--target", type=float, default=300)
    args = argument_parser.parse_args(argv)
    if args.blocks < 1 or (args.smoke and args.execution == "fresh"):
        argument_parser.error("positive blocks required; fresh execution needs real measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        argument_parser.error("choose a fresh output directory")
    data, samples = (
        ({"backend": "synthetic-smoke"}, [(60.0, 100), (90.0, 160)])
        if args.smoke
        else load_measurements(args.measurements)
    )
    difficulty = calibrate_difficulty(samples, args.miners, args.target)
    executions = []
    if args.execution == "fresh":
        if data["backend"] != args.backend or args.backend == "smoke":
            argument_parser.error("measurement backend must match the fresh backend")
        backend = make_backend(args)
        try:
            if data["model"] != backend.identity:
                raise ValueError("calibration bank and fresh backend model/schedule differ")
            if args.prompts:
                prompts = args.prompts.read_text().splitlines()
            elif args.backend == "gpt2":
                from poml_sim.benchmark_data import benchmark_examples

                examples = benchmark_examples(
                    backend.tokenizer, "wikitext2", count=args.queries, seed=args.seed
                )
                prompts = [
                    backend.tokenizer.decode(row["context"][: (8, 16, 24, 32)[i % 4]])
                    for i, row in enumerate(examples)
                ]
            else:
                prompts = None
            system = PoMLSystem(backend, miners=args.miners, difficulty=difficulty, seed=args.seed)
            races = run_chain(
                system,
                blocks=args.blocks,
                pool_size=args.queries,
                seed=args.seed,
                prompts=prompts,
                max_output=args.max_output,
            )
            executions = system.executions
        finally:
            backend.close()
    else:
        races = [
            simulate_race(
                samples,
                miners=args.miners,
                queries=args.queries,
                difficulty=difficulty,
                seed=args.seed + i,
                replenish=True,
            )
            for i in range(args.blocks)
        ]
    if args.pow_mode == "hash":
        from poml_sim.pow import run_pow

        pow_run = run_pow(
            blocks=args.blocks, miners=args.miners, target=args.target, seed=args.seed
        )
        pow_intervals = [r["block_time"] for r in pow_run["records"]]
    else:
        rng = random.Random(f"pow:{args.seed}")
        pow_intervals = [rng.expovariate(1 / args.target) for _ in races]
        pow_run = {"mode": "Poisson statistical baseline"}
    times = [r["block_time"] for r in races if r["adopted"]]
    report = {
        "schema": "poml-liveness-1",
        "source": data["backend"],
        "mode": "synthetic-smoke"
        if args.smoke
        else "fresh-proofs-virtual-miners"
        if args.execution == "fresh"
        else "measured-pair-replay",
        "difficulty": str(difficulty),
        "target": args.target,
        "miners": args.miners,
        "queries": args.queries,
        "seed": args.seed,
        "races": races,
        "poml": summarize(times),
        "pow": summarize(pow_intervals),
        "pow_intervals": pow_intervals,
        "pow_run": pow_run,
        "pair_measurements": samples,
        "executions": executions,
    }
    import numpy as np

    x, y = np.array([c for _, c in samples], dtype=float), np.array([t for t, _ in samples])
    if np.ptp(x):
        # Center/scale huge public counts before OLS to avoid ill-conditioning.
        slope, intercept = np.polyfit(x / x.max(), y, 1)
        predicted = intercept + slope * x / x.max()
        report["ols"] = {
            "intercept_seconds": float(intercept),
            "seconds_per_complexity": float(slope / x.max()),
            "mae_seconds": float(np.mean(np.abs(predicted - y))),
        }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([times, pow_intervals], tick_labels=["PoML", f"PoW ({args.pow_mode})"])
    axes[0].set_ylabel("Block interval (s)")
    axes[1].scatter(x, y)
    if "ols" in report:
        order = np.argsort(x)
        axes[1].plot(x[order], predicted[order], color="black", label="OLS")
        axes[1].legend()
    axes[1].set(xlabel="Public complexity C", ylabel="Measured pair duration (s)")
    figure.tight_layout()
    figure.savefig(args.output / "liveness.pdf")
    plt.close(figure)
    print(json.dumps({"poml": report["poml"], "pow": report["pow"], "output": str(args.output)}))
    return 0 if all(r["adopted"] for r in races) and len(races) == args.blocks else 1


if __name__ == "__main__":
    raise SystemExit(main())
