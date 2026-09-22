"""Fit nonnegative operation weights to genuine GPT-2 inference/proof timings."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.backends import load_schedule
from poml_sim.experiments import load_measurements
from poml_sim.runtime_weights import fit_runtime_schedule


def main(argv=None):
    """Export integer weights and nested prompt-grouped validation errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (args.output / "weights.json").exists():
        parser.error("choose a fresh output directory")
    data, _ = load_measurements(args.measurements)
    if data["backend"] != "gpt2":
        parser.error("the appendix operation schedule applies only to GPT-2/DeepProve")
    fit, predictions = fit_runtime_schedule(data["records"], load_schedule(), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "weights.json").write_text(json.dumps(fit, indent=2) + "\n")
    (args.output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    print(
        json.dumps(
            {
                "records": fit["records"],
                "held_out_uniform": fit["held_out_uniform"],
                "held_out_weighted": fit["held_out_weighted"],
                "rank": fit["normalized_feature_rank"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
