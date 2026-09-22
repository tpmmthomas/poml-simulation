"""Appendix DDPM compatibility: paired latent trajectories, cosine and min L∞."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from poml_sim.crypto import canonical, sha256
from poml_sim.diffusion import DDPMModel


def trajectory_metrics(first, second):
    """Compare every pre-denoiser state without conflating with U-Net layer hooks."""
    import torch

    if len(first) != len(second) or not first:
        raise ValueError("aligned nonempty trajectories required")
    rows = []
    for a, b in zip(first, second):
        a, b = a.double().flatten(), b.double().flatten()
        rows.append(
            {
                "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                "linf": float((a - b).abs().max()),
            }
        )
    return rows


def main(argv=None):
    """Run the paper defaults or a deliberately smaller explicit wiring check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--sigmas", default="0.001,0.01,0.1,0.5,1.0")
    parser.add_argument("--prompt", default="a photo of a cat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results/diffusion-compatibility")
    )
    args = parser.parse_args(argv)
    sigmas = [float(value) for value in args.sigmas.split(",")]
    if args.samples < 1 or min(sigmas) <= 0:
        parser.error("positive samples and sigmas required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "trajectories.jsonl").exists():
        parser.error("choose a fresh output directory")
    model = DDPMModel(args.model, device=args.device, steps=args.steps, size=args.size)
    records = []
    for sample in range(args.samples):
        seeds = [sha256(canonical([args.seed, sample, "base", t])) for t in range(args.steps)]
        independent = [
            sha256(canonical([args.seed, sample, "independent", t])) for t in range(args.steps)
        ]
        base = model.trajectory(args.prompt, seeds)
        perturbation = model.noise(sha256(canonical([args.seed, sample, "perturbation"])))
        for sigma in [*sigmas, None]:
            # The user confirmed common reverse-step noise for perturbations;
            # the separate baseline uses independent initial and reverse noise.
            other = model.trajectory(
                args.prompt,
                independent if sigma is None else seeds,
                perturbation=None if sigma is None else sigma * perturbation,
            )
            row = {"sample": sample, "sigma": sigma, "steps": trajectory_metrics(base, other)}
            records.append(row)
            with (args.output / "trajectories.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        print(f"Completed {sample + 1}/{args.samples}", flush=True)
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    summary = []
    for sigma in [*sigmas, None]:
        rows = [r for r in records if r["sigma"] == sigma]
        cosines = np.array([[r["cosine"] for r in row["steps"]] for row in rows])
        linf = min(r["linf"] for row in rows for r in row["steps"])
        mean, std = cosines.mean(axis=0), cosines.std(axis=0)
        x = np.arange(1, args.steps + 1)
        ax.plot(x, mean, label="independent" if sigma is None else f"σ={sigma}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.12)
        summary.append(
            {
                "sigma": sigma,
                "min_linf": linf,
                "cosine_mean": mean.tolist(),
                "cosine_sd": std.tolist(),
            }
        )
    ax.set(xlabel="Denoiser input step", ylabel="Cosine similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "cosine.pdf")
    plt.close(fig)
    metadata = {
        **vars(args),
        "output": str(args.output),
        "scheduler": "DDPM fixed_small, no clipping, final step noiseless",
        "perturbation_reverse_noise": "shared",
        "independent_baseline": "all noise independent",
        "dtype": str(model.dtype),
        "summary": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
