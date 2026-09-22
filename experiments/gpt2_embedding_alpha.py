"""Run utility and layer-distance experiments with one relative Gaussian law.

Both paths derive sigma_abs = alpha * std(E) from the same frozen table and
use Gaussian samples without coordinate clipping or extra noise quantization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt2_embedding_experiments import (  # noqa: E402
    UtilityExample,
    model_weight_sha256,
    relative_noise_scales,
    run_utility_experiment,
)
from gpt2_embedding_linf import (  # noqa: E402
    clean_continuation,
    load_manifest,
    measure_prompt,
    summarize_records,
)
from gpt2_experiments import _load_model  # noqa: E402

DEFAULT_ALPHAS = (0.05, 0.1, 0.2, 0.4)
PAPER_TASKS = ("wikitext2", "hellaswag", "piqa", "arc_easy")
SCHEMA = "embedding-alpha-gaussian-1"


def load_utility_manifest(path: Path) -> list[UtilityExample]:
    """Reuse the four paper tasks' exact contexts, targets, and answer choices."""
    examples = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["task"] not in PAPER_TASKS:
            continue
        examples.append(
            UtilityExample(
                row["example_id"],
                row["task"],
                "",
                tuple(row["context_ids"]),
                tuple(row["target_ids"]),
                tuple(tuple(choice) for choice in row["choices"]),
                row["answer"],
            )
        )
    if not examples or len({row.example_id for row in examples}) != len(examples):
        raise ValueError("utility manifest must contain unique examples")
    return examples


def paper_boundaries(layers: int) -> list[tuple[str, str]]:
    """Return layer order from embedded context through projected sublayers to logits."""
    result = [("embedding", "Embedding")]
    for index in range(layers):
        result.extend(
            [
                (f"attention_{index}", f"{index + 1}: attention"),
                (f"feedforward_{index}", f"{index + 1}: feed-forward"),
            ]
        )
    return result + [("final_norm", "Final normalization"), ("logits", "Logits")]


def layer_minima(
    records: Sequence[dict[str, Any]], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pool minimum whole-context distances over prompts, pairs, and prefix lengths."""
    summary = summarize_records(records, metadata["quantization_scale"])
    result = []
    for boundary, label in paper_boundaries(metadata["layers"]):
        for alpha, sigma in zip(metadata["alphas"], metadata["sigmas"]):
            scope = "active_token" if boundary == "logits" else "context"
            rows = [
                r
                for r in summary
                if r["boundary"] == boundary
                and r["scope"] == scope
                and r["sigma_abs"] == sigma
            ]
            if not rows:
                raise ValueError(f"missing measurements for {boundary}, alpha={alpha}")
            result.append(
                {
                    "boundary": boundary,
                    "label": label,
                    "alpha": alpha,
                    "sigma_abs": sigma,
                    "raw_linf_min": min(row["raw_linf_min"] for row in rows),
                    "quantized_steps_min": min(
                        row["quantized_linf_steps_min"] for row in rows
                    ),
                    "comparisons": sum(row["comparisons"] for row in rows),
                    "clipped_coordinates": sum(
                        row["clipped_coordinates"] for row in rows
                    ),
                }
            )
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_layer_report(
    records: Sequence[dict[str, Any]], metadata: dict[str, Any], output: Path
) -> None:
    """Write exact layer tables and a log-scale plot with quantization threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = layer_minima(records, metadata)
    _write_csv(output / "layer_minima.csv", rows)
    _write_csv(
        output / "boundary_summary.csv",
        summarize_records(records, metadata["quantization_scale"]),
    )
    alphas = metadata["alphas"]
    header = " & ".join(["Layer output"] + [rf"$\alpha={alpha:g}$" for alpha in alphas])
    tex = [
        r"\begin{tabular}{l" + "r" * len(alphas) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    markdown = [
        "| Layer output | " + " | ".join(f"alpha={alpha:g}" for alpha in alphas) + " |",
        "| --- | " + " | ".join("---:" for _ in alphas) + " |",
    ]
    for boundary, label in paper_boundaries(metadata["layers"]):
        selected = [row for row in rows if row["boundary"] == boundary]
        values = [f"{row['raw_linf_min']:.5g}" for row in selected]
        tex.append(" & ".join([label] + values) + r" \\")
        markdown.append("| " + " | ".join([label] + values) + " |")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "layer_table.tex").write_text("\n".join(tex) + "\n")
    (output / "layer_table.md").write_text("\n".join(markdown) + "\n")
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    for index, alpha in enumerate(alphas):
        marker = ("o", "s", "^", "D", "v", "P")[index % 6]
        selected = [row for row in rows if row["alpha"] == alpha]
        ax.plot(
            range(len(selected)),
            [row["raw_linf_min"] for row in selected],
            marker=marker,
            markersize=3,
            linewidth=1.1,
            label=rf"$\alpha={alpha:g}$",
        )
    ax.axhline(
        metadata["quantization_scale"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=r"Quantization width $\Delta=10^{-3}$",
    )
    labels = (
        ["Emb."]
        + [
            f"{i}{kind}"
            for i in range(1, metadata["layers"] + 1)
            for kind in ("A", "F")
        ]
        + ["Norm", "Logits"]
    )
    ax.set_xticks(range(len(labels)), labels, rotation=60, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"Minimum $L_\infty$ distance")
    ax.set_xlabel("Layer output (A: attention; F: feed-forward)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=8, loc="best")
    fig.savefig(output / "layer_distances.pdf")
    fig.savefig(output / "layer_distances.png", dpi=180)
    plt.close(fig)
    # Native PGFPlots keeps the manuscript figure self-contained and portable.
    plot_tex = [
        r"\begin{tikzpicture}",
        r"\begin{semilogyaxis}[width=\linewidth,height=5.2cm,",
        r"xlabel={Layer output (A: attention; F: feed-forward)},",
        r"ylabel={Minimum $L^\infty$ distance},",
        r"tick label style={font=\scriptsize},label style={font=\small},",
        "xtick={" + ",".join(str(i) for i in range(len(labels))) + "},",
        "xticklabels={" + ",".join(labels) + "},",
        r"x tick label style={rotate=60,anchor=east},",
        r"grid=major,grid style={gray!15},",
        r"legend style={font=\scriptsize,at={(0.5,1.02)},anchor=south,",
        r"legend columns=3,draw=none}]",
    ]
    for index, alpha in enumerate(alphas):
        color = ("blue", "orange", "green!50!black", "red")[index % 4]
        mark = ("*", "square*", "triangle*", "diamond*")[index % 4]
        selected = [row for row in rows if row["alpha"] == alpha]
        coordinates = " ".join(
            f"({i},{row['raw_linf_min']:.10g})" for i, row in enumerate(selected)
        )
        plot_tex.append(
            rf"\addplot[{color},mark={mark},mark size=1pt] coordinates {{"
            + coordinates
            + "};"
        )
        plot_tex.append(rf"\addlegendentry{{$\alpha={alpha:g}$}}")
    plot_tex.extend(
        [
            rf"\addplot[black,dashed,no marks] coordinates {{(0,{metadata['quantization_scale']}) ({len(labels) - 1},{metadata['quantization_scale']})}};",
            r"\addlegendentry{Quantization width $\Delta=10^{-3}$}",
            r"\end{semilogyaxis}",
            r"\end{tikzpicture}",
        ]
    )
    (output / "layer_plot.tex").write_text("\n".join(plot_tex) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run either experiment with a single audited alpha-to-sigma conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("utility", "trace", "report"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/gpt2_embedding_alpha"),
    )
    parser.add_argument(
        "--utility-manifest",
        type=Path,
        default=Path(
            "experiments/results/gpt2_embedding/utility/prompt_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--trace-manifest",
        type=Path,
        default=Path(
            "experiments/results/gpt2_embedding/separation/prompt_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS)
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--revision", default="607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--steps", default="0,1,4,8,16,32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.mode == "report":
        directory = args.output_dir / "trace"
        metadata = json.loads((directory / "metadata.json").read_text())
        records = [
            json.loads(line)
            for line in (directory / "distances.jsonl").read_text().splitlines()
        ]
        if len(records) != metadata["examples"]:
            raise ValueError("cannot publish an incomplete campaign")
        write_layer_report(records, metadata, directory)
        return 0
    alphas = sorted(set(float(value) for value in args.alphas.split(",")))
    steps = sorted(set(int(value) for value in args.steps.split(",")))
    if (
        args.pairs < 1
        or args.replicates < 1
        or min(steps) < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("counts must be positive and steps nonnegative")
    model, _, torch = _load_model(args.model, args.device, args.revision)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    table_std, sigmas = relative_noise_scales(model, alphas)
    if args.mode == "utility":
        examples = load_utility_manifest(args.utility_manifest)
        manifest = args.utility_manifest
    else:
        # Match the user's six-dataset manuscript: omit the old repeat-copy stratum.
        examples = [
            p
            for p in load_manifest(args.trace_manifest)
            if p["stratum"] != "repeat_copy_logic"
        ]
        manifest = args.trace_manifest
    if args.limit:
        examples = examples[: args.limit]
    output = args.output_dir / args.mode
    metadata = {
        "schema": SCHEMA,
        "mode": args.mode,
        "alphas": alphas,
        "sigmas": list(sigmas),
        "embedding_table_std": table_std,
        "std_definition": "population std of every frozen embedding-table entry, FP64",
        "noise": "VRF/SHA256 Box-Muller standard Gaussian; no clipping; no extra noise rounding; cast to model FP32",
        "noise_clip": None,
        "noise_quantum": 0.0,
        "model": args.model,
        "revision": args.revision,
        "model_weight_sha256": model_weight_sha256(model),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "examples": len(examples),
        "pairs": args.pairs,
        "replicates": args.replicates,
        "steps": steps,
        "seed": args.seed,
        "layers": model.config.n_layer,
        "quantization_scale": 0.001,
        "quantization_clip": 2**31 - 1,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "source_hashes": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "gpt2_embedding_alpha.py",
                "gpt2_embedding_linf.py",
                "gpt2_embedding_experiments.py",
            )
        },
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        parser.error("saved campaign settings differ; choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"s_E={table_std:.15g}; alpha={alphas}; sigma_abs={sigmas}", flush=True)
    if args.mode == "utility":
        run_utility_experiment(
            model,
            examples,
            sigmas=sigmas,
            perturbation_seeds=args.replicates,
            seed=args.seed,
            device=args.device,
            output_dir=output,
            noise_clip=None,
            quantization_quantum=0.0,
        )
    else:
        checkpoint = output / "distances.jsonl"
        records = (
            [json.loads(line) for line in checkpoint.read_text().splitlines()]
            if checkpoint.exists()
            else []
        )
        completed = {r["prompt_id"] for r in records}
        from tqdm import tqdm

        for prompt in tqdm(examples, desc="Relative-noise layers", unit="prompt"):
            if prompt["prompt_id"] in completed:
                continue
            continuation = clean_continuation(model, prompt["token_ids"], max(steps))
            distances = measure_prompt(
                model,
                prompt,
                continuation,
                sigmas=sigmas,
                pairs=args.pairs,
                steps=steps,
                seed=args.seed,
                scale=0.001,
                clip=2**31 - 1,
                noise_clip=None,
                noise_quantum=0.0,
                detailed_layers=True,
            )
            record = {**prompt, "continuation": continuation, "distances": distances}
            with checkpoint.open("a") as handle:
                handle.write(
                    json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                )
            records.append(record)
        write_layer_report(records, metadata, output)
    print(f"Completed {len(examples)} {args.mode} examples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
