#!/usr/bin/env python3
"""Fit archived GPT-2 work timings and replay Experiment 2 without a prover.

The input campaign is read-only. All fitted weights, validation predictions,
fresh literal lotteries and source snapshots are retained in a separate run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_live_llm_experiments import (  # noqa: E402
    calibrate_difficulty,
    cherry_pick,
    read_jsonl,
    run_chain,
    write_csv,
    write_json,
)
from poml_sim.gpt2_work import reference_counts, weighted_cost  # noqa: E402
from poml_sim.llm_benchmark import SELECTION_POLICIES  # noqa: E402
from poml_sim.llm_simulation import load_schedule  # noqa: E402
from poml_sim.lottery import scaled_complexity  # noqa: E402
from poml_sim.profiled_mining import MeasuredBank, file_digest  # noqa: E402
from poml_sim.runtime_weights import fit_runtime_schedule  # noqa: E402


def plot_predictions(rows, destination):
    """Plot nested held-out predictions, separate from the final fitted replay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharex=True, sharey=True)
    actual = [r["actual_seconds"] for r in rows]
    for ax, key, title in zip(
        axes,
        ("uniform_held_out_seconds", "weighted_held_out_seconds"),
        ("Uniform operation weights", "Fitted operation weights"),
    ):
        ax.scatter(
            actual,
            [r[key] for r in rows],
            c=[r["output_length"] for r in rows],
            cmap="viridis",
            s=16,
            alpha=0.75,
        )
        ax.plot([55, 125], [55, 125], color="gray", linestyle="--", linewidth=1)
        ax.set(
            title=title,
            xlabel="Measured inference + proof (s)",
            xlim=(55, 125),
            ylim=(55, 125),
        )
    axes[0].set_ylabel("Held-out prediction (s)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"held_out_predictions.{suffix}", dpi=180)
    plt.close(fig)


def summarize_policies(directory):
    """Pool completed work and chain time, retaining every zero-win chain."""
    results = []
    for policy in SELECTION_POLICIES:
        chains = sorted(directory.glob(f"*-{policy}"))
        summaries = [json.loads((p / "summary.json").read_text()) for p in chains]
        attempts = [
            r
            for p in chains
            for r in json.loads((p / "attempts.json").read_text())
            if r["logical_status"] == "completed" and r["miner"] == 0
        ]
        blocks = sum(r["blocks"] for r in summaries)
        wins = sum(r["attacker_blocks"] for r in summaries)
        seconds = sum(r["virtual_chain_seconds"] for r in summaries)
        results.append(
            {
                "policy": policy,
                "blocks": blocks,
                "attacker_blocks": wins,
                "attacker_block_share": wins / blocks,
                "attacker_blocks_per_hour": wins * 3600 / seconds,
                "tickets_per_busy_second": sum(r["complexity"] for r in attempts)
                / sum(r["duration"] for r in attempts),
                "mean_realised_K": sum(r["output_length"] for r in attempts)
                / len(attempts),
            }
        )
    baseline = next(r for r in results if r["policy"] == "uniform")
    for row in results:
        row["ticket_rate_relative_to_uniform"] = (
            row["tickets_per_busy_second"] / baseline["tickets_per_busy_second"]
        )
        row["block_yield_relative_to_uniform"] = (
            row["attacker_blocks_per_hour"] / baseline["attacker_blocks_per_hour"]
            if baseline["attacker_blocks_per_hour"]
            else None
        )
    return results


def write_comparison(source, output, fit):
    """Save the archived uniform-weight control and frozen-weight replay results."""
    old = summarize_policies(source / "cherry_pick")
    new = summarize_policies(output / "cherry_pick")
    comparison = [
        {"schedule": schedule, **row}
        for schedule, rows in (("uniform", old), ("fitted", new))
        for row in rows
    ]
    write_json(output / "comparison.json", comparison)
    write_csv(output / "comparison.csv", comparison)
    lines = [
        "# Runtime-weighted selection replay",
        "",
        f"Source: `{source}`. No new inference, proofs, or output-length profiles were generated.",
        "",
        f"Fit: {fit['records']} genuine pairs, {fit['distinct_prompts']} distinct prompts, "
        f"{fit['distinct_shapes']} realised (N,K) shapes. {fit['operation_columns']} count columns have "
        f"rank {fit['normalized_feature_rank']}; individual weights are not separately identified.",
        "",
        f"Nested prompt-held-out MAPE: uniform {fit['held_out_uniform']['mape_percent']:.3f}%; "
        f"fitted {fit['held_out_weighted']['mape_percent']:.3f}%. "
        f"MAE: {fit['held_out_uniform']['mae_seconds']:.3f} → {fit['held_out_weighted']['mae_seconds']:.3f} seconds.",
        "",
        "| Schedule | Policy | Wins / blocks | Share | Tickets / busy s | Rate / uniform policy | Yield / uniform policy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison:
        relative_yield = row["block_yield_relative_to_uniform"]
        yield_text = (
            f"{relative_yield:.4f}" if relative_yield is not None else "undefined"
        )
        lines.append(
            f"| {row['schedule']} | {row['policy']} | {row['attacker_blocks']}/{row['blocks']} | "
            f"{100 * row['attacker_block_share']:.2f}% | {row['tickets_per_busy_second']:.3f} | "
            f"{row['ticket_rate_relative_to_uniform']:.4f} | {yield_text} |"
        )
    lines.extend(
        [
            "",
            "The final schedule is fitted on the whole existing bank. This replay is an in-bank "
            "sensitivity experiment; nested validation separately estimates prediction on held-out prompts. "
            "The expected symmetric share is 25%, not the observed uniform-policy share. "
            "Completion races, cancellations, and finite literal-lottery samples can leave unequal block shares "
            "even when completed-ticket throughput is close. Three seed-level chains per policy give limited precision.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(comparison, indent=2), flush=True)


def main(argv=None):
    """Run a CPU-only fit, calibration and replay against an immutable input bank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schedule", type=Path, default=ROOT / "config/gpt2_reference_schedule.json"
    )
    parser.add_argument("--fit-seed", type=int, default=20260916)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output.resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
        or args.workers < 1
    ):
        parser.error("use separate input/output trees and at least one worker")
    if (
        not (source / "preparation/COMPLETED").exists()
        or not (source / "cherry_pick/COMPLETED").exists()
    ):
        parser.error("source must have completed preparation and Experiment 2")
    source_config = json.loads((source / "campaign.json").read_text())
    source_calibration = json.loads(
        (source / "preparation/calibration.json").read_text()
    )
    schedule = load_schedule(args.schedule)
    pool = json.loads((source / "preparation/pool.json").read_text())
    records = read_jsonl(source / "preparation/timings.jsonl")
    paths = [
        *sorted((ROOT / "src/poml_sim").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "experiments/run_live_llm_experiments.py",
        ROOT / "experiments/audit_live_llm_run.py",
    ]
    manifest = {
        "schema": "poml-runtime-selection-1",
        "source": str(source),
        "output": str(output),
        "fit_seed": args.fit_seed,
        "workers": args.workers,
        "miners": source_config["miners"],
        "target": source_config["target"],
        "selection_blocks": source_config["selection_blocks"],
        "seeds": source_config["seeds"],
        "seed": source_config["seed"],
        "alpha": source_config["alpha"],
        "queries": len(pool),
        "records": len(records),
        "fresh_inference_or_proofs": False,
        "sources": {str(p.relative_to(ROOT)): file_digest(p) for p in paths},
        "input_files": {
            name: file_digest(source / name)
            for name in (
                "campaign.json",
                "preparation/pool.json",
                "preparation/timings.jsonl",
                "preparation/calibration.json",
            )
        },
        "schedule_sha256": file_digest(args.schedule),
    }
    manifest_path = output / "campaign.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "existing run requires --resume and identical sources/configuration"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty")
    else:
        write_json(manifest_path, manifest)
        for path in paths:
            destination = output / "source_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        shutil.copy2(args.schedule, output / "source_snapshot/reference_schedule.json")
        write_json(
            output / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": __import__("numpy").__version__,
                "command": sys.argv,
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
        )
        (output / "preparation").mkdir()
        for name in ("pool.json", "timings.jsonl"):
            shutil.copy2(source / "preparation" / name, output / "preparation" / name)
        write_json(output / "preparation/source_calibration.json", source_calibration)
    print(
        f"Validate {len(records)} existing proof/output records; no prover will be started.",
        flush=True,
    )
    bank = MeasuredBank(pool, records, schedule, source_calibration["scale"])
    fit_path = output / "runtime_weights.json"
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
    else:
        fit, predictions = fit_runtime_schedule(
            records,
            schedule,
            seed=args.fit_seed,
            ticket_target=source_config["ticket_target"],
        )
        fit["bank_sha256"] = bank.sha256
        write_json(fit_path, fit)
        write_csv(output / "validation_predictions.csv", predictions)
        write_csv(
            output / "operation_weights.csv",
            [
                {
                    "operation": name,
                    "integer_weight": value,
                    "seconds_per_operation": value
                    / fit["fixed_point_units_per_second"],
                    "normalization": fit["normalization"][name],
                }
                for name, value in fit["weights"].items()
            ],
        )
        plot_predictions(predictions, output)
    print(
        json.dumps(
            {
                key: fit[key]
                for key in (
                    "held_out_uniform",
                    "held_out_weighted",
                    "training_weighted",
                    "penalty",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fit_only:
        return
    calibration_path = output / "preparation/calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
    else:
        inputs = [
            {
                "query_id": row["query_id"],
                "output_length": row["output_length"],
                "duration": row["inference_proof_seconds"],
                "complexity": scaled_complexity(
                    weighted_cost(
                        reference_counts(
                            row["prompt_length"], row["output_length"], schedule
                        )["combined"],
                        fit["weights"],
                    ),
                    fit["scale"],
                ),
            }
            for row in records
        ]
        write_json(output / "preparation/calibration_inputs.json", inputs)
        calibration = {
            **source_calibration,
            **calibrate_difficulty(
                pool,
                inputs,
                schedule,
                manifest["miners"],
                manifest["target"],
                manifest["seed"],
            ),
            "weights": fit["weights"],
            "scale": fit["scale"],
            "weights_file_sha256": file_digest(fit_path),
            "source_bank_scale": source_calibration["scale"],
            "new_inference_proof_pairs": 0,
        }
        write_json(calibration_path, calibration)
        (output / "preparation/COMPLETED").write_text(
            "Existing bank validated; fitted weights and difficulty frozen.\n"
        )
    replay_args = SimpleNamespace(**{**manifest, "output": output})
    (output / "cherry_pick").mkdir(exist_ok=True)
    # Chains share only immutable bank records. Each has its own RNG, keys,
    # output directory and checkpoint; scheduling cannot affect its lottery.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_chain,
                replay_args,
                calibration["prover"],
                bank,
                schedule,
                calibration,
                output / "cherry_pick" / f"{repeat:03d}-{policy}",
                manifest["selection_blocks"],
                policy,
                manifest["seed"] + 100000 + repeat,
            )
            for repeat in range(manifest["seeds"])
            for policy in SELECTION_POLICIES
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Completed selection chains"
        ):
            future.result()
    cherry_pick(replay_args, calibration["prover"], bank, schedule, calibration)
    write_comparison(source, output, fit)
    (output / "COMPLETED").write_text(
        "Runtime fit and Experiment 2 replay complete; zero new inference/proof pairs.\n"
    )


if __name__ == "__main__":
    main()
