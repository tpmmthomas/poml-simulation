#!/usr/bin/env python3
"""Profiled GPT-2/DeepProve Experiments 1–2 and measured-trace Experiment 3."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from poml_sim.gpt2_work import reference_counts  # noqa: E402
from poml_sim.live_prover import LiveProver  # noqa: E402
from poml_sim.live_mining import FreshWork, LiveChain, append_json  # noqa: E402
from poml_sim.profiled_mining import MeasuredBank, ProfiledWork  # noqa: E402
from poml_sim.llm_benchmark import wikitext_prompts, profile_pool, SELECTION_POLICIES  # noqa: E402
from poml_sim.llm_simulation import QuerySpec, QueryTrace, simulate_race, load_schedule  # noqa: E402
from poml_sim.lottery import LIMIT, calibrate_scale  # noqa: E402
from poml_sim.protocol_inputs import (  # noqa: E402
    canonical,
    digest,
    experimental_key,
    gaussian_noise,
    decoding_uniforms,
)
from experiments.build_real_llm_trace_bank import _load_wikitext  # noqa: E402

SCHEMA = "poml-live-campaign-1"


def write_json(path: Path, value):
    """Atomically save a completed stage or checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    """Read the durable JSON-lines records from a stage."""
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def write_csv(path, rows):
    """Write scalar analysis rows without dropping columns silently."""
    if not rows:
        return
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    """Report block/pair distributions, including spread and quantiles."""
    import numpy as np

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "cv": statistics.stdev(values) / statistics.fmean(values)
        if len(values) > 1
        else None,
        "minimum": min(values),
        "maximum": max(values),
        **{f"p{p}": float(np.percentile(values, p)) for p in (5, 25, 75, 95, 99)},
    }


def make_pool(args):
    """Persist multiple distinct genuine WikiText-2 token windows and provenance."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2", local_files_only=True
    )
    rows, provenance = _load_wikitext(args.dataset_cache_file)
    prompts = wikitext_prompts(
        rows,
        tokenizer.encode,
        count=args.queries,
        lengths=tuple(args.prompt_lengths),
        max_output=args.max_output,
        seed=args.seed,
    )
    write_json(args.output / "prompts.json", prompts)
    write_json(args.output / "dataset.json", provenance)
    return prompts


def prepare(args, prover, schedule):
    """Build the genuine measurement bank used by the fast replay simulations."""
    directory = args.output / "preparation"
    directory.mkdir(exist_ok=True)
    completed = directory / "COMPLETED"
    if completed.exists():
        if not (directory / "bank_manifest.json").exists():
            raise RuntimeError(
                "existing preparation predates the profiled replay bank; use a new output directory"
            )
        saved = json.loads((directory / "calibration.json").read_text())
        if saved["prover"]["setup_sha256"] != prover.ready["setup_sha256"]:
            raise RuntimeError(
                "registered prover setup changed; refusing to mix models/setups"
            )
        return saved
    prompts = (
        json.loads((args.output / "prompts.json").read_text())
        if (args.output / "prompts.json").exists()
        else make_pool(args)
    )
    profile_path = directory / "profiles.jsonl"
    profiles = read_jsonl(profile_path)
    finished = {(r["query_id"], r["replicate"]) for r in profiles}
    for q in tqdm(prompts, desc="Profile benchmark prompts"):
        for replicate in range(args.profile_replicates):
            if (q["query_id"], replicate) in finished:
                continue
            challenge = canonical([args.seed, "profile", q["query_id"], replicate])
            noise, vrfs = gaussian_noise(
                experimental_key(args.seed, "profiling-inf"),
                digest(challenge),
                q["prompt_length"],
                q["max_output_length"],
                args.alpha,
                prover.ready["embedding_std"],
            )
            identity = digest(challenge).hex()
            destination = directory / "profiles" / identity
            if destination.exists():
                # An interrupted execution has no recorded outcome: use a new
                # directory and execute again, never trust an incomplete proof.
                destination = (
                    directory
                    / "profiles"
                    / f"{identity}-{len(list(destination.parent.iterdir()))}"
                )
            result = prover.run(
                {
                    "request_id": identity,
                    "prompt_tokens": q["prompt_tokens"],
                    "noise": noise,
                    "uniforms": decoding_uniforms(vrfs, q["prompt_length"]),
                    "max_output": q["max_output_length"],
                    "mode": "profile",
                    "output_directory": str(destination),
                }
            )
            row = {
                **result,
                "query_id": q["query_id"],
                "prompt_sha256": q["prompt_sha256"],
                "prompt_length": q["prompt_length"],
                "phase": "profile",
                "replicate": replicate,
                "challenge_seed": challenge.hex(),
                "gpt2_duration": result["inference_seconds"],
                "raw_complexity": sum(
                    reference_counts(
                        q["prompt_length"], result["output_length"], schedule
                    )["combined"].values()
                ),
                "vrf_transcript": vrfs,
            }
            append_json(profile_path, row)
            profiles.append(row)
    pool = profile_pool(prompts, profiles)
    scale = calibrate_scale([r["raw_complexity"] for r in profiles], args.ticket_target)
    write_json(directory / "pool.json", pool)
    write_json(directory / "complexity_scale.json", scale)
    # Calibration uses separate VRF keys and challenges, and actual proofs.
    work = FreshWork(
        prover, schedule, scale, directory / "proofs", args.seed + 1, args.alpha
    )
    existing = list(work.directory.glob("attempt-*"))
    work.executions = max([int(p.name.split("-")[1]) for p in existing], default=-1) + 1
    timing_path = directory / "timings.jsonl"
    timings = read_jsonl(timing_path)
    done = {(r["query_id"], r["replicate"]) for r in timings}
    for q in tqdm(pool, desc="Build genuine measurement bank"):
        for replicate in range(args.calibration_replicates):
            if (q["query_id"], replicate) in done:
                continue
            binding = digest(
                canonical(["calibration", args.seed, q["query_id"], replicate])
            )
            row = work.execute(
                {**q, "request_id": f"calibration:{q['query_id']}:{replicate}"},
                0,
                binding,
                binding,
                b"",
                1,
            )
            row.pop("ciphertext_prefix")
            row["replicate"] = replicate
            # Replay uses only inference+proof service time; retain host time separately.
            row["service_seconds"] = row["duration"]
            row["duration"] = row["inference_proof_seconds"]
            append_json(timing_path, row)
            timings.append(row)
    bank_manifest = {
        "schema": "poml-measured-bank-1",
        "records": len(timings),
        "prompts": len(pool),
        "replicates_per_prompt": args.calibration_replicates,
        "prompt_ids": [q["query_id"] for q in pool],
        "output_lengths": sorted({r["output_length"] for r in timings}),
        "prompt_lengths": sorted({r["prompt_length"] for r in timings}),
        "duration_statistics": summarize([r["duration"] for r in timings]),
        "fresh_inference_and_proof": all(
            r.get("fresh_inference") is True and r.get("fresh_proof") is True
            for r in timings
        ),
        "bank_sha256": digest(canonical(timings)).hex(),
    }
    write_json(directory / "bank_manifest.json", bank_manifest)
    calibration = calibrate_difficulty(
        pool, timings, schedule, args.miners, args.target, args.seed
    )
    calibration.update(
        schema=SCHEMA,
        scale=scale,
        prover=prover.ready,
        profile_rank_identifiable=len({q["profile_mean_output_length"] for q in pool})
        > 1,
        profile_cap_fraction=statistics.fmean(
            r["stop_reason"] == "length_cap" for r in profiles
        ),
        duration_statistics=summarize([r["duration"] for r in timings]),
        measured_bank=bank_manifest,
        notes=[
            "Reference C uses one uniformly scaled unit-weight schedule.",
            "Modified noise/logit graph costs are not a new audited symbolic schedule.",
            "The measurement bank runs genuine perturbed GPT-2 inference and full-output proofs.",
            "Experiments 1 and 2 replay measured inference+proof service times; auxiliary protocol work and literal lotteries are recomputed.",
            "Setup, proof verification, encryption, hashing, and propagation are excluded from virtual attempt duration.",
        ],
    )
    if args.calibration_replicates >= 2:
        from poml_sim.llm_benchmark import prediction_diagnostics

        heldout = [
            {**row, "phase": "evaluation", "challenge_seed": row["r"]}
            for row in timings
        ]
        diagnostics = prediction_diagnostics(pool, profiles, heldout)
        write_json(directory / "prediction_diagnostics.json", diagnostics)
        calibration["prediction_diagnostics"] = diagnostics
    calibration["profiling_inference_seconds"] = sum(
        r["gpt2_duration"] for r in profiles
    )
    calibration["calibration_physical_seconds"] = sum(
        r["host_seconds"] for r in timings
    )
    calibration["measurement_bank_physical_seconds"] = (
        calibration["profiling_inference_seconds"]
        + calibration["calibration_physical_seconds"]
    )
    total_blocks = (
        args.blocks + len(SELECTION_POLICIES) * args.selection_blocks * args.seeds
    )
    calibration[
        "approximate_serial_campaign_hours_before_setup_profiling_and_cancellations"
    ] = total_blocks * args.miners * args.target / 3600
    write_json(directory / "calibration.json", calibration)
    completed.write_text("complete\n")
    print(json.dumps(calibration, indent=2))
    return calibration


def calibrate_difficulty(pool, timings, schedule, miners, target, seed):
    """Tune D on empirical renewal times; this calibration replay is never mining data."""
    queries = [
        QuerySpec(q["query_id"], q["prompt_length"], q["max_output_length"])
        for q in pool
    ]
    bank = {q.query_id: [] for q in queries}
    for row in timings:
        bank[row["query_id"]].append(
            QueryTrace(row["output_length"], row["complexity"], row["duration"])
        )
    rate = sum(r["complexity"] for r in timings) / sum(r["duration"] for r in timings)

    def mean(p, seeds):
        return statistics.fmean(
            simulate_race(
                queries,
                miner_count=miners,
                difficulty_probability=p,
                seed=s,
                schedule=schedule,
                trace_bank=bank,
                trace_duration=True,
                require_trace_bank=True,
                replenish_pool=True,
                max_events=None,
            ).block_time
            for s in seeds
        )

    seeds = range(seed, seed + 1000)
    low, high = (
        1 / (miners * rate * target) / 100,
        min(0.1, 100 / (miners * rate * target)),
    )
    fastest = mean(high, seeds)
    if fastest > target:
        raise ValueError(
            f"target {target}s is below the empirical achievable {fastest:.2f}s; increase --target"
        )
    while mean(low, seeds) < target:
        low /= 10
    for _ in tqdm(range(18), desc="Calibrate block difficulty"):
        middle = math.sqrt(low * high)
        if mean(middle, seeds) > target:
            low = middle
        else:
            high = middle
    p = math.sqrt(low * high)
    difficulty = max(1, int(p * LIMIT))
    return {
        "difficulty": str(difficulty),
        "per_ticket_probability": difficulty / LIMIT,
        "target_seconds": target,
        "miners": miners,
        "ticket_rate": rate,
        "calibration_seed_count": 1000,
        "independent_validation_mean_seconds": mean(
            difficulty / LIMIT, range(seed + 10000, seed + 13000)
        ),
    }


def chain_state(chain):
    """Snapshot only the state after an adopted block."""
    return {
        "pending": chain.pending,
        "next_qid": chain.next_qid,
        "height": chain.height,
        "parent": chain.parent.hex(),
        "rng": chain.rng.getstate(),
    }


def restore_chain(chain, state):
    """Restore query state and deterministic selection after an interruption."""

    def tuples(value):
        return tuple(map(tuples, value)) if isinstance(value, list) else value

    chain.pending, chain.next_qid, chain.height = (
        state["pending"],
        state["next_qid"],
        state["height"],
    )
    chain.parent = bytes.fromhex(state["parent"])
    chain.rng.setstate(tuples(state["rng"]))


def run_chain(
    args, ready, bank, schedule, calibration, destination, blocks, policy, seed
):
    """Replay measured service times while recomputing protocol events and hashes."""
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETED").exists():
        return json.loads((destination / "summary.json").read_text())
    pool = json.loads((args.output / "preparation/pool.json").read_text())
    work = ProfiledWork(
        bank,
        ready,
        schedule,
        calibration["scale"],
        destination / "work",
        seed,
        args.alpha,
        weights=calibration.get("weights"),
    )
    work.executions = (
        max(
            [int(p.name.split("-")[1]) for p in work.directory.glob("attempt-*")],
            default=-1,
        )
        + 1
    )
    chain = LiveChain(
        work, pool, args.miners, int(calibration["difficulty"]), seed, policy
    )
    checkpoint = destination / "checkpoint.json"
    rows, attempts = [], []
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        restore_chain(chain, saved["chain"])
        rows, attempts = saved["blocks"], saved["attempts"]
    for _ in tqdm(
        range(chain.height, blocks),
        initial=chain.height,
        total=blocks,
        desc=f"{policy}: adopted blocks",
    ):
        block, events = chain.block()
        rows.append(block)
        attempts.extend(events)
        write_json(
            checkpoint,
            {"chain": chain_state(chain), "blocks": rows, "attempts": attempts},
        )
    completed = [a for a in attempts if a["logical_status"] == "completed"]
    attacker = [a for a in completed if a["miner"] == 0]
    seconds = sum(r["block_time"] for r in rows)
    wins = sum(r["winner"] == 0 for r in rows)
    physical = read_jsonl(work.directory / "executed.jsonl")
    summary = {
        "policy": policy,
        "seed": seed,
        "blocks": blocks,
        "block_intervals": summarize([r["block_time"] for r in rows]),
        "pair_durations": summarize([r["duration"] for r in completed]),
        "attacker_blocks": wins,
        "attacker_block_share": wins / blocks,
        "attacker_tickets_per_chain_second": sum(r["complexity"] for r in attacker)
        / seconds,
        "attacker_work_share": sum(r["complexity"] for r in attacker)
        / sum(r["complexity"] for r in completed),
        "attacker_blocks_per_hour": wins * 3600 / seconds,
        "attacker_tickets_per_busy_second": (
            sum(r["complexity"] for r in attacker)
            / sum(r["duration"] for r in attacker)
        )
        if attacker
        else None,
        "attacker_mean_output_length": statistics.fmean(
            r["output_length"] for r in attacker
        )
        if attacker
        else None,
        "logical_completed_attempts": len(completed),
        "logical_canceled_attempts": len(attempts) - len(completed),
        "profiled_replay_executions_including_abandoned_blocks": len(physical),
        "allocated_attempt_directories": work.executions,
        "profiled_host_seconds_including_abandoned_blocks": sum(
            r["host_seconds"] for r in physical
        ),
        "host_seconds": sum(r["host_seconds"] for r in rows),
        "virtual_chain_seconds": seconds,
    }
    write_json(destination / "summary.json", summary)
    write_json(destination / "blocks.json", rows)
    write_json(destination / "attempts.json", attempts)
    write_csv(
        destination / "blocks.csv",
        [{k: v for k, v in r.items() if k != "block"} for r in rows],
    )
    write_csv(
        destination / "attempts.csv",
        [
            {k: v for k, v in a.items() if not isinstance(v, (dict, list))}
            for a in attempts
        ],
    )
    (destination / "COMPLETED").write_text("complete\n")
    return summary


def plot_liveness(args):
    """Produce boxes and whiskers without individual points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = args.output / "liveness"
    blocks = json.loads((directory / "blocks.json").read_text())
    attempts = json.loads((directory / "attempts.json").read_text())
    rng = random.Random(args.seed)
    pow_times = [rng.expovariate(1 / args.target) for _ in blocks]
    write_json(directory / "pow_simulated_intervals.json", pow_times)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    axes[0].boxplot(
        [[r["block_time"] for r in blocks], pow_times],
        tick_labels=["PoML", "PoW model"],
        showfliers=False,
    )
    axes[0].set_ylabel("Block interval (s)")
    axes[1].boxplot(
        [r["duration"] for r in attempts if r["logical_status"] == "completed"],
        tick_labels=["Inference + proof + lottery"],
        showfliers=False,
    )
    axes[1].set_ylabel("Completed attempt time (s)")
    fig.tight_layout()
    fig.savefig(directory / "boxplots.pdf")
    fig.savefig(directory / "boxplots.png", dpi=250)
    plt.close(fig)


def pooled_relative_yield(rows, baseline, seed):
    """Compare pooled block rates without dropping zero-win baseline seeds."""
    baseline = {r["seed"]: r for r in baseline}
    pairs = [(r, baseline[r["seed"]]) for r in rows]

    def ratio(selected):
        wins = [
            sum(r[i]["attacker_block_share"] * r[i]["blocks"] for r in selected)
            for i in (0, 1)
        ]
        seconds = [sum(r[i]["virtual_chain_seconds"] for r in selected) for i in (0, 1)]
        return wins[0] * seconds[1] / (wins[1] * seconds[0]) if wins[1] else None

    result = {"relative_block_yield": ratio(pairs)}
    if len(pairs) >= 2:
        rng = random.Random(seed)
        draws = [ratio(rng.choices(pairs, k=len(pairs))) for _ in range(2000)]
        finite = sorted(d for d in draws if d is not None)
        result["relative_yield_undefined_bootstrap_fraction"] = 1 - len(finite) / len(
            draws
        )
        # An interval with too many zero-denominator samples is not identified.
        if len(finite) >= 1950:
            result["relative_block_yield_ci95_low"] = finite[int(0.025 * len(finite))]
            result["relative_block_yield_ci95_high"] = finite[
                min(len(finite) - 1, int(0.975 * len(finite)))
            ]
    return result


def cherry_pick(args, ready, bank, schedule, calibration):
    """Compare one deviating miner against uniform miners using measured work."""
    directory = args.output / "cherry_pick"
    directory.mkdir(exist_ok=True)
    results = []
    for repeat in range(args.seeds):
        policies = list(SELECTION_POLICIES)
        random.Random(args.seed + repeat).shuffle(policies)
        for policy in policies:
            results.append(
                run_chain(
                    args,
                    ready,
                    bank,
                    schedule,
                    calibration,
                    directory / f"{repeat:03d}-{policy}",
                    args.selection_blocks,
                    policy,
                    args.seed + 100000 + repeat,
                )
            )
    report = []
    for row in results:
        baseline = next(
            r for r in results if r["seed"] == row["seed"] and r["policy"] == "uniform"
        )
        report.append(
            {
                **{k: v for k, v in row.items() if not isinstance(v, dict)},
                "measurement_bank_cost_seconds": calibration[
                    "measurement_bank_physical_seconds"
                ],
                "profiling_cost_seconds": calibration["profiling_inference_seconds"],
                "attacker_blocks_per_hour_including_profiling": row[
                    "attacker_block_share"
                ]
                * row["blocks"]
                * 3600
                / (
                    row["virtual_chain_seconds"]
                    + (
                        calibration["profiling_inference_seconds"]
                        if row["policy"] in {"profiled-short", "profiled-long"}
                        else 0.0
                    )
                ),
                "attacker_blocks_per_hour_including_measurement_bank": row[
                    "attacker_block_share"
                ]
                * row["blocks"]
                * 3600
                / (
                    row["virtual_chain_seconds"]
                    + calibration["measurement_bank_physical_seconds"]
                ),
                "relative_block_yield": row["attacker_blocks_per_hour"]
                / baseline["attacker_blocks_per_hour"]
                if baseline["attacker_blocks_per_hour"]
                else None,
            }
        )
    intervals = []
    for policy in SELECTION_POLICIES:
        rows = [r for r in report if r["policy"] == policy]
        entry = {"policy": policy, "independent_chains": len(rows)}
        for metric in (
            "attacker_block_share",
            "attacker_work_share",
            "attacker_blocks_per_hour",
            "attacker_blocks_per_hour_including_profiling",
            "attacker_blocks_per_hour_including_measurement_bank",
        ):
            values = [r[metric] for r in rows if r[metric] is not None]
            if values:
                entry[metric] = statistics.fmean(values)
                if len(values) >= 2:
                    rng = random.Random(args.seed)
                    draws = sorted(
                        statistics.fmean(rng.choices(values, k=len(values)))
                        for _ in range(2000)
                    )
                    entry[metric + "_ci95_low"] = draws[49]
                    entry[metric + "_ci95_high"] = draws[1949]
        baseline_rows = [r for r in report if r["policy"] == "uniform"]
        entry.update(pooled_relative_yield(rows, baseline_rows, args.seed))
        intervals.append(entry)
    write_csv(directory / "policy_metrics.csv", intervals)
    write_csv(directory / "per_seed_metrics.csv", report)
    write_json(
        directory / "summary.json",
        {
            "results": results,
            "profile_rank_identifiable": calibration["profile_rank_identifiable"],
            "caution": "A flat output-length ranking makes short-vs-long selection unidentifiable; small block counts give wide uncertainty.",
        },
    )
    plot_selection(report, directory)
    (directory / "COMPLETED").write_text("complete\n")


def plot_selection(rows, directory):
    """Show block share, work share, and ticket throughput for the four policies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, metric, label in zip(
        axes,
        [
            "attacker_block_share",
            "attacker_work_share",
            "attacker_tickets_per_busy_second",
        ],
        ["Block share", "Completed complexity share", "Tickets / busy second"],
    ):
        values = [
            [r[metric] for r in rows if r["policy"] == p and r[metric] is not None]
            for p in SELECTION_POLICIES
        ]
        ax.boxplot(
            values,
            tick_labels=["Uniform", "Short K", "Long K", "Short N"],
            showfliers=False,
        )
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(directory / "selection.pdf")
    fig.savefig(directory / "selection.png", dpi=250)
    plt.close(fig)


def collision_cell(job):
    """Execute an independent empirical cell in a CPU worker."""
    args, schedule, bank, pool, rate, target, m, q = job
    rng = random.Random(args.seed + q)
    queries = [
        QuerySpec(
            f"request-{i}",
            template["prompt_length"],
            template["max_output_length"],
            trace_query_id=template["query_id"],
        )
        for i, template in enumerate(rng.choices(pool, k=q))
    ]
    difficulty = max(1, int(-math.expm1(-1 / (m * rate * target)) * LIMIT))
    # Both modes use the same integer threshold. Aggregate mode samples whether
    # any of C independent uniform hashes wins: 1 - (1 - D / 2**256)**C.
    probability = difficulty / LIMIT
    runs = []
    for repeat in range(args.collision_seeds):
        seed = args.seed + repeat + m * 100000 + q * 17 + int(target)
        result = simulate_race(
            queries,
            miner_count=m,
            difficulty_probability=probability,
            seed=seed,
            schedule=schedule,
            response_inclusion_probability=1,
            trace_bank=bank,
            trace_duration=True,
            require_trace_bank=True,
            max_events=None,
            literal_lottery=args.collision_lottery == "literal",
        )
        runs.append(
            {
                "target_block_time": target,
                "miners": m,
                "pool_size": q,
                "seed": seed,
                "difficulty": str(difficulty),
                "per_ticket_probability": probability,
                "lottery": args.collision_lottery,
                "adopted": result.adopted,
                "block_time": result.block_time,
                "completed_complexity": result.total_completed_complexity,
                "collision_complexity": result.discarded_complexity,
                "response_complexity": result.settled_complexity,
                "unfinished_complexity": result.unfinished_complexity,
                "wasted_work_pct": 100
                * result.discarded_complexity
                / result.total_completed_complexity
                if result.adopted
                else None,
            }
        )
    values = [r["wasted_work_pct"] for r in runs if r["adopted"]]
    cell = {
        "target_block_time": target,
        "miners": m,
        "pool_size": q,
        "seeds": len(runs),
        "adopted_races": len(values),
        "failed_races": len(runs) - len(values),
        "mean_wasted_work_pct": statistics.fmean(values) if values else math.nan,
        "stdev_wasted_work_pct": statistics.stdev(values)
        if len(values) > 1
        else math.nan,
    }
    return {"cell": cell, "runs": runs}


def wasted_work(args, schedule):
    """Use the authorized empirical-duration simulation with collision-only accounting."""
    from experiments.run_llm_experiments import _plot_wasted_work

    directory = args.output / "wasted_work"
    directory.mkdir(exist_ok=True)
    timings = read_jsonl(args.output / "preparation/timings.jsonl")
    pool = json.loads((args.output / "preparation/pool.json").read_text())
    bank = {q["query_id"]: [] for q in pool}
    for r in timings:
        bank[r["query_id"]].append(
            QueryTrace(r["output_length"], r["complexity"], r["duration"])
        )
    rate = sum(r["complexity"] for r in timings) / sum(r["duration"] for r in timings)
    cells, all_runs = [], []
    jobs = [
        (target, m, q)
        for target in args.targets
        for m in args.miner_grid
        for q in args.pool_grid
        if q > m
    ]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    pending = []
    for target, m, q in jobs:
        cell_path = directory / f"cell-{target:g}-{m}-{q}.json"
        if cell_path.exists():
            saved = json.loads(cell_path.read_text())
            cells.append(saved["cell"])
            all_runs.extend(saved["runs"])
        else:
            pending.append((args, schedule, bank, pool, rate, target, m, q))
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {executor.submit(collision_cell, job): job for job in pending}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Experiment 3 cells"
        ):
            saved = future.result()
            *_, target, m, q = futures[future]
            write_json(directory / f"cell-{target:g}-{m}-{q}.json", saved)
            cells.append(saved["cell"])
            all_runs.extend(saved["runs"])
    write_csv(directory / "cells.csv", cells)
    write_csv(directory / "runs.csv", all_runs)
    write_json(
        directory / "method.json",
        {
            "fresh_inference": False,
            "fresh_proof": False,
            "lottery": args.collision_lottery,
            "win_probability": "1 - (1 - D / 2**256)**C",
            "probability_evaluation": "-expm1(C * log1p(-D / 2**256))",
            "probability_assumption": "independent uniform ticket hashes; distributional equivalence, not identical outcomes to literal hashing",
            "hash_binding": (
                "seeded simulation identifier; ciphertext/proofs are not regenerated in Experiment 3"
                if args.collision_lottery == "literal"
                else "seeded Bernoulli draw per completion; no ticket hashes or ciphertext/proof regeneration in Experiment 3"
            ),
            "duration": "resampled isolated fresh-prover measurements",
            "collision_definition": "duplicate qid after crediting the full winning prefix and one response per other qid",
            "trace_pool": "request identifiers may share benchmark templates; collision identity is qid",
        },
    )
    _plot_wasted_work(cells, directory, args.targets)
    (directory / "COMPLETED").write_text("complete\n")


def parser():
    """Define one explicit command for the live campaign and each experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=["all", "calibrate", "liveness", "cherry-pick", "wasted-work"],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--binary",
        type=Path,
        default=ROOT / ".scratch/deep-prove/target/release/poml-prover",
    )
    p.add_argument(
        "--schedule", type=Path, default=ROOT / "config/gpt2_reference_schedule.json"
    )
    p.add_argument("--dataset-cache-file", type=Path)
    p.add_argument("--queries", type=int, default=32)
    p.add_argument("--prompt-lengths", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--max-output", type=int, default=32)
    p.add_argument("--profile-replicates", type=int, default=8)
    p.add_argument("--calibration-replicates", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--ticket-target", type=int, default=10000)
    p.add_argument("--target", type=float, default=300)
    p.add_argument("--miners", type=int, default=4)
    p.add_argument("--blocks", type=int, default=50)
    p.add_argument("--selection-blocks", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--collision-seeds", type=int, default=100)
    p.add_argument(
        "--collision-lottery",
        choices=["literal", "aggregate"],
        default="aggregate",
        help="Experiment 3 only: aggregate samples the equivalent probability of at least one winning ticket",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--targets", type=float, nargs="+", default=[300, 600, 900])
    p.add_argument(
        "--miner-grid",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    )
    p.add_argument(
        "--pool-grid",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
    )
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    """Validate the campaign manifest and run only requested unfinished stages."""
    args = parser().parse_args(argv)
    if (
        args.queries < 4
        or args.max_output < 1
        or args.ticket_target < 1
        or not math.isfinite(args.target)
        or args.target <= 0
        or not math.isfinite(args.alpha)
        or not 0 <= args.alpha <= 1
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
        or args.top_k < 0
        or not 0 < args.top_p <= 1
        or min(args.targets) <= 0
        or min(args.miner_grid) < 1
        or min(args.pool_grid) < 1
        or args.workers < 1
        or args.profile_replicates < 2
        or args.calibration_replicates < 1
        or min(args.prompt_lengths) < 2
        or max(args.prompt_lengths) + args.max_output > 64
        or min(
            args.blocks,
            args.selection_blocks,
            args.seeds,
            args.collision_seeds,
            args.miners,
        )
        < 1
    ):
        raise ValueError(
            "invalid grid: need >=4 prompts, >=2 profiles, positive repetitions, and N+cap<=64"
        )
    args.output = args.output.resolve()
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"resume", "command"}
    }
    config["schema"] = SCHEMA
    config["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    config["schedule_sha256"] = hashlib.sha256(args.schedule.read_bytes()).hexdigest()
    config["sources"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), *sorted((ROOT / "src/poml_sim").glob("*.py"))]
    }
    manifest = args.output / "campaign.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != config:
            raise ValueError(
                "existing campaign requires --resume with identical configuration and source"
            )
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing a nonempty output directory")
    write_json(manifest, config)
    import shutil

    snapshot = args.output / "source_snapshot"
    if not snapshot.exists():
        for relative in config["sources"]:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        for name in (
            "deepprove_protocol.patch",
            "deepprove_work.patch",
            "dp_crypto_work.patch",
        ):
            shutil.copy2(ROOT / "scripts" / name, snapshot / name)
        shutil.copy2(args.schedule, snapshot / "reference_schedule.json")
    if not (args.output / "environment.json").exists():
        import importlib.metadata
        import os
        import platform
        import subprocess

        environment = {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "gpu_selection": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "datasets",
                    "cryptography",
                    "numpy",
                    "tqdm",
                )
            },
        }
        for name, command in {
            "gpu": [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            "rust": ["rustc", "--version"],
            "deepprove_revision": [
                "git",
                "-C",
                str(ROOT / ".scratch/deep-prove"),
                "rev-parse",
                "HEAD",
            ],
        }.items():
            result = subprocess.run(
                command,
                cwd=ROOT / ".scratch/deep-prove",
                capture_output=True,
                text=True,
                check=True,
            )
            environment[name] = result.stdout.strip()
        write_json(args.output / "environment.json", environment)
    schedule = load_schedule(args.schedule)
    if args.command == "wasted-work":
        if not (args.output / "preparation/COMPLETED").exists():
            raise ValueError(
                "calibrate first to obtain actual inference/proof measurements"
            )
        wasted_work(args, schedule)
        return 0
    preparation_done = (args.output / "preparation/COMPLETED").exists()
    if preparation_done:
        calibration = json.loads(
            (args.output / "preparation/calibration.json").read_text()
        )
        ready = calibration["prover"]
    else:
        with LiveProver(
            args.binary,
            args.output / "prover",
            args.device,
            threads=args.threads,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ) as prover:
            write_json(args.output / "prover/handshake.json", prover.ready)
            calibration = prepare(args, prover, schedule)
            ready = prover.ready
    timing_path = args.output / "preparation/timings.jsonl"
    bank = MeasuredBank(
        json.loads((args.output / "preparation/pool.json").read_text()),
        read_jsonl(timing_path),
        schedule,
        calibration["scale"],
    )
    if args.command in {"all", "liveness"}:
        run_chain(
            args,
            ready,
            bank,
            schedule,
            calibration,
            args.output / "liveness",
            args.blocks,
            "uniform",
            args.seed,
        )
        plot_liveness(args)
    if args.command in {"all", "cherry-pick"}:
        cherry_pick(args, ready, bank, schedule, calibration)
    if args.command == "all":
        wasted_work(args, schedule)
        (args.output / "COMPLETED").write_text("all three experiments complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
