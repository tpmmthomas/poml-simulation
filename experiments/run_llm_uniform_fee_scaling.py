#!/usr/bin/env python3
"""Replay the original uniform-fee grid with pooled GPT-2 work measurements.

Query identifiers share one empirical (duration, complexity) distribution.
The first completion for an identifier is useful; later completions are waste.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from decimal import Decimal, localcontext
import hashlib
import heapq
import json
import math
import multiprocessing
from pathlib import Path
import random
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.exp5_uniform_fee_collisions import derive_seed  # noqa: E402
from experiments.exp5c_m_q_heatmap import (  # noqa: E402
    MINER_GRID,
    QUERY_GRID,
    design_grid,
)
from poml_sim.llm_simulation import LazyPermutation  # noqa: E402

LIMIT = 1 << 256


def load_measurements(path):
    """Validate genuine bank records and retain unscaled appendix complexity."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError("measurement bank is empty")
    samples = []
    for row in rows:
        duration, complexity = row["duration"], row["raw_complexity"]
        if not all(
            row.get(k) is True for k in ("verified", "fresh_inference", "fresh_proof")
        ):
            raise ValueError("measurement bank must contain verified fresh pairs")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("measurement durations must be finite and positive")
        if type(complexity) is not int or complexity <= 0:
            raise ValueError("measurement complexities must be positive integers")
        if complexity != sum(row["reference_counts"]["combined"].values()):
            raise ValueError(
                "raw complexity disagrees with unit-weight reference counts"
            )
        samples.append((duration, complexity))
    return samples


def calibrated_difficulty(samples, miners, target):
    """Generalize the original E[T]/(M tau) threshold to complexity units."""
    with localcontext() as context:
        context.prec = 100
        # Dividing sums equals E[T]/E[C] without separately rounding either mean.
        probability = sum(Decimal(str(t)) for t, _ in samples) / (
            Decimal(sum(c for _, c in samples)) * miners * Decimal(str(target))
        )
        difficulty = int(Decimal(LIMIT) * probability)
    if not 0 < difficulty < LIMIT:
        raise ValueError("calibrated per-unit threshold must lie in (0, 2**256)")
    return difficulty


def simulate_uniform_race(samples, *, miners, queries, difficulty, seed):
    """Run one homogeneous race, retaining completed first/duplicate work totals.

    This occupancy experiment treats identifiers as exchangeable requests, not
    fixed benchmark texts. A measured duration and its own C always stay paired.
    Independent permutations are generated lazily to avoid M*Q allocation.
    """
    if miners < 1 or queries < 1 or not samples:
        raise ValueError("positive miner/query counts and measurements required")
    if not 0 < difficulty < LIMIT:
        raise ValueError("difficulty must lie in (0, 2**256)")
    if any(
        not math.isfinite(t) or t <= 0 or type(c) is not int or c <= 0
        for t, c in samples
    ):
        raise ValueError(
            "samples require finite positive durations and integer complexities"
        )
    rng = random.Random(derive_seed(seed, "attempts"))
    lottery_rng = random.Random(derive_seed(seed, "lottery"))
    orderings = [
        LazyPermutation(queries, random.Random(derive_seed(seed, "permutation", m)))
        for m in range(miners)
    ]
    probabilities = {
        c: -math.expm1(c * math.log1p(-difficulty / LIMIT)) for _, c in samples
    }
    queue = []
    sequence = 0

    def start(miner, time):
        nonlocal sequence
        query = orderings[miner].draw()
        if query is None:
            return
        duration, complexity = rng.choice(samples)
        sequence += 1
        # Empirical sampling creates exact timing ties. Randomize their order,
        # as in the original experiment, before adopting the first winner.
        heapq.heappush(
            queue,
            (time + duration, rng.getrandbits(64), sequence, miner, query, complexity),
        )

    for miner in range(miners):
        start(miner, 0.0)
    seen = set()
    total = first = duplicate = count = collisions = 0
    winner = None
    completion = 0.0
    while queue:
        completion, _, _, miner, query, complexity = heapq.heappop(queue)
        count += 1
        total += complexity
        if query in seen:
            duplicate += complexity
            collisions += 1
        else:
            seen.add(query)
            first += complexity
        if lottery_rng.random() < probabilities[complexity]:
            winner = miner
            break
        start(miner, completion)
    return {
        "adopted": winner is not None,
        "termination_reason": "adopted"
        if winner is not None
        else "query_pool_exhausted",
        "block_time": completion,
        "completed_pairs": count,
        "unique_queries": len(seen),
        "duplicate_pairs": collisions,
        "completed_complexity": total,
        "first_completed_complexity": first,
        "collision_complexity": duplicate,
        "canceled_attempts": len(queue),
        "wasted_work_pct": 100 * duplicate / total if winner is not None else None,
    }


def run_cell(job):
    """Return all seeded races and conditional mean/sample SD for one cell."""
    samples, target, miners, queries, repeats, seed_root = job
    difficulty = calibrated_difficulty(samples, miners, target)
    runs = []
    for repeat in range(repeats):
        seed = derive_seed(
            seed_root, "uniform-fee-race", target, miners, queries, repeat
        )
        runs.append(
            {
                "target_block_time": target,
                "miners": miners,
                "pool_size": queries,
                "replicate": repeat,
                "seed": seed,
                "lottery": "aggregate",
                "difficulty": str(difficulty),
                "uniform_fee": 1,
                **simulate_uniform_race(
                    samples,
                    miners=miners,
                    queries=queries,
                    difficulty=difficulty,
                    seed=seed,
                ),
            }
        )
    values = [r["wasted_work_pct"] for r in runs if r["adopted"]]
    return {
        "cell": {
            "target_block_time": target,
            "miners": miners,
            "pool_size": queries,
            "seeds": repeats,
            "adopted_races": len(values),
            "failed_races": repeats - len(values),
            "mean_wasted_work_pct": statistics.fmean(values) if values else None,
            "stdev_wasted_work_pct": statistics.stdev(values)
            if len(values) > 1
            else None,
        },
        "runs": runs,
    }


def write_json(path, value):
    """Atomically checkpoint JSON so interrupted cells can be rerun safely."""
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def write_csv(path, rows):
    """Write stable report columns from nonempty experiment rows."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_campaign(args):
    """Run or resume a source-bound campaign without changing the input bank."""
    if args.replicates < 2 or args.workers < 1:
        raise ValueError("at least two replicates and one worker required")
    if (
        not args.targets
        or len(set(args.targets)) != len(args.targets)
        or any(not math.isfinite(t) or t <= 0 for t in args.targets)
    ):
        raise ValueError("targets must be unique, finite and positive")
    for grid in (args.miner_grid, args.pool_grid):
        if not grid or len(set(grid)) != len(grid):
            raise ValueError("grid values must be nonempty and unique")
    cells = design_grid(tuple(args.miner_grid), tuple(args.pool_grid), full_grid=True)
    source = args.input.resolve()
    measurement_path = source / "preparation/timings.jsonl"
    if not (source / "preparation/COMPLETED").exists():
        raise ValueError("source measurement bank is incomplete")
    samples = load_measurements(measurement_path)
    sources = [
        Path(__file__),
        ROOT / "src/poml_sim/llm_simulation.py",
        ROOT / "experiments/exp5_uniform_fee_collisions.py",
        ROOT / "experiments/exp5c_m_q_heatmap.py",
    ]
    config = {
        "schema": "poml-llm-uniform-fee-scaling-1",
        "input": str(source),
        "targets": args.targets,
        "miner_grid": args.miner_grid,
        "pool_grid": args.pool_grid,
        "collision_seeds": args.replicates,
        "seed": args.seed,
        "full_grid": True,
        "collision_metric": "first_completion",
        "sampling": "pooled paired (duration, raw C), iid with replacement, independent of qid",
        "lottery": "aggregate",
        "difficulty": "floor(2**256 * E[T] / (M * target * E[C]))",
        "win_probability": "1 - (1 - D / 2**256)**C",
        "uniform_fee": 1,
        "sample_count": len(samples),
        "mean_duration_s": statistics.fmean(t for t, _ in samples),
        "stdev_duration_s": statistics.stdev(t for t, _ in samples),
        "mean_complexity": statistics.fmean(c for _, c in samples),
        "input_sha256": hashlib.sha256(measurement_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "failed_race_policy": "record exhausted races; omit from mean and sample SD",
        "fresh_inference": False,
        "fresh_proof": False,
    }
    output = args.output.resolve()
    manifest = output / "campaign.json"
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text()) != config:
            raise ValueError(
                "resume requires identical inputs, design and simulation sources"
            )
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(manifest, config)
        shutil.copyfile(measurement_path, output / "measurements.jsonl")
        for path in sources:
            destination = output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    jobs = [
        (
            samples,
            t,
            int(c["miners"]),
            int(c["query_pool_size"]),
            args.replicates,
            args.seed,
        )
        for t in args.targets
        for c in cells
    ]
    results, pending = [], []
    for job in jobs:
        _, target, miners, queries, *_ = job
        path = output / f"cell-{target:g}-{miners}-{queries}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append((job, path))
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {executor.submit(run_cell, job): path for job, path in pending}
        for future in as_completed(futures):
            result = future.result()
            write_json(futures[future], result)
            results.append(result)
            print(
                f"[uniform fees] {len(results)}/{len(jobs)} cells complete", flush=True
            )
    results.sort(
        key=lambda r: tuple(
            r["cell"][k] for k in ("target_block_time", "miners", "pool_size")
        )
    )
    write_csv(output / "cells.csv", [r["cell"] for r in results])
    write_csv(output / "runs.csv", [trial for r in results for trial in r["runs"]])
    from experiments.plot_llm_paper_results import (
        collision_plots,
        collision_statistics,
        read_csv,
    )

    report = collision_statistics(
        read_csv(output / "cells.csv"), read_csv(output / "runs.csv"), config
    )
    write_json(output / "summary.json", report)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    collision_plots(read_csv(output / "cells.csv"), config, figures)
    (output / "COMPLETED").write_text("complete\n")
    return report


def main():
    """Expose a CPU-only rerun of the original uniform-fee design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[300.0, 600.0, 900.0]
    )
    parser.add_argument("--miner-grid", type=int, nargs="+", default=list(MINER_GRID))
    parser.add_argument("--pool-grid", type=int, nargs="+", default=list(QUERY_GRID))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_campaign(args), indent=2))


if __name__ == "__main__":
    main()
