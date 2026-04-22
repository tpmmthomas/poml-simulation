#!/usr/bin/env python3
"""Experiment 1: Block generation stability (PoML vs. simplified PoW).

Steps:
1. Derive the PoML difficulty that targets a 300 s expected block time by
   rescaling the difficulty used in logs/run_20260423_050539.log (we know
   that run produced an average of 214.28 s per block at its difficulty).
2. Run PoML for 50 blocks with the derived difficulty (default config).
3. Load the PoW calibration from results/pow_calibration.json and run the
   simplified PoW simulator for 50 blocks with the same number of miners.
4. Emit per-block CSVs, a summary CSV, and a console side-by-side table.

Usage:
    python experiments/exp1_block_time_stability.py [--blocks 50] [--target 300]
    python experiments/exp1_block_time_stability.py --smoke   # quick 3-block run
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.pow_calibrate import calibrate as pow_calibrate  # noqa: E402
from experiments.pow_sim import PowSimulator  # noqa: E402
from experiments.utils import (  # noqa: E402
    DEFAULT_CPU_LIMIT,
    apply_cpu_limit,
    parse_log_for_calibration,
    run_poml,
    scale_difficulty,
    summarize,
)
import json  # noqa: E402


RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CALIBRATION_LOG = _PROJECT_ROOT / "logs" / "run_20260423_050539.log"
POW_CAL_FILE = RESULTS_DIR / "pow_calibration.json"


def _console_logger() -> logging.Logger:
    """Console logger used by the driver itself (separate from per-run handlers)."""
    logger = logging.getLogger("exp1")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[exp1] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def run_poml_phase(num_blocks: int, target_seconds: float, smoke: bool) -> dict:
    log = _console_logger()

    if smoke:
        # Maximum difficulty (every proof wins the lottery) — one proof per block,
        # so 3 blocks @ 4 miners takes ~1-2 min instead of ~5 min.
        difficulty = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        num_blocks = 3
        log.info("SMOKE mode: difficulty=max (p=1.0), blocks=%d", num_blocks)
    else:
        old_diff, observed = parse_log_for_calibration(CALIBRATION_LOG)
        difficulty = scale_difficulty(old_diff, observed, target_seconds)
        log.info(
            "PoML calibration: reference_log=%s -> old_diff=%s..., observed=%.2fs, target=%.2fs",
            CALIBRATION_LOG.name, old_diff[:20], observed, target_seconds,
        )
        log.info("PoML derived difficulty: %s", difficulty)

    # One block per query (max_queries_per_block=1, num_queries=num_blocks) so the
    # coordinator terminates cleanly after exactly `num_blocks` confirmations.
    # Generous simulation_timeout: 3x the expected wall time as a safety ceiling.
    overrides = {
        "num_miners": 4,
        "num_queries": num_blocks,
        "initial_burst": num_blocks,  # drop all into mempool at t=0
        "steady_interval_s": 1.0,     # irrelevant once the burst lands
        "max_queries_per_block": 1,
        "difficulty": difficulty,
        "simulation_timeout": (num_blocks * target_seconds * 3) if not smoke else 600.0,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"exp1_poml_{stamp}.log"
    log.info("Running PoML (logs -> %s)...", log_path)

    result = run_poml(overrides, log_path, timeout=overrides["simulation_timeout"])

    log.info(
        "PoML done: blocks=%d, completed_proofs=%d, included_proofs=%d, wasted=%d (%.1f%%)",
        result.total_blocks, result.completed_proofs, result.included_proofs,
        result.wasted_proofs, 100 * result.wasted_ratio,
    )
    stats = summarize(result.block_times)
    log.info("PoML block times: mean=%.2fs, min=%.2fs, max=%.2fs, stdev=%.2fs, variance=%.2f",
             stats["mean"], stats["min"], stats["max"], stats["stdev"], stats["variance"])

    # Per-block CSV
    csv_path = RESULTS_DIR / "exp1_poml_blocks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_index", "time_since_last_block_s"])
        for i, t in enumerate(result.block_times):
            w.writerow([i + 1, f"{t:.4f}"])
    log.info("Wrote %s", csv_path)

    return {
        "mechanism": "PoML",
        "difficulty": difficulty,
        "num_miners": overrides["num_miners"],
        "block_times": result.block_times,
        "stats": stats,
    }


def run_pow_phase(num_blocks: int, target_seconds: float, smoke: bool) -> dict:
    log = _console_logger()

    if smoke:
        # Trivial difficulty so each block is found instantly.
        difficulty_hex = "0x" + "f" * 64
        num_miners = 2
        num_blocks = 3
        log.info("SMOKE mode: PoW difficulty=trivial, miners=%d, blocks=%d", num_miners, num_blocks)
    elif POW_CAL_FILE.exists():
        with open(POW_CAL_FILE) as f:
            cal = json.load(f)
        difficulty_hex = cal["difficulty_hex"]
        num_miners = cal["num_miners"]
        log.info("Loaded PoW calibration from %s (difficulty=%s..., miners=%d, target=%.1fs)",
                 POW_CAL_FILE.name, difficulty_hex[:20], num_miners, cal["target_block_time_seconds"])
    else:
        log.info("No prior calibration file; running inline benchmark (5s)...")
        cal = pow_calibrate(target_seconds, num_miners=4, benchmark_seconds=5.0)
        difficulty_hex = cal["difficulty_hex"]
        num_miners = cal["num_miners"]

    # Pin PoW phase to the same CPU budget as the PoML phase for apples-to-apples
    # wall-clock comparison. Attach a stdout handler so per-block progress shows
    # up in run_all.sh's master log.
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(logging.INFO)
    stdout_h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(stdout_h)

    try:
        apply_cpu_limit(DEFAULT_CPU_LIMIT, num_miners)
        sim = PowSimulator(num_miners=num_miners, difficulty_hex=difficulty_hex)
        blocks = sim.run(num_blocks=num_blocks)
    finally:
        root.removeHandler(stdout_h)
        stdout_h.close()

    block_times = [b.time_since_last_block for b in blocks]
    stats = summarize(block_times)
    log.info("PoW block times: mean=%.2fs, min=%.2fs, max=%.2fs, stdev=%.2fs, variance=%.2f",
             stats["mean"], stats["min"], stats["max"], stats["stdev"], stats["variance"])

    csv_path = RESULTS_DIR / "exp1_pow_blocks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_index", "miner_id", "nonce", "time_since_last_block_s"])
        for b in blocks:
            w.writerow([b.height, b.miner_id, b.nonce, f"{b.time_since_last_block:.4f}"])
    log.info("Wrote %s", csv_path)

    return {
        "mechanism": "PoW",
        "difficulty": difficulty_hex,
        "num_miners": num_miners,
        "block_times": block_times,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--target", type=float, default=300.0,
                        help="Target expected block time, seconds")
    parser.add_argument("--skip-poml", action="store_true")
    parser.add_argument("--skip-pow", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick 3-block sanity run (trivial difficulty)")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = _console_logger()

    poml = pow = None
    if not args.skip_poml:
        poml = run_poml_phase(args.blocks, args.target, args.smoke)
    if not args.skip_pow:
        pow = run_pow_phase(args.blocks, args.target, args.smoke)

    # Summary CSV
    summary_path = RESULTS_DIR / "exp1_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mechanism", "difficulty", "num_miners", "blocks_measured",
                    "mean_s", "min_s", "max_s", "stdev_s", "variance"])
        for r in (poml, pow):
            if r is None:
                continue
            s = r["stats"]
            w.writerow([r["mechanism"], r["difficulty"], r["num_miners"], s["count"],
                        f"{s['mean']:.4f}", f"{s['min']:.4f}", f"{s['max']:.4f}",
                        f"{s['stdev']:.4f}", f"{s['variance']:.4f}"])
    log.info("Wrote %s", summary_path)

    # Console table
    print()
    print(f"{'Mechanism':<10} {'N':>4} {'Mean':>10} {'Min':>10} {'Max':>10} {'Stdev':>10} {'Var':>12}")
    print("-" * 76)
    for r in (poml, pow):
        if r is None:
            continue
        s = r["stats"]
        print(f"{r['mechanism']:<10} {s['count']:>4} {s['mean']:>10.2f} {s['min']:>10.2f} "
              f"{s['max']:>10.2f} {s['stdev']:>10.2f} {s['variance']:>12.2f}")


if __name__ == "__main__":
    main()
