#!/usr/bin/env python3
"""Experiment 2: Wasted work and network efficiency in PoML.

Two sweeps, each 10 blocks per configuration:

A. Block-time sweep (fixing miners = 4): expected block times 200 / 300 / 400 s
B. Miner-count sweep (fixing expected block time = 300 s): miners = 2 / 4 / 8

For each run, we compute:
    completed_proofs   = "Miner X: proof N/M done" lines across all miners
                         (all proofs produced, WIN or miss, whether included or not)
    included_proofs    = sum of lottery_attempts over confirmed blocks
    wasted_ratio       = 1 - included / completed

Difficulty scaling:
- Baseline: logs/run_20260423_050539.log (difficulty 0x3fff...ff, 4 miners, 214.28s avg).
- Block-time axis (4 miners fixed):  diff(t) = baseline * (214.28 / t)
- Miner-count axis (300 s target, N miners): diff_300s_4m * (4 / N)
  (per-miner rate is ~constant, so aggregate rate scales with N).

Usage:
    python experiments/exp2_wasted_work.py [--blocks 10]
    python experiments/exp2_wasted_work.py --smoke   # 2-block dry run
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

from experiments.utils import (  # noqa: E402
    parse_log_for_calibration,
    run_poml,
    scale_difficulty,
    summarize,
)


RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CALIBRATION_LOG = _PROJECT_ROOT / "logs" / "run_20260423_050539.log"


def _console_logger() -> logging.Logger:
    logger = logging.getLogger("exp2")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[exp2] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _scale_by_miners(difficulty_hex: str, baseline_miners: int, new_miners: int) -> str:
    """Keep expected block time constant as miner count changes.

    Aggregate hashrate (or proof-rate in PoML) is proportional to number of
    miners; halving miners doubles expected block time unless difficulty is
    also halved. So diff_new = diff_old * (baseline_miners / new_miners).
    """
    old_int = int(difficulty_hex, 16)
    new_int = int(old_int * baseline_miners / new_miners)
    new_int = max(1, min(new_int, (1 << 256) - 1))
    return "0x" + format(new_int, "x").rjust(64, "0")


def run_one(
    label: str,
    num_miners: int,
    difficulty: str,
    num_blocks: int,
    target_seconds: float,
) -> dict:
    """Drive a single PoML run and return a row suitable for CSV output."""
    log = _console_logger()

    overrides = {
        "num_miners": num_miners,
        "num_queries": num_blocks,
        "initial_burst": num_blocks,
        "steady_interval_s": 1.0,
        "max_queries_per_block": 1,
        "difficulty": difficulty,
        # 3x the expected wall time as a safety ceiling so the driver doesn't
        # hang forever if difficulty is mis-calibrated.
        "simulation_timeout": num_blocks * target_seconds * 3,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"exp2_{label}_{stamp}.log"
    log.info("[%s] miners=%d diff=%s... blocks=%d", label, num_miners, difficulty[:20], num_blocks)

    result = run_poml(overrides, log_path, timeout=overrides["simulation_timeout"])

    stats = summarize(result.block_times)
    log.info(
        "[%s] done: blocks=%d completed=%d included=%d wasted=%d (%.1f%%) mean_bt=%.2fs",
        label, result.total_blocks, result.completed_proofs, result.included_proofs,
        result.wasted_proofs, 100 * result.wasted_ratio, stats["mean"],
    )

    return {
        "label": label,
        "num_miners": num_miners,
        "target_block_time_s": target_seconds,
        "difficulty": difficulty,
        "blocks_confirmed": result.total_blocks,
        "completed_proofs": result.completed_proofs,
        "included_proofs": result.included_proofs,
        "wasted_proofs": result.wasted_proofs,
        "wasted_ratio": result.wasted_ratio,
        "mean_block_time_s": stats["mean"],
        "stdev_block_time_s": stats["stdev"],
        "log_path": str(result.log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=10,
                        help="Blocks per configuration (default: 10)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny 2-block sanity run with easy difficulty")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = _console_logger()

    old_diff, observed = parse_log_for_calibration(CALIBRATION_LOG)
    log.info("Baseline from %s: diff=%s..., observed=%.2fs @ 4 miners",
             CALIBRATION_LOG.name, old_diff[:20], observed)

    if args.smoke:
        # Maximum difficulty: every proof wins, so 2 blocks completes in ~1 min.
        easy = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        blocks_sweep = [
            ("blocktime_smoke_miners2", 2, easy, 2, 300.0),
        ]
        miner_sweep = [
            ("minercount_smoke_miners4", 4, easy, 2, 300.0),
        ]
    else:
        n = args.blocks
        # Sweep A: vary expected block time, fix miners=4.
        blocks_sweep = [
            (f"blocktime{t}s_miners4",
             4,
             scale_difficulty(old_diff, observed, float(t)),
             n,
             float(t))
            for t in (200, 300, 400)
        ]
        # Sweep B: fix 300 s target, vary miner count. Start from the 4-miner
        # 300 s difficulty then rescale by the miner-count factor so the
        # expected block time stays ~300 s across the axis.
        diff_300s_4m = scale_difficulty(old_diff, observed, 300.0)
        miner_sweep = [
            (f"blocktime300s_miners{m}",
             m,
             _scale_by_miners(diff_300s_4m, baseline_miners=4, new_miners=m),
             n,
             300.0)
            for m in (2, 4, 8)
        ]

    # Run Sweep A
    rows_a = []
    for label, miners, diff, n_blocks, target in blocks_sweep:
        rows_a.append(run_one(label, miners, diff, n_blocks, target))

    # Run Sweep B
    rows_b = []
    for label, miners, diff, n_blocks, target in miner_sweep:
        rows_b.append(run_one(label, miners, diff, n_blocks, target))

    # Emit CSVs
    cols = ["label", "num_miners", "target_block_time_s", "difficulty",
            "blocks_confirmed", "completed_proofs", "included_proofs",
            "wasted_proofs", "wasted_ratio", "mean_block_time_s",
            "stdev_block_time_s", "log_path"]

    if rows_a:
        path_a = RESULTS_DIR / "exp2_blocktime_sweep.csv"
        with open(path_a, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_a)
        log.info("Wrote %s", path_a)

    if rows_b:
        path_b = RESULTS_DIR / "exp2_miner_sweep.csv"
        with open(path_b, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_b)
        log.info("Wrote %s", path_b)

    # Console summary
    print()
    print(f"{'Config':<28} {'Miners':>6} {'TgtBT':>6} {'Blocks':>7} "
          f"{'Done':>5} {'Incl':>5} {'Waste':>7} {'MeanBT':>8}")
    print("-" * 88)
    for r in rows_a + rows_b:
        print(f"{r['label']:<28} {r['num_miners']:>6} {r['target_block_time_s']:>6.0f} "
              f"{r['blocks_confirmed']:>7} {r['completed_proofs']:>5} "
              f"{r['included_proofs']:>5} {100*r['wasted_ratio']:>6.1f}% "
              f"{r['mean_block_time_s']:>8.1f}")


if __name__ == "__main__":
    main()
