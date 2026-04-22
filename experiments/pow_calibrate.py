#!/usr/bin/env python3
"""Calibrate PoW difficulty for a target expected block time.

Runs a short single-core SHA-256 benchmark to measure h/s, scales to the
aggregate hashrate across `num_miners`, then picks a target such that the
expected number of hashes to find a block equals hashrate * target_seconds.

Usage:
    python experiments/pow_calibrate.py [--target 300] [--miners 4] [--benchmark-seconds 5]

Writes results to experiments/results/pow_calibration.json so the Exp 1
driver can reuse the calibration without re-benchmarking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.pow_sim import benchmark_hashrate  # noqa: E402
from experiments.utils import DEFAULT_CPU_LIMIT, apply_cpu_limit  # noqa: E402


RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"


def calibrate(target_seconds: float, num_miners: int, benchmark_seconds: float) -> dict:
    # Pin the benchmark to the same 16-core budget used by the real runs so
    # the measured hashrate matches what the simulator will actually see.
    apply_cpu_limit(DEFAULT_CPU_LIMIT, num_miners)
    print(f"Benchmarking SHA-256 for {benchmark_seconds}s (single core)...")
    per_core_hps = benchmark_hashrate(seconds=benchmark_seconds)
    aggregate_hps = per_core_hps * num_miners

    # Expected hashes to produce a block ≈ hashrate * target_seconds.
    # Per-attempt success probability = target_int / 2^256, so
    # E[attempts] = 2^256 / target_int -> target_int = 2^256 / E[attempts].
    expected_attempts = aggregate_hps * target_seconds
    if expected_attempts < 2:
        raise ValueError("Aggregate hashrate too low for the requested target — "
                         "increase benchmark_seconds or reduce num_miners.")
    target_int = (1 << 256) // int(expected_attempts)
    target_hex = "0x" + format(target_int, "x").rjust(64, "0")

    result = {
        "per_core_hashes_per_second": per_core_hps,
        "aggregate_hashes_per_second": aggregate_hps,
        "num_miners": num_miners,
        "target_block_time_seconds": target_seconds,
        "expected_attempts_per_block": expected_attempts,
        "difficulty_hex": target_hex,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=300.0,
                        help="Target expected block time, seconds (default: 300)")
    parser.add_argument("--miners", type=int, default=4,
                        help="Number of PoW miners (default: 4)")
    parser.add_argument("--benchmark-seconds", type=float, default=5.0,
                        help="Length of the hashrate benchmark (default: 5.0)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "pow_calibration.json")
    args = parser.parse_args()

    # Minimal logging so apply_cpu_limit() confirmation shows on stdout.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = calibrate(args.target, args.miners, args.benchmark_seconds)

    print()
    print("=== PoW Calibration ===")
    print(f"  Per-core hashrate:      {result['per_core_hashes_per_second']:.2e} h/s")
    print(f"  Miners:                 {result['num_miners']}")
    print(f"  Aggregate hashrate:     {result['aggregate_hashes_per_second']:.2e} h/s")
    print(f"  Target block time:      {result['target_block_time_seconds']:.1f} s")
    print(f"  Expected hashes/block:  {result['expected_attempts_per_block']:.2e}")
    print(f"  Calibrated difficulty:  {result['difficulty_hex']}")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
