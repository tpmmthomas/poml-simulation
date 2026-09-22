"""Actual parallel double-SHA-256 races for the liveness comparison.

This is a Bitcoin-style hash lottery over 80-byte headers, not a Bitcoin node.
Calibration and races use the same number of CPU worker processes.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import math
import multiprocessing
import time

from .lottery import LIMIT

_STOP = None


def _initialize(stop):
    global _STOP
    _STOP = stop


def _hashes(arguments):
    prefix, difficulty, calibration_seconds = arguments
    start, count, digest = time.perf_counter(), 0, None
    while not _STOP.is_set():
        header = prefix + count.to_bytes(8, "little")
        digest = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        count += 1
        if calibration_seconds:
            if count % 1024 == 0 and time.perf_counter() - start >= calibration_seconds:
                break
        elif int.from_bytes(digest, "little") < difficulty:
            _STOP.set()
            return {
                "won": True,
                "seconds": time.perf_counter() - start,
                "hashes": count,
                "digest": digest.hex(),
                "header": header.hex(),
            }
    return {"won": False, "seconds": time.perf_counter() - start, "hashes": count}


def run_pow(*, blocks, miners, target, seed=42, calibration_seconds=0.5):
    """Calibrate observed hash rate, then measure fresh CPU hash races."""
    if (
        min(blocks, miners) < 1
        or not math.isfinite(target)
        or target <= 0
        or calibration_seconds <= 0
    ):
        raise ValueError("positive block/miner counts, target and calibration duration required")
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    parent = bytes(32)
    with ProcessPoolExecutor(
        max_workers=miners, mp_context=context, initializer=_initialize, initargs=(stop,)
    ) as workers:

        def prefixes(height):
            return [
                hashlib.shake_256(parent + f"pow:{seed}:{height}:{m}".encode()).digest(72)
                for m in range(miners)
            ]

        measured = list(workers.map(_hashes, [(p, 0, calibration_seconds) for p in prefixes(-1)]))
        rate = sum(row["hashes"] / row["seconds"] for row in measured)
        difficulty = max(1, min(LIMIT, int(LIMIT / (rate * target))))
        records = []
        for height in range(blocks):
            stop.clear()
            start = time.perf_counter()
            rows = list(workers.map(_hashes, [(p, difficulty, 0) for p in prefixes(height)]))
            # Include process dispatch and cancellation overhead in observed time.
            elapsed = time.perf_counter() - start
            winner = min((r for r in rows if r["won"]), key=lambda r: r["seconds"])
            records.append(
                {
                    "block_time": elapsed,
                    "hashes": sum(r["hashes"] for r in rows),
                    "winning_digest": winner["digest"],
                    "header": winner["header"],
                    "parent": parent.hex(),
                }
            )
            parent = bytes.fromhex(winner["digest"])
    return {
        "mode": "measured double-SHA-256 CPU races",
        "hashes_per_second": rate,
        "difficulty": str(difficulty),
        "records": records,
    }
