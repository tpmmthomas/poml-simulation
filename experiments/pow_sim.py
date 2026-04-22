"""Simplified Bitcoin-style PoW simulator.

Each miner hashes SHA-256(prev_hash || miner_id || nonce) in a tight loop,
incrementing nonce until `int(hash, 'big') < target`. The first miner to
succeed publishes the block; the driver terminates all workers, seeds the
next round with the winner's hash, and loops until `num_blocks` are produced.

This is intentionally stripped down:
- no transactions / fees / blockchain validation
- no fork resolution
- no network latency modelling

We only measure the per-block wall-clock interval so we can compare the
block-time distribution against PoML at the same expected block time.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Spread miners across the nonce space so they do non-overlapping work.
# 2**48 is larger than any realistic per-block hash count at our target
# difficulty (~1e8 hashes/block aggregate) so collisions are effectively zero.
_MINER_NONCE_STRIDE = 1 << 48


def _miner_worker(miner_id: int, prev_hash: bytes, target: int,
                  stop_event, result_queue) -> None:
    """Worker loop: grind nonces, push the first winning one to the driver."""
    # Local aliases for speed — Python attribute lookups in a hot loop add up.
    sha256 = hashlib.sha256
    miner_id_bytes = miner_id.to_bytes(4, "big")
    nonce = miner_id * _MINER_NONCE_STRIDE

    # Check the stop flag periodically rather than every attempt — polling a
    # shared event on each hash halves throughput on this machine.
    CHECK_INTERVAL = 4096
    counter = 0

    while True:
        counter += 1
        if counter >= CHECK_INTERVAL:
            if stop_event.is_set():
                return
            counter = 0

        nonce_bytes = nonce.to_bytes(12, "big")
        h = sha256(prev_hash + miner_id_bytes + nonce_bytes).digest()
        if int.from_bytes(h, "big") < target:
            result_queue.put({
                "miner_id": miner_id,
                "nonce": nonce,
                "hash": h,
                "time": time.time(),
            })
            return
        nonce += 1


@dataclass
class PowBlock:
    height: int
    miner_id: int
    prev_hash: str
    block_hash: str
    nonce: int
    time_since_last_block: float
    produced_at: float


class PowSimulator:
    """Multi-process SHA-256 PoW lottery simulator."""

    def __init__(self, num_miners: int, difficulty_hex: str, genesis_seed: bytes = b"POW_GENESIS"):
        self.num_miners = num_miners
        self.difficulty_hex = difficulty_hex
        self.target = int(difficulty_hex, 16)
        self.genesis_seed = genesis_seed

    def run(self, num_blocks: int, timeout_per_block: Optional[float] = None) -> list[PowBlock]:
        """Produce `num_blocks` blocks, returning a PowBlock per confirmed block."""
        logger.info(
            "PoW sim starting: miners=%d, difficulty=%s..., blocks=%d",
            self.num_miners, self.difficulty_hex[:20], num_blocks,
        )

        ctx = multiprocessing.get_context("spawn")
        prev_hash = hashlib.sha256(self.genesis_seed).digest()
        start = time.time()
        last_block_time = start
        blocks: list[PowBlock] = []

        for h in range(1, num_blocks + 1):
            stop_event = ctx.Event()
            result_queue: multiprocessing.Queue = ctx.Queue()

            workers = [
                ctx.Process(
                    target=_miner_worker,
                    args=(mid, prev_hash, self.target, stop_event, result_queue),
                    daemon=True,
                )
                for mid in range(self.num_miners)
            ]
            for w in workers:
                w.start()

            # Block until someone finds a valid nonce (or optional per-block timeout).
            winner = result_queue.get(timeout=timeout_per_block) if timeout_per_block else result_queue.get()

            stop_event.set()
            for w in workers:
                w.join(timeout=2.0)
                if w.is_alive():
                    w.terminate()

            now = winner["time"]
            dt = now - last_block_time
            last_block_time = now

            block = PowBlock(
                height=h,
                miner_id=winner["miner_id"],
                prev_hash=prev_hash.hex(),
                block_hash=winner["hash"].hex(),
                nonce=winner["nonce"],
                time_since_last_block=dt,
                produced_at=now,
            )
            blocks.append(block)
            logger.info(
                "PoW block %d by miner %d after %.2fs (nonce=%d, hash=%s...)",
                h, block.miner_id, dt, block.nonce, block.block_hash[:16],
            )
            prev_hash = winner["hash"]

        logger.info("PoW sim done: %d blocks in %.1fs", num_blocks, time.time() - start)
        return blocks


def benchmark_hashrate(seconds: float = 5.0) -> float:
    """Measure single-core SHA-256 hashrate on this machine (hashes/second).

    Used by pow_calibrate.py to pick a difficulty target. Intentionally matches
    the inner loop of `_miner_worker` so the measurement reflects the actual
    per-attempt cost (digest + int conversion + nonce increment).
    """
    sha256 = hashlib.sha256
    prev_hash = hashlib.sha256(b"BENCHMARK").digest()
    miner_id_bytes = (0).to_bytes(4, "big")

    count = 0
    deadline = time.time() + seconds
    nonce = 0
    while time.time() < deadline:
        # Batch inside a tight inner loop so we don't call time.time() per hash.
        for _ in range(10000):
            nonce_bytes = nonce.to_bytes(12, "big")
            h = sha256(prev_hash + miner_id_bytes + nonce_bytes).digest()
            _ = int.from_bytes(h, "big")
            nonce += 1
        count += 10000
    elapsed = time.time() - (deadline - seconds)
    return count / elapsed
