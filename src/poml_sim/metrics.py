"""Metrics collection and output for PoML simulation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BlockMetric:
    height: int
    miner_id: int
    miner_pk: str
    timestamp: float
    num_queries: int
    num_transactions: int
    lottery_hash: str
    time_since_last_block: float
    lottery_attempts: int = 0


@dataclass
class SimulationMetrics:
    """Collects per-block and aggregate simulation metrics."""

    blocks: list[BlockMetric] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_proofs_generated: int = 0

    def record_block(
        self,
        height: int,
        miner_id: int,
        miner_pk: bytes,
        timestamp: float,
        num_queries: int,
        num_transactions: int,
        lottery_hash: bytes,
        lottery_attempts: int = 0,
    ) -> None:
        time_since_last = 0.0
        if self.blocks:
            time_since_last = timestamp - self.blocks[-1].timestamp

        self.blocks.append(
            BlockMetric(
                height=height,
                miner_id=miner_id,
                miner_pk=miner_pk.hex()[:24],
                timestamp=timestamp,
                num_queries=num_queries,
                num_transactions=num_transactions,
                lottery_hash=lottery_hash.hex()[:24],
                time_since_last_block=time_since_last,
                lottery_attempts=lottery_attempts,
            )
        )
        self.total_proofs_generated += num_queries

    def aggregate(self) -> dict:
        if not self.blocks:
            return {"total_blocks": 0}

        block_times = [b.time_since_last_block for b in self.blocks[1:]]
        miner_wins: dict[int, int] = {}
        for b in self.blocks:
            miner_wins[b.miner_id] = miner_wins.get(b.miner_id, 0) + 1

        queries_per_block = [b.num_queries for b in self.blocks]

        return {
            "total_blocks": len(self.blocks),
            "total_proofs_generated": self.total_proofs_generated,
            "avg_block_time": sum(block_times) / len(block_times) if block_times else 0.0,
            "min_block_time": min(block_times) if block_times else 0.0,
            "max_block_time": max(block_times) if block_times else 0.0,
            "avg_queries_per_block": sum(queries_per_block) / len(queries_per_block),
            "miner_win_distribution": miner_wins,
            "simulation_duration": self.end_time - self.start_time,
        }

    def dump(self, filepath: str | Path = "metrics.json") -> None:
        data = {
            "blocks": [
                {
                    "height": b.height,
                    "miner_id": b.miner_id,
                    "miner_pk": b.miner_pk,
                    "timestamp": b.timestamp,
                    "num_queries": b.num_queries,
                    "num_transactions": b.num_transactions,
                    "lottery_hash": b.lottery_hash,
                    "time_since_last_block": round(b.time_since_last_block, 4),
                    "lottery_attempts": b.lottery_attempts,
                }
                for b in self.blocks
            ],
            "aggregate": self.aggregate(),
        }
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Metrics written to %s", filepath)
