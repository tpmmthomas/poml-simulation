"""Process-safe mempool for PoML simulation.

Bitcoin-style semantics: queries are NOT removed when a miner fetches them.
Multiple miners may concurrently work on overlapping query sets. A query is
removed from the mempool only when the coordinator confirms a block that
includes it (via `remove_queries`).

Backed by a multiprocessing.Manager so the underlying list is shared across
miner subprocesses. All mutating operations serialize through a manager Lock
so concurrent add + remove cannot lose entries.
"""

from __future__ import annotations

import logging
import random
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager

    from poml_sim.types import Query

logger = logging.getLogger(__name__)


class FetchStrategy(str, Enum):
    """How a miner orders the mempool when picking queries to mine."""

    SEQUENTIAL = "sequential"  # FIFO: oldest queries first (insertion order)
    HIGH_FEE = "high_fee"      # highest `fee` first
    RANDOM = "random"          # uniformly shuffled

    @classmethod
    def parse(cls, name: str) -> "FetchStrategy":
        try:
            return cls(name.lower())
        except ValueError as e:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(
                f"unknown fetch strategy {name!r}; expected one of: {valid}"
            ) from e


class Mempool:
    """Manager-backed peek-style mempool shared across miner processes."""

    def __init__(self, queries_proxy, lock) -> None:
        # `queries_proxy` is a Manager().list(); `lock` is a Manager().Lock().
        # Construct via `Mempool.create(manager)` — direct __init__ is for
        # subprocess inheritance, not for callers building a fresh mempool.
        self._queries = queries_proxy
        self._lock = lock

    @classmethod
    def create(cls, manager: "SyncManager") -> "Mempool":
        return cls(manager.list(), manager.Lock())

    def add_query(self, query: "Query") -> None:
        # Lock so a concurrent remove_queries can't snapshot/filter/replace
        # right around our append and lose it.
        with self._lock:
            self._queries.append(query)

    def remove_queries(self, query_ids) -> int:
        """Drop queries by id. Returns the count actually removed.

        Called by the coordinator when a block is confirmed — those queries
        are now in the chain and miners should stop working on them.
        """
        ids = set(query_ids)
        if not ids:
            return 0
        with self._lock:
            snapshot = list(self._queries)
            # Build the indices to remove from the live proxy. Iterate in
            # reverse so deletions don't shift earlier indices.
            doomed = [i for i, q in enumerate(snapshot) if q.query_id in ids]
            for i in reversed(doomed):
                del self._queries[i]
            return len(doomed)

    def get_queries(
        self,
        strategy: FetchStrategy,
        max_n: int,
        rng: random.Random | None = None,
    ) -> list["Query"]:
        """Peek (do NOT remove) up to `max_n` queries ordered by `strategy`."""
        with self._lock:
            snapshot = list(self._queries)
        if not snapshot:
            return []

        if strategy == FetchStrategy.SEQUENTIAL:
            ordered = snapshot
        elif strategy == FetchStrategy.HIGH_FEE:
            # Stable sort by fee desc; ties keep arrival order.
            ordered = sorted(snapshot, key=lambda q: q.fee, reverse=True)
        elif strategy == FetchStrategy.RANDOM:
            ordered = snapshot.copy()
            (rng or random).shuffle(ordered)
        else:
            raise ValueError(f"Unknown fetch strategy: {strategy}")

        return ordered[:max_n]

    @property
    def size(self) -> int:
        # Single proxy call; safe to read without the lock. Worst case we see
        # an instantaneously-stale value, which is fine for the early-
        # termination check that's the only consumer.
        return len(self._queries)
