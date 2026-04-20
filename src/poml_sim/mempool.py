"""Process-safe mempool for PoML simulation.

Wraps multiprocessing.Queue for sharing queries across miner processes.
"""

from __future__ import annotations

import logging
import multiprocessing
from queue import Empty
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poml_sim.types import Query

logger = logging.getLogger(__name__)


class Mempool:
    """Thread/process-safe mempool backed by multiprocessing.Queue."""

    def __init__(self) -> None:
        self._queue: multiprocessing.Queue[Query] = multiprocessing.Queue()
        self._count = multiprocessing.Value("i", 0)

    def add_query(self, query: Query) -> None:
        self._queue.put(query)
        with self._count.get_lock():
            self._count.value += 1

    def get_queries(self, max_n: int) -> list[Query]:
        """Pop up to max_n queries (non-blocking)."""
        queries: list[Query] = []
        for _ in range(max_n):
            try:
                q = self._queue.get_nowait()
                queries.append(q)
            except Empty:
                break
        return queries

    def return_queries(self, queries: list[Query]) -> None:
        """Return un-mined queries back to the mempool."""
        for q in queries:
            self._queue.put(q)

    @property
    def size(self) -> int:
        return self._queue.qsize()
