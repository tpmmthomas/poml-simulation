"""Query generator for PoML simulation.

Spawns queries at a configured rate and pushes them to the mempool.
"""

from __future__ import annotations

import logging
import os
import random
import struct
import threading
import time
from typing import TYPE_CHECKING

from poml_sim.crypto import generate_keypair, sha256, sign
from poml_sim.encryption import generate_encryption_keypair
from poml_sim.types import Query

if TYPE_CHECKING:
    from poml_sim.mempool import Mempool

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Generates inference queries and pushes them to the mempool."""

    def __init__(
        self,
        mempool: Mempool,
        num_queries: int,
        query_rate: float,
        spatial_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.mempool = mempool
        self.num_queries = num_queries
        self.query_rate = query_rate
        self.spatial_size = spatial_size
        self.rng = random.Random(seed)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Create a few simulated users
        # (signing_pk, signing_sk, task_counter, encryption_pk, encryption_sk)
        self._users: list[tuple[bytes, bytes, int, bytes, bytes]] = []
        for _ in range(5):
            pk, sk = generate_keypair()
            enc_pk, enc_sk = generate_encryption_keypair()
            self._users.append((pk, sk, 0, enc_pk, enc_sk))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        interval = 1.0 / self.query_rate if self.query_rate > 0 else 0.5
        generated = 0

        for query_id in range(self.num_queries):
            if self._stop.is_set():
                break

            query = self._make_query(query_id)
            self.mempool.add_query(query)
            generated += 1
            logger.info("Query %d generated (user=%s)", query_id, query.user_pk.hex()[:12])

            if generated < self.num_queries:
                time.sleep(interval)

        logger.info("Query generator finished: %d queries produced", generated)

    def _make_query(self, query_id: int) -> Query:
        # Pick a random user
        idx = self.rng.randint(0, len(self._users) - 1)
        pk, sk, task_counter, enc_pk, enc_sk = self._users[idx]
        task_id = task_counter
        self._users[idx] = (pk, sk, task_counter + 1, enc_pk, enc_sk)

        # Random conditioning input
        conditioning = [self.rng.gauss(0, 1) for _ in range(self.spatial_size)]

        # Commitment = SHA256(conditioning || taskID)
        conditioning_bytes = struct.pack(f">{len(conditioning)}f", *conditioning)
        commitment_randomness = os.urandom(16)
        commitment = sha256(
            conditioning_bytes,
            struct.pack(">I", task_id),
            commitment_randomness,
        )

        # Signature
        fee = self.rng.randint(1, 10)
        sig_data = struct.pack(">I", task_id) + commitment + struct.pack(">I", fee)
        signature = sign(sk, sig_data)

        return Query(
            query_id=query_id,
            user_pk=pk,
            task_id=task_id,
            conditioning_input=conditioning,
            commitment=commitment,
            commitment_randomness=commitment_randomness,
            fee=fee,
            signature=signature,
            encryption_pk=enc_pk,
        )
