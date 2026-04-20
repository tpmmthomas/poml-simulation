"""Coordinator: orchestrates the PoML simulation.

Responsibilities:
1. Load config
2. Initialize mempool, network bus, blockchain, account state
3. Spawn N miner processes
4. Start query generator
5. Main loop: receive blocks from miners, validate, broadcast, apply rewards
6. Termination and metrics dump
"""

from __future__ import annotations

import logging
import multiprocessing
import time

from poml_sim.accounts import AccountState
from poml_sim.blockchain import Blockchain
from poml_sim.config import SimConfig, load_config
from poml_sim.crypto import generate_keypair, hash_block
from poml_sim.mempool import Mempool
from poml_sim.metrics import SimulationMetrics
from poml_sim.miner import MinerProcess
from poml_sim.network import MessageType, NetworkBus, NetworkMessage
from poml_sim.query_generator import QueryGenerator
from poml_sim.types import Block

logger = logging.getLogger(__name__)


class Coordinator:
    """Main orchestration process for the PoML simulation."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or load_config()
        self.blockchain = Blockchain(self.config.difficulty_int)
        self.account_state = AccountState()
        self.mempool = Mempool()
        self.network = NetworkBus(self.config.num_miners, self.config.network_latency_ms)
        self.metrics = SimulationMetrics()

        # Coordinator receives blocks from miners via this queue
        self.coordinator_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

        # Generate miner keypairs
        self.miner_keys: list[tuple[bytes, bytes]] = []
        for _ in range(self.config.num_miners):
            pk, sk = generate_keypair()
            self.miner_keys.append((pk, sk))
            # Give each miner an initial balance for the simulation
            self.account_state.credit(pk, 100)

        self.miners: list[MinerProcess] = []
        self.query_generator: QueryGenerator | None = None

    def run(self, timeout: float = 300.0) -> None:
        """Run the full simulation."""
        logger.info("=" * 60)
        logger.info("PoML Simulation Starting")
        logger.info(
            "  Miners: %d | Queries: %d | Difficulty: %s",
            self.config.num_miners,
            self.config.num_queries,
            self.config.difficulty[:20] + "...",
        )
        logger.info("=" * 60)

        self.metrics.start_time = time.time()

        # Spawn miners
        config_dict = {
            "ezkl_artifacts_dir": self.config.ezkl_artifacts_dir,
            "difficulty_int": self.config.difficulty_int,
            "max_queries_per_block": self.config.max_queries_per_block,
            "input_shape": self.config.input_shape,
        }
        for i in range(self.config.num_miners):
            pk, sk = self.miner_keys[i]
            miner = MinerProcess(
                miner_id=i,
                miner_pk=pk,
                miner_sk=sk,
                mempool=self.mempool,
                inbox=self.network.inboxes[i],
                coordinator_queue=self.coordinator_queue,
                stop_event=self.stop_event,
                config=config_dict,
            )
            self.miners.append(miner)

        # Start query generator
        spatial = self.config.input_shape[2] * self.config.input_shape[3]
        self.query_generator = QueryGenerator(
            mempool=self.mempool,
            num_queries=self.config.num_queries,
            query_rate=self.config.query_rate,
            spatial_size=spatial,
            seed=self.config.seed,
        )
        self.query_generator.start()

        # Start miners
        for miner in self.miners:
            miner.start()

        # Main coordinator loop
        deadline = time.time() + timeout
        blocks_produced = 0

        try:
            while time.time() < deadline and not self.stop_event.is_set():
                try:
                    msg = self.coordinator_queue.get(timeout=1.0)
                except Exception:
                    continue

                if msg["type"] == "new_block":
                    block: Block = msg["block"]
                    miner_id: int = msg["miner_id"]

                    valid, reason = self.blockchain.validate_block(block, self.account_state)
                    if valid:
                        self.blockchain.add_block(block, self.account_state)

                        # Apply rewards
                        self.account_state.apply_block_reward(
                            block.header.miner_pk,
                            self.config.block_reward,
                        )
                        # Apply query fees
                        total_fees = sum(q.fee for q in block.queries)
                        self.account_state.apply_fees(block.header.miner_pk, total_fees)

                        # Record metrics
                        self.metrics.record_block(
                            height=block.header.block_height,
                            miner_id=miner_id,
                            miner_pk=block.header.miner_pk,
                            timestamp=block.header.timestamp,
                            num_queries=len(block.queries),
                            num_transactions=len(block.transactions),
                            lottery_hash=block.header.lottery_hash,
                            lottery_attempts=len(block.results),
                        )

                        # Broadcast to all miners
                        net_msg = NetworkMessage(
                            msg_type=MessageType.NEW_BLOCK,
                            sender_id=miner_id,
                            payload=block,
                            timestamp=time.time(),
                        )
                        self.network.broadcast(net_msg, sender_id=miner_id)
                        # Also update the winning miner directly (no delay)
                        self.network.send_to(miner_id, net_msg)

                        blocks_produced += 1
                        logger.info(
                            ">>> Block %d confirmed (miner %d, %d queries, chain height %d)",
                            block.header.block_height,
                            miner_id,
                            len(block.queries),
                            self.blockchain.get_height(),
                        )
                    else:
                        logger.warning(
                            "Block from miner %d rejected: %s",
                            miner_id,
                            reason,
                        )

                # Check termination: all queries processed and a reasonable number of blocks
                if blocks_produced > 0 and self.mempool.size == 0:
                    # Wait a bit for any in-flight blocks
                    time.sleep(2.0)
                    if self.mempool.size == 0:
                        logger.info("All queries processed, shutting down")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._shutdown()

        self.metrics.end_time = time.time()
        self._report()

    def _shutdown(self) -> None:
        """Gracefully stop all miners."""
        logger.info("Shutting down miners...")
        self.stop_event.set()
        self.network.stop_all()

        if self.query_generator:
            self.query_generator.stop()

        for miner in self.miners:
            miner.join(timeout=10.0)
            if miner.is_alive():
                logger.warning("Miner %d did not stop gracefully, terminating", miner.miner_id)
                miner.terminate()

    def _report(self) -> None:
        """Print summary and dump metrics."""
        agg = self.metrics.aggregate()

        logger.info("=" * 60)
        logger.info("Simulation Complete")
        logger.info("  Total blocks: %d", agg.get("total_blocks", 0))
        logger.info("  Total proofs: %d", agg.get("total_proofs_generated", 0))
        logger.info("  Avg block time: %.2fs", agg.get("avg_block_time", 0))
        logger.info("  Avg queries/block: %.1f", agg.get("avg_queries_per_block", 0))
        logger.info("  Duration: %.1fs", agg.get("simulation_duration", 0))
        logger.info("  Miner wins: %s", agg.get("miner_win_distribution", {}))
        logger.info("=" * 60)

        # Account balances
        logger.info("Final balances:")
        for i, (pk, _) in enumerate(self.miner_keys):
            bal = self.account_state.get_balance(pk)
            logger.info("  Miner %d: %d coins", i, bal)

        self.metrics.dump("metrics.json")
