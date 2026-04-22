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
from poml_sim.vrf import generate_vrf_keypair

logger = logging.getLogger(__name__)


class Coordinator:
    """Main orchestration process for the PoML simulation."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or load_config()
        self.blockchain = Blockchain(
            self.config.difficulty_int,
            diffusion_steps=self.config.diffusion_steps,
        )
        self.account_state = AccountState()
        # Manager owns the shared list backing the Mempool. Held on self so it
        # outlives method scope and gets shut down explicitly in _shutdown().
        self._manager = multiprocessing.Manager()
        self.mempool = Mempool.create(self._manager)
        self.network = NetworkBus(self.config.num_miners, self.config.network_latency_ms)
        self.metrics = SimulationMetrics()

        # Coordinator receives blocks from miners via this queue
        self.coordinator_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

        # Generate miner keypairs: identity + dedicated VRF key per miner
        # (paper §PoML Protocol: each miner holds (pk_m, sk_m) and
        # independent (vk^VRF_m, sk^VRF_m)).
        self.miner_keys: list[tuple[bytes, bytes]] = []
        self.miner_vrf_keys: list[tuple[bytes, bytes]] = []
        for _ in range(self.config.num_miners):
            pk, sk = generate_keypair()
            self.miner_keys.append((pk, sk))
            vrf_vk, vrf_sk = generate_vrf_keypair()
            self.miner_vrf_keys.append((vrf_vk, vrf_sk))
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

        # Resolve per-miner fetch strategies. Cycle the configured list to
        # cover all miners — empty list falls back to SEQUENTIAL for everyone.
        from poml_sim.mempool import FetchStrategy

        strategy_names = self.config.fetch_strategies or ["sequential"]
        per_miner_strategies = [
            FetchStrategy.parse(strategy_names[i % len(strategy_names)]).value
            for i in range(self.config.num_miners)
        ]
        logger.info("Miner fetch strategies: %s", per_miner_strategies)

        # Spawn miners
        config_dict = {
            "ezkl_artifacts_dir": self.config.ezkl_artifacts_dir,
            "difficulty_int": self.config.difficulty_int,
            "max_queries_per_block": self.config.max_queries_per_block,
            "input_shape": self.config.input_shape,
            "diffusion_steps": self.config.diffusion_steps,
        }
        for i in range(self.config.num_miners):
            pk, sk = self.miner_keys[i]
            vrf_vk, vrf_sk = self.miner_vrf_keys[i]
            miner_config = {**config_dict, "fetch_strategy": per_miner_strategies[i]}
            miner = MinerProcess(
                miner_id=i,
                miner_pk=pk,
                miner_sk=sk,
                miner_vrf_vk=vrf_vk,
                miner_vrf_sk=vrf_sk,
                mempool=self.mempool,
                inbox=self.network.inboxes[i],
                coordinator_queue=self.coordinator_queue,
                stop_event=self.stop_event,
                config=miner_config,
            )
            self.miners.append(miner)

        # Start query generator
        spatial = self.config.input_shape[2] * self.config.input_shape[3]
        self.query_generator = QueryGenerator(
            mempool=self.mempool,
            num_queries=self.config.num_queries,
            initial_burst=self.config.initial_burst,
            steady_interval_s=self.config.steady_interval_s,
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

                        # Drop confirmed queries from the mempool BEFORE the
                        # broadcast so any miner who reacts to the new-block
                        # message sees the updated mempool on its next fetch.
                        removed = self.mempool.remove_queries(
                            [q.query_id for q in block.queries]
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
                            ">>> Block %d confirmed (miner %d, %d queries, chain height %d, %d removed from mempool)",
                            block.header.block_height,
                            miner_id,
                            len(block.queries),
                            self.blockchain.get_height(),
                            removed,
                        )
                    else:
                        # Mempool fetch is peek-only, so a rejected block does
                        # NOT need its queries re-emitted — they were never
                        # removed in the first place. Other miners (or the
                        # same one on its next round) will pick them up
                        # naturally from the still-populated mempool.
                        logger.warning(
                            "Block from miner %d rejected: %s",
                            miner_id,
                            reason,
                        )

                # Early termination: only stop once the generator has produced
                # its full quota AND the mempool has drained. Without the
                # is_done check, the drip-paced generator would leave the
                # mempool transiently empty between queries and trigger
                # shutdown after the first block.
                generator_done = (
                    self.query_generator is not None and self.query_generator.is_done
                )
                if blocks_produced > 0 and generator_done and self.mempool.size == 0:
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

        # Tear down the Manager process backing the shared mempool.
        try:
            self._manager.shutdown()
        except Exception:
            pass

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
