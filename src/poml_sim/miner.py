"""Miner worker process for PoML simulation.

Each miner runs Algorithm 1 from the paper:
1. Retrieve queries from mempool
2. For each query: derive seed, run inference+proof, evaluate lottery
3. If lottery won → assemble block and submit to coordinator
4. Between queries, check for new blocks from the network
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from queue import Empty
from typing import Any

from poml_sim.crypto import (
    block_fingerprint,
    derive_seed,
    evaluate_lottery,
    generate_dummy_meta_proof,
    seed_to_noise,
    sha256,
)
from poml_sim.network import MessageType, NetworkMessage
from poml_sim.types import Block, BlockHeader, InferenceResult, Query

logger = logging.getLogger(__name__)


class MinerProcess(multiprocessing.Process):
    """Single miner process running the PoML block production loop."""

    def __init__(
        self,
        miner_id: int,
        miner_pk: bytes,
        miner_sk: bytes,
        mempool: Any,  # Mempool — can't type-hint due to pickling
        inbox: multiprocessing.Queue,
        coordinator_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
        config: dict,
    ) -> None:
        super().__init__(daemon=True)
        self.miner_id = miner_id
        self.miner_pk = miner_pk
        self.miner_sk = miner_sk
        self.mempool = mempool
        self.inbox = inbox
        self.coordinator_queue = coordinator_queue
        self.stop_event = stop_event
        self.config = config

        # Local chain state (just track tip hash and height)
        self.local_tip_hash: bytes = b"\x00" * 32
        self.local_height: int = 0

    def run(self) -> None:
        logger.info("Miner %d started (pk=%s)", self.miner_id, self.miner_pk.hex()[:12])
        artifacts_dir = self.config["ezkl_artifacts_dir"]
        difficulty = self.config["difficulty_int"]
        max_queries = self.config["max_queries_per_block"]
        input_shape = self.config["input_shape"]
        spatial = input_shape[2] * input_shape[3]

        while not self.stop_event.is_set():
            # Check inbox for new blocks first
            if self._process_inbox():
                continue  # Chain advanced, restart mining

            # Fetch queries from mempool
            queries = self.mempool.get_queries(max_queries)
            if not queries:
                # No queries available, wait briefly
                time.sleep(0.1)
                continue

            logger.debug(
                "Miner %d fetched %d queries, mining at height %d",
                self.miner_id,
                len(queries),
                self.local_height + 1,
            )

            # Build proof chain per Algorithm 1
            fingerprint = block_fingerprint(self.local_tip_hash, [])
            results: list[InferenceResult] = []
            proofs_so_far: list[bytes] = []
            won = False

            for i, query in enumerate(queries):
                # Check inbox between queries
                if self._process_inbox():
                    # Chain advanced — return unused queries to mempool
                    unused = queries[i:]
                    self.mempool.return_queries(unused)
                    won = False
                    break

                if self.stop_event.is_set():
                    self.mempool.return_queries(queries[i:])
                    return

                position = i + 1

                # Step 1: Compute chain binding per §4.4
                # bind_1 = G(s,x), bind_i = H(π_{i-1}) for i >= 2
                if position == 1:
                    chain_binding = fingerprint
                else:
                    chain_binding = sha256(proofs_so_far[-1])

                # Step 2: Derive seed r_i
                seed = derive_seed(
                    fingerprint,
                    query.commitment,
                    query.task_id,
                    self.miner_pk,
                    position,
                )

                # Step 3: Prepare input (noise from seed + conditioning)
                noise = seed_to_noise(seed, spatial)
                conditioning = query.conditioning_input[:spatial]
                # Pad conditioning if needed
                while len(conditioning) < spatial:
                    conditioning.append(0.0)

                # Step 4: Run inference + generate proof via EZKL
                prove_start = time.time()
                try:
                    from poml_sim.zkp import run_inference_and_prove

                    output_values, proof_bytes = run_inference_and_prove(
                        conditioning=conditioning,
                        noise=noise,
                        artifacts_dir=artifacts_dir,
                        input_shape=input_shape,
                    )
                except Exception as e:
                    logger.error("Miner %d proof generation failed: %s", self.miner_id, e)
                    continue
                prove_time = time.time() - prove_start

                # Step 5: Encrypt output: ct_i = Enc(pk_u, y_i || bind_i || taskID)
                encrypted_output = b""
                if query.encryption_pk:
                    try:
                        from poml_sim.encryption import encrypt_output

                        encrypted_output = encrypt_output(
                            recipient_pk_bytes=query.encryption_pk,
                            output_values=output_values,
                            chain_binding=chain_binding,
                            task_id=query.task_id,
                        )
                    except Exception as e:
                        logger.warning("Miner %d: encryption failed: %s", self.miner_id, e)

                # Step 6: Generate dummy meta-proof
                meta_proof = generate_dummy_meta_proof()

                result = InferenceResult(
                    query_id=query.query_id,
                    output=output_values,
                    proof_bytes=proof_bytes,
                    meta_proof_dummy=meta_proof,
                    seed_used=seed,
                    chain_binding=chain_binding,
                    encrypted_output=encrypted_output,
                )
                results.append(result)
                proofs_so_far.append(proof_bytes)

                logger.debug(
                    "Miner %d: proof %d/%d done (%.2fs)",
                    self.miner_id,
                    position,
                    len(queries),
                    prove_time,
                )

                # Step 5: Evaluate lottery
                lottery_hash_int, lottery_won = evaluate_lottery(
                    fingerprint, proofs_so_far, difficulty
                )

                if lottery_won:
                    logger.info(
                        "Miner %d WON lottery at proof %d (hash=%s)",
                        self.miner_id,
                        position,
                        hex(lottery_hash_int)[:20],
                    )
                    won = True

                    # Assemble block
                    block = self._assemble_block(
                        queries[:position],
                        results,
                        lottery_hash_int.to_bytes(32, "big"),
                    )

                    # Submit to coordinator
                    self.coordinator_queue.put(
                        {
                            "type": "new_block",
                            "miner_id": self.miner_id,
                            "block": block,
                            "prove_time_total": prove_time,
                        }
                    )

                    # Return unused queries
                    unused = queries[position:]
                    if unused:
                        self.mempool.return_queries(unused)
                    break

            if not won and results:
                # Didn't win — return all queries to mempool
                self.mempool.return_queries(queries)

        logger.info("Miner %d stopped", self.miner_id)

    def _process_inbox(self) -> bool:
        """Check for new blocks from the network. Returns True if chain advanced."""
        advanced = False
        try:
            while True:
                msg: NetworkMessage = self.inbox.get_nowait()
                if msg.msg_type == MessageType.NEW_BLOCK:
                    block: Block = msg.payload
                    if block.header.block_height > self.local_height:
                        self.local_tip_hash = sha256(
                            block.header.prev_hash,
                            block.header.lottery_hash,
                        )
                        self.local_height = block.header.block_height
                        advanced = True
                        logger.debug(
                            "Miner %d: chain advanced to height %d",
                            self.miner_id,
                            self.local_height,
                        )
                elif msg.msg_type == MessageType.STOP:
                    self.stop_event.set()
                    return True
        except Empty:
            pass
        return advanced

    def _assemble_block(
        self,
        queries: list[Query],
        results: list[InferenceResult],
        lottery_hash: bytes,
    ) -> Block:
        header = BlockHeader(
            block_height=self.local_height + 1,
            prev_hash=self.local_tip_hash,
            miner_pk=self.miner_pk,
            timestamp=time.time(),
            difficulty=self.config["difficulty_int"],
            lottery_hash=lottery_hash,
        )
        return Block(
            header=header,
            queries=queries,
            results=results,
            transactions=[],
        )
