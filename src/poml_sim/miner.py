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

from poml_sim.blockchain import create_genesis_block
from poml_sim.crypto import (
    block_fingerprint,
    derive_seed,
    evaluate_lottery,
    hash_block,
    sha256,
)
from poml_sim.network import MessageType, NetworkMessage
from poml_sim.types import Block, BlockHeader, InferenceResult, Query
from poml_sim.vrf import vrf_noise_schedule

logger = logging.getLogger(__name__)


class MinerProcess(multiprocessing.Process):
    """Single miner process running the PoML block production loop."""

    def __init__(
        self,
        miner_id: int,
        miner_pk: bytes,
        miner_sk: bytes,
        miner_vrf_vk: bytes,
        miner_vrf_sk: bytes,
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
        # Separate VRF keypair used to derive per-step inference noise per
        # the revised paper protocol (§PoML Protocol).
        self.miner_vrf_vk = miner_vrf_vk
        self.miner_vrf_sk = miner_vrf_sk
        self.mempool = mempool
        self.inbox = inbox
        self.coordinator_queue = coordinator_queue
        self.stop_event = stop_event
        self.config = config

        # Local chain state (just track tip hash and height). The initial
        # tip is the genesis block's hash — not all zeros — so the first
        # block we produce chains correctly against Blockchain.get_tip_hash().
        self.local_tip_hash: bytes = hash_block(create_genesis_block())
        self.local_height: int = 0

    def run(self) -> None:
        logger.info(
            "Miner %d started (pk=%s, vrf_vk=%s)",
            self.miner_id,
            self.miner_pk.hex()[:12],
            self.miner_vrf_vk.hex()[:12],
        )
        artifacts_dir = self.config["ezkl_artifacts_dir"]
        difficulty = self.config["difficulty_int"]
        max_queries = self.config["max_queries_per_block"]
        input_shape = self.config["input_shape"]
        spatial = input_shape[2] * input_shape[3]
        T = self.config["diffusion_steps"]

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

            # Build inference chain per Algorithm 1
            fingerprint = block_fingerprint(self.local_tip_hash, [])
            results: list[InferenceResult] = []
            ciphertexts_so_far: list[bytes] = []
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

                # Step 1: Chain binding per paper §4.4 (revised).
                # bind_1 = G(s,x), bind_i = H(ct_{i-1}) for i >= 2.
                if position == 1:
                    chain_binding = fingerprint
                else:
                    chain_binding = sha256(ciphertexts_so_far[-1])

                # Step 2: Derive seed r_i = H(G(s,x) || c_c || taskID || pk_m || i)
                seed = derive_seed(
                    fingerprint,
                    query.commitment,
                    query.task_id,
                    self.miner_pk,
                    position,
                )

                # Step 3: Derive noise schedule U_i and VRF transcript via the
                # miner's unbiasable VRF key. Only U_i[0] is fed to the
                # single-pass circuit; all T transcript entries go into the
                # block so validators can verify them.
                U_i, transcript = vrf_noise_schedule(
                    self.miner_vrf_sk, seed, T, spatial
                )
                noise = U_i[0]
                conditioning = query.conditioning_input[:spatial]
                while len(conditioning) < spatial:
                    conditioning.append(0.0)

                # Step 4: Run inference + generate ZKP via EZKL
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
                    logger.error(
                        "Miner %d proof generation failed: %s", self.miner_id, e
                    )
                    continue
                prove_time = time.time() - prove_start

                # Step 5: Deterministic encryption.
                # ct_i = Enc(pk_u, y_i || bind_i || taskID). Paper requires a
                # deterministic PKE so the lottery hash over ciphertexts is
                # unambiguous; we use textbook RSA (see encryption.py).
                from poml_sim.encryption import encrypt_output

                ciphertext = encrypt_output(
                    recipient_pk_bytes=query.encryption_pk,
                    output_values=output_values,
                    chain_binding=chain_binding,
                    task_id=query.task_id,
                )

                result = InferenceResult(
                    query_id=query.query_id,
                    output=output_values,
                    proof_bytes=proof_bytes,
                    seed_used=seed,
                    chain_binding=chain_binding,
                    ciphertext=ciphertext,
                    vrf_transcript=transcript,
                )
                results.append(result)
                ciphertexts_so_far.append(ciphertext)

                logger.debug(
                    "Miner %d: inference %d/%d done (%.2fs)",
                    self.miner_id,
                    position,
                    len(queries),
                    prove_time,
                )

                # Step 6: Lottery over ciphertexts.
                # H_i = H(G(s,x), (ct_1, ..., ct_i)) < D.
                lottery_hash_int, lottery_won = evaluate_lottery(
                    fingerprint, ciphertexts_so_far, difficulty
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
                        # Must match Blockchain.get_tip_hash() which uses
                        # hash_block() over the full block, not just a
                        # sha256 of (prev_hash, lottery_hash).
                        self.local_tip_hash = hash_block(block)
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
            miner_vrf_vk=self.miner_vrf_vk,
        )
        return Block(
            header=header,
            queries=queries,
            results=results,
            transactions=[],
        )
