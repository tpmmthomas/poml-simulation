"""Miner worker process for PoML simulation.

Each miner runs Algorithm 1 from the paper:
1. Retrieve queries from mempool
2. For each query: derive seed, run inference+proof, evaluate lottery
3. If lottery won → assemble block and submit to coordinator
4. While a proof is running, watch the network inbox — if a new block
   lands, terminate the in-flight proof subprocess and restart mining on
   the new tip immediately (no waiting for the current ~60-90s proof
   to finish).
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
from poml_sim.vrf import vrf_eval, vrf_noise_schedule

logger = logging.getLogger(__name__)


# How often the miner checks its inbox while waiting on a proof worker.
# 100ms is low enough that abort latency is dominated by subprocess teardown
# (not polling) while still being cheap on CPU.
_INBOX_POLL_INTERVAL_S = 0.1


def _prove_worker(
    conditioning: list[float],
    noise: list[float],
    artifacts_dir: str,
    input_shape: list[int],
    result_q: "multiprocessing.Queue[Any]",
) -> None:
    """Run `run_inference_and_prove` inside an isolated subprocess.

    Placed at module scope (rather than as a method) so it can be targeted by
    multiprocessing.Process under any start method — fork inherits the module
    reference directly, spawn re-imports it by qualified name.

    Results are shipped back via `result_q` as a status tuple:
        ("ok", output_values, proof_bytes) on success
        ("error", repr(exception))         on failure
    """
    try:
        # Lazy import — ezkl is heavy and should only be paid for here.
        from poml_sim.zkp import run_inference_and_prove

        output_values, proof_bytes = run_inference_and_prove(
            conditioning=conditioning,
            noise=noise,
            artifacts_dir=artifacts_dir,
            input_shape=input_shape,
        )
        result_q.put(("ok", output_values, proof_bytes))
    except Exception as e:
        # Surface to the parent; the parent treats this like a failed proof.
        result_q.put(("error", repr(e)))


def _terminate_proc(proc: multiprocessing.Process) -> None:
    """Best-effort terminate-then-kill for a proof-worker subprocess.

    SIGTERM first so the worker gets a chance to clean up (ezkl opens tmpdirs
    and mmaps SRS files). If it's still alive after 5s we escalate to SIGKILL
    — we'd rather leak a tmpdir than block the main mining loop.
    """
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=5.0)
    if proc.is_alive():
        logger.warning(
            "proof worker pid=%s did not terminate on SIGTERM; killing",
            proc.pid,
        )
        proc.kill()
        proc.join(timeout=2.0)


class MinerProcess(multiprocessing.Process):
    """Single miner process running the PoML block production loop."""

    def __init__(
        self,
        miner_id: int,
        miner_pk: bytes,
        miner_sk: bytes,
        miner_vrf_vk: bytes,
        miner_vrf_sk: bytes,
        miner_enc_vrf_vk: bytes,
        miner_enc_vrf_sk: bytes,
        mempool: Any,  # Mempool — can't type-hint due to pickling
        inbox: multiprocessing.Queue,
        coordinator_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
        config: dict,
    ) -> None:
        # Non-daemon so the miner can spawn its own grandchild proof-worker
        # subprocess (Python forbids daemon processes from having children).
        # Cleanup is handled explicitly in Coordinator._shutdown() via
        # stop_event → STOP messages → join(timeout) → terminate fallback.
        super().__init__(daemon=False)
        self.miner_id = miner_id
        self.miner_pk = miner_pk
        self.miner_sk = miner_sk
        # Inference VRF keypair: derives per-step noise schedule U_i.
        self.miner_vrf_vk = miner_vrf_vk
        self.miner_vrf_sk = miner_vrf_sk
        # Encryption VRF keypair: derives per-query encryption randomness r_enc.
        self.miner_enc_vrf_vk = miner_enc_vrf_vk
        self.miner_enc_vrf_sk = miner_enc_vrf_sk
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
        # Local imports so the strategy enum and per-miner RNG stay private.
        import random as _random

        from poml_sim.mempool import FetchStrategy

        # Warm up the zkp module once in the miner process so every forked
        # proof worker inherits the already-loaded module (ezkl import is
        # expensive; paying it here rather than per-proof saves ~1-2s/fork).
        import poml_sim.zkp  # noqa: F401

        artifacts_dir = self.config["ezkl_artifacts_dir"]
        difficulty = self.config["difficulty_int"]
        max_queries = self.config["max_queries_per_block"]
        input_shape = self.config["input_shape"]
        spatial = input_shape[2] * input_shape[3]
        T = self.config["diffusion_steps"]
        strategy = FetchStrategy.parse(self.config.get("fetch_strategy", "sequential"))
        # Per-miner RNG so the RANDOM strategy is reproducible per miner. Mix
        # the miner_id into the seed so different miners see different orders.
        rng = _random.Random(0xC0FFEE ^ self.miner_id)

        logger.info(
            "Miner %d started (pk=%s, vrf_vk=%s, strategy=%s)",
            self.miner_id,
            self.miner_pk.hex()[:12],
            self.miner_vrf_vk.hex()[:12],
            strategy.value,
        )

        while not self.stop_event.is_set():
            # Check inbox for new blocks first
            if self._process_inbox():
                continue  # Chain advanced, restart mining

            # Fetch queries from mempool — peek-only, so multiple miners may
            # legitimately come back with overlapping batches. The blockchain
            # rejects cross-block duplicates and the mempool drops queries on
            # block confirmation, so consistency is enforced at confirm time.
            queries = self.mempool.get_queries(strategy, max_queries, rng=rng)
            if not queries:
                # No queries available, wait briefly
                time.sleep(0.1)
                continue

            logger.debug(
                "Miner %d fetched %d queries (%s), mining at height %d",
                self.miner_id,
                len(queries),
                strategy.value,
                self.local_height + 1,
            )

            # Build inference chain per Algorithm 1
            fingerprint = block_fingerprint(self.local_tip_hash, [])
            results: list[InferenceResult] = []
            ciphertexts_so_far: list[bytes] = []

            for i, query in enumerate(queries):
                # Check inbox between queries — abandon if chain advanced.
                # No mempool bookkeeping needed: we never removed these queries
                # from the mempool when we fetched them (peek semantics).
                if self._process_inbox():
                    break

                if self.stop_event.is_set():
                    break

                position = i + 1

                # Step 1: Proof-based chain binding per revised paper §4.4.
                # bind_1 = G(s,tx), bind_i = H(pi_{i-1}) for i >= 2.
                if position == 1:
                    chain_binding = fingerprint
                else:
                    chain_binding = sha256(results[-1].proof_bytes)

                # Step 2: Derive seed r_i = H(bind_i || h_{u,i} || taskID_i || vk^sig_m).
                seed = derive_seed(
                    chain_binding,
                    query.commitment,
                    query.task_id,
                    self.miner_pk,
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

                # Step 4: Run inference + generate ZKP via EZKL.
                #
                # Delegated to a child subprocess that we can kill the instant
                # a new block lands on the network — otherwise we'd burn 60+
                # seconds finishing a proof that's already stale, since ezkl
                # is a blocking native call with no Python-side cancel hook.
                logger.info(
                    "Miner %d: starting proof %d/%d (query=%s)",
                    self.miner_id,
                    position,
                    len(queries),
                    query.query_id,
                )
                prove_start = time.time()
                proof_result = self._prove_interruptible(
                    conditioning=conditioning,
                    noise=noise,
                    artifacts_dir=artifacts_dir,
                    input_shape=input_shape,
                )
                prove_time = time.time() - prove_start

                if proof_result is None:
                    # Either a new block arrived mid-proof (the helper has
                    # already advanced our local tip via _process_inbox) or
                    # stop was signalled or the worker crashed. In all cases:
                    # abandon this block attempt and let the outer loop pick
                    # up a fresh batch at the new tip.
                    logger.info(
                        "Miner %d: proof %d/%d aborted after %.1fs — restarting at height %d",
                        self.miner_id,
                        position,
                        len(queries),
                        prove_time,
                        self.local_height + 1,
                    )
                    break

                output_values, proof_bytes = proof_result

                # Step 5a: Derive encryption randomness via the encryption VRF.
                # r_enc, pi_enc = VRF.Eval(sk_VRF_enc, seed || taskID)
                # This gives each ciphertext unique, verifiable randomness.
                import struct as _struct

                r_enc, pi_enc = vrf_eval(
                    self.miner_enc_vrf_sk,
                    seed + _struct.pack(">I", query.task_id),
                )

                # Step 5b: Encrypt output with VRF-derived randomness.
                # ct_i = Enc(pk_u, y_i || taskID; r_enc). The r_enc is
                # embedded in the ciphertext plaintext so the decryptor can
                # verify the VRF relationship. ZKP does not attest to this.
                from poml_sim.encryption import encrypt_output

                ciphertext = encrypt_output(
                    recipient_pk_bytes=query.encryption_pk,
                    output_values=output_values,
                    enc_randomness=r_enc,
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
                    enc_vrf_randomness=r_enc,
                    enc_vrf_proof=pi_enc,
                )
                results.append(result)
                ciphertexts_so_far.append(ciphertext)

                # Step 6: Lottery over ciphertexts.
                # H_i = H(G(s,x), (ct_1, ..., ct_i)) < D.
                lottery_hash_int, lottery_won = evaluate_lottery(
                    fingerprint, ciphertexts_so_far, difficulty
                )

                logger.info(
                    "Miner %d: proof %d/%d done (%.1fs), lottery hash=%s -> %s",
                    self.miner_id,
                    position,
                    len(queries),
                    prove_time,
                    hex(lottery_hash_int)[:14],
                    "WIN" if lottery_won else "miss",
                )

                if lottery_won:
                    # The proof took ~80-90s. Drain the inbox before assembling
                    # so we don't submit a block on a stale tip — the
                    # coordinator would just reject it on prev_hash mismatch.
                    if self._process_inbox():
                        logger.info(
                            "Miner %d: won lottery but chain advanced during proof — discarding",
                            self.miner_id,
                        )
                        break

                    # Assemble block
                    block = self._assemble_block(
                        queries[:position],
                        results,
                        lottery_hash_int.to_bytes(32, "big"),
                    )

                    # Submit to coordinator. No mempool action needed: the
                    # coordinator removes the included queries from the mempool
                    # on confirm. Queries beyond `position` simply remain in the
                    # mempool for someone else to pick up.
                    self.coordinator_queue.put(
                        {
                            "type": "new_block",
                            "miner_id": self.miner_id,
                            "block": block,
                            "prove_time_total": prove_time,
                        }
                    )
                    break

            # No mempool fallback under peek semantics — the mempool was never
            # mutated by this miner, so there's nothing to release.

        logger.info("Miner %d stopped", self.miner_id)

    def _prove_interruptible(
        self,
        conditioning: list[float],
        noise: list[float],
        artifacts_dir: str,
        input_shape: list[int],
    ) -> tuple[list[float], bytes] | None:
        """Run a proof in a child subprocess, aborting it if the chain advances.

        Polls `self.inbox` at ``_INBOX_POLL_INTERVAL_S`` while the worker runs.
        The instant a NEW_BLOCK lands that advances our local tip — or a STOP
        message arrives, or `self.stop_event` fires — the worker is terminated
        and this method returns ``None`` so the caller can restart mining at
        the new height immediately, without waiting for the (now-stale) proof
        to finish.

        Returns:
            (output_values, proof_bytes) on success.
            None on abort or worker failure; in the abort case, ``_process_inbox``
            has already updated ``self.local_tip_hash`` / ``self.local_height``.
        """
        # Use the same start method that the miner itself was started under so
        # the grandchild inherits ezkl state on Linux (fork) and cleanly
        # re-imports it on macOS / Windows (spawn). Passing no name keeps the
        # default context.
        ctx = multiprocessing.get_context()
        result_q: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_prove_worker,
            args=(conditioning, noise, artifacts_dir, input_shape, result_q),
            daemon=True,
        )
        proc.start()

        try:
            while True:
                # Interrupt check: a newly confirmed block on the network, or
                # a shutdown, means this in-flight proof is wasted work.
                if self._process_inbox() or self.stop_event.is_set():
                    logger.info(
                        "Miner %d: aborting in-flight proof (pid=%s) — new block or stop received",
                        self.miner_id,
                        proc.pid,
                    )
                    _terminate_proc(proc)
                    return None

                try:
                    result = result_q.get(timeout=_INBOX_POLL_INTERVAL_S)
                except Empty:
                    if not proc.is_alive():
                        # Worker exited without posting a result (crash, OOM
                        # kill, etc.). Treat like a failed proof so the outer
                        # loop moves on instead of waiting forever.
                        logger.error(
                            "Miner %d: proof worker died without producing a result",
                            self.miner_id,
                        )
                        return None
                    continue

                # Worker finished — reap it and interpret the status tuple.
                proc.join(timeout=5.0)
                status = result[0] if result else None
                if status == "error":
                    logger.error(
                        "Miner %d proof worker failed: %s",
                        self.miner_id,
                        result[1],
                    )
                    return None
                if status == "ok":
                    return result[1], result[2]
                # Unknown status — be defensive: log and drop.
                logger.error(
                    "Miner %d: unexpected proof worker status %r",
                    self.miner_id,
                    status,
                )
                return None
        finally:
            # Belt-and-suspenders: ensure the child is reaped on any exit
            # path (exceptions, early returns) and the queue's feeder thread
            # doesn't block process shutdown.
            _terminate_proc(proc)
            try:
                result_q.close()
                result_q.join_thread()
            except Exception:
                pass

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
            miner_enc_vrf_vk=self.miner_enc_vrf_vk,
        )
        return Block(
            header=header,
            queries=queries,
            results=results,
            transactions=[],
        )
