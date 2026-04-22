"""Blockchain state management for PoML simulation.

Implements:
- Chain storage (longest chain)
- Block validation (lottery, proofs, transactions)
- Fork resolution (longest chain wins)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import struct

from poml_sim.crypto import block_fingerprint, evaluate_lottery, hash_block, sha256
from poml_sim.types import Block, BlockHeader
from poml_sim.vrf import vrf_verify

if TYPE_CHECKING:
    from poml_sim.accounts import AccountState
    from poml_sim.zkp import verify_proof

logger = logging.getLogger(__name__)


GENESIS_HASH = b"\x00" * 32


def create_genesis_block() -> Block:
    """Create the genesis block (block 0)."""
    header = BlockHeader(
        block_height=0,
        prev_hash=b"\x00" * 32,
        miner_pk=b"\x00" * 32,
        timestamp=0.0,
        difficulty=0,
        lottery_hash=b"\x00" * 32,
    )
    return Block(header=header)


class Blockchain:
    """Manages the canonical chain state."""

    def __init__(self, difficulty: int, diffusion_steps: int = 1) -> None:
        self.difficulty = difficulty
        self.diffusion_steps = diffusion_steps
        self.chain: list[Block] = [create_genesis_block()]
        self._block_hashes: dict[int, bytes] = {0: hash_block(self.chain[0])}
        # Set of query_ids already included somewhere in the chain. Mempool
        # is peek-style, so two miners may legitimately race and both submit
        # blocks containing the same query — the second one must be rejected.
        self._included_query_ids: set[int] = set()

    def get_tip(self) -> Block:
        return self.chain[-1]

    def get_tip_hash(self) -> bytes:
        return self._block_hashes[self.get_height()]

    def get_height(self) -> int:
        return len(self.chain) - 1

    def validate_block(
        self,
        block: Block,
        account_state: AccountState | None = None,
        verify_zkp: bool = False,
        artifacts_dir: str = "model/",
    ) -> tuple[bool, str]:
        """Validate a block against the current chain state.

        Returns (is_valid, reason).
        """
        header = block.header

        # 1. Check prev_hash matches tip
        tip_hash = self.get_tip_hash()
        if header.prev_hash != tip_hash:
            return False, f"prev_hash mismatch: expected {tip_hash.hex()[:16]}, got {header.prev_hash.hex()[:16]}"

        # 2. Check block height
        expected_height = self.get_height() + 1
        if header.block_height != expected_height:
            return False, f"height mismatch: expected {expected_height}, got {header.block_height}"

        # 3. Non-empty queries + results
        if len(block.queries) == 0 or len(block.results) == 0:
            return False, "block has no queries or results"

        if len(block.queries) != len(block.results):
            return False, f"query/result count mismatch: {len(block.queries)} vs {len(block.results)}"

        # 3b. Reject blocks containing queries already in the chain. Without
        # this check, two miners racing on overlapping batches could both win
        # at adjacent heights and the same query_id would land in the chain
        # twice. (Mempool fetch is peek-style — overlap is by design.)
        duplicates = [
            q.query_id for q in block.queries if q.query_id in self._included_query_ids
        ]
        if duplicates:
            return False, f"queries already in chain: {duplicates}"

        # 4. Lottery: H(G(s,x), (ct_1, ..., ct_k)) < D, and header's recorded
        #    hash must match what we'd compute from the ciphertexts.
        fingerprint = block_fingerprint(header.prev_hash, block.transactions)
        ciphertexts = [r.ciphertext for r in block.results]
        lottery_int, lottery_won = evaluate_lottery(
            fingerprint, ciphertexts, self.difficulty
        )
        if not lottery_won:
            return False, f"lottery hash {lottery_int} >= difficulty {self.difficulty}"
        if header.lottery_hash != lottery_int.to_bytes(32, "big"):
            return False, "lottery_hash in header does not match recomputed hash"

        # 5. Verify inference ZKPs (optional — expensive). The circuit only
        #    attests to the model computation; VRF and encryption are
        #    verified separately below.
        if verify_zkp:
            from poml_sim.zkp import verify_proof as _verify_proof

            for i, result in enumerate(block.results):
                if not _verify_proof(result.proof_bytes, artifacts_dir):
                    return False, f"inference proof {i} failed verification"

        # 5b. Chain binding: bind_1 = G(s,x), bind_i = H(ct_{i-1}) for i >= 2.
        for i, result in enumerate(block.results):
            if i == 0:
                expected_binding = fingerprint
            else:
                expected_binding = sha256(block.results[i - 1].ciphertext)
            if result.chain_binding != expected_binding:
                return False, f"chain binding mismatch at position {i}"

        # 5c. VRF transcript verification: every (z_{i,t}, pi^VRF_{i,t}) must
        #     verify against the miner's declared VRF vk on input r_i || t.
        if not header.miner_vrf_vk:
            return False, "header missing miner_vrf_vk"
        for i, result in enumerate(block.results):
            if len(result.vrf_transcript) != self.diffusion_steps:
                return (
                    False,
                    f"vrf transcript at position {i} has length "
                    f"{len(result.vrf_transcript)}, expected {self.diffusion_steps}",
                )
            for t, (y_t, pi_t) in enumerate(result.vrf_transcript, start=1):
                vrf_input = result.seed_used + struct.pack(">I", t)
                if not vrf_verify(header.miner_vrf_vk, vrf_input, y_t, pi_t):
                    return False, f"VRF proof {i},{t} failed verification"

        # 6. Validate transactions
        if account_state is not None:
            for tx in block.transactions:
                if not account_state.validate_transaction(tx):
                    return False, f"invalid transaction from {tx.sender.hex()[:16]}"

        return True, "ok"

    def add_block(
        self,
        block: Block,
        account_state: AccountState | None = None,
        verify_zkp: bool = False,
        artifacts_dir: str = "model/",
    ) -> bool:
        """Validate and append a block to the chain."""
        valid, reason = self.validate_block(block, account_state, verify_zkp, artifacts_dir)
        if not valid:
            logger.warning("Block %d rejected: %s", block.header.block_height, reason)
            return False

        self.chain.append(block)
        self._block_hashes[block.header.block_height] = hash_block(block)
        for q in block.queries:
            self._included_query_ids.add(q.query_id)
        logger.info(
            "Block %d added (miner=%s, queries=%d)",
            block.header.block_height,
            block.header.miner_pk.hex()[:12],
            len(block.queries),
        )
        return True
