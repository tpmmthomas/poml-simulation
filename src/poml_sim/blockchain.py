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

from poml_sim.crypto import evaluate_lottery, hash_block, sha256
from poml_sim.types import Block, BlockHeader

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

    def __init__(self, difficulty: int) -> None:
        self.difficulty = difficulty
        self.chain: list[Block] = [create_genesis_block()]
        self._block_hashes: dict[int, bytes] = {0: hash_block(self.chain[0])}

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

        # 4. Lottery hash < difficulty
        lottery_int = int.from_bytes(header.lottery_hash, "big")
        if lottery_int >= self.difficulty:
            return False, f"lottery hash {lottery_int} >= difficulty {self.difficulty}"

        # 5. Verify inference proofs (optional — expensive)
        if verify_zkp:
            from poml_sim.zkp import verify_proof as _verify_proof

            for i, result in enumerate(block.results):
                if not _verify_proof(result.proof_bytes, artifacts_dir):
                    return False, f"inference proof {i} failed verification"

        # 6. Meta-proof verification: always passes (dummy)

        # 7. Validate transactions
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
        logger.info(
            "Block %d added (miner=%s, queries=%d)",
            block.header.block_height,
            block.header.miner_pk.hex()[:12],
            len(block.queries),
        )
        return True
