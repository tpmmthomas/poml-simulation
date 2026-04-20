"""Tests for blockchain state management."""

import time

from poml_sim.accounts import AccountState
from poml_sim.blockchain import Blockchain, create_genesis_block
from poml_sim.crypto import evaluate_lottery, generate_keypair, hash_block, block_fingerprint
from poml_sim.types import Block, BlockHeader, InferenceResult, Query


def _make_query(query_id: int = 0) -> Query:
    pk, _ = generate_keypair()
    return Query(
        query_id=query_id,
        user_pk=pk,
        task_id=0,
        conditioning_input=[0.0] * 64,
        commitment=b"\xaa" * 32,
        commitment_randomness=b"\xbb" * 16,
        fee=5,
        signature=b"\xcc" * 32,
    )


def _make_result(query_id: int = 0) -> InferenceResult:
    return InferenceResult(
        query_id=query_id,
        output=[0.0] * 64,
        proof_bytes=b"fake_proof_bytes",
        meta_proof_dummy="ab" * 32,
        seed_used=b"\xdd" * 32,
    )


def _make_valid_block(chain: Blockchain, miner_pk: bytes) -> Block:
    """Create a block that passes the lottery check."""
    tip_hash = chain.get_tip_hash()
    difficulty = chain.difficulty

    queries = [_make_query(0)]
    results = [_make_result(0)]

    # Brute-force a proof that wins the lottery
    fingerprint = block_fingerprint(tip_hash, [])
    for nonce in range(100_000):
        proof_bytes = f"proof_{nonce}".encode()
        results[0] = InferenceResult(
            query_id=0,
            output=[0.0] * 64,
            proof_bytes=proof_bytes,
            meta_proof_dummy="ab" * 32,
            seed_used=b"\xdd" * 32,
        )
        lottery_int, won = evaluate_lottery(fingerprint, [proof_bytes], difficulty)
        if won:
            header = BlockHeader(
                block_height=chain.get_height() + 1,
                prev_hash=tip_hash,
                miner_pk=miner_pk,
                timestamp=time.time(),
                difficulty=difficulty,
                lottery_hash=lottery_int.to_bytes(32, "big"),
            )
            return Block(header=header, queries=queries, results=results)

    raise RuntimeError("Could not find a winning proof in 100k attempts — difficulty too hard for test")


class TestGenesis:
    def test_genesis_block(self):
        genesis = create_genesis_block()
        assert genesis.header.block_height == 0
        assert genesis.header.prev_hash == b"\x00" * 32

    def test_blockchain_init(self):
        bc = Blockchain(difficulty=2**256 - 1)
        assert bc.get_height() == 0


class TestBlockValidation:
    def test_valid_block(self):
        difficulty = 2**256 - 1  # Very easy — any hash wins
        bc = Blockchain(difficulty)
        pk, _ = generate_keypair()
        block = _make_valid_block(bc, pk)
        valid, reason = bc.validate_block(block)
        assert valid, reason

    def test_wrong_prev_hash(self):
        difficulty = 2**256 - 1
        bc = Blockchain(difficulty)
        pk, _ = generate_keypair()
        block = _make_valid_block(bc, pk)
        block.header.prev_hash = b"\xff" * 32
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "prev_hash" in reason

    def test_wrong_height(self):
        difficulty = 2**256 - 1
        bc = Blockchain(difficulty)
        pk, _ = generate_keypair()
        block = _make_valid_block(bc, pk)
        block.header.block_height = 5
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "height" in reason

    def test_empty_queries_rejected(self):
        difficulty = 2**256 - 1
        bc = Blockchain(difficulty)
        pk, _ = generate_keypair()
        header = BlockHeader(
            block_height=1,
            prev_hash=bc.get_tip_hash(),
            miner_pk=pk,
            timestamp=time.time(),
            difficulty=difficulty,
            lottery_hash=b"\x00" * 32,
        )
        block = Block(header=header)
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "no queries" in reason


class TestChainGrowth:
    def test_add_block(self):
        difficulty = 2**256 - 1
        bc = Blockchain(difficulty)
        pk, _ = generate_keypair()
        block = _make_valid_block(bc, pk)
        added = bc.add_block(block)
        assert added
        assert bc.get_height() == 1

    def test_add_multiple_blocks(self):
        difficulty = 2**256 - 1
        bc = Blockchain(difficulty)
        pk, _ = generate_keypair()
        for expected_height in range(1, 4):
            block = _make_valid_block(bc, pk)
            added = bc.add_block(block)
            assert added
            assert bc.get_height() == expected_height
