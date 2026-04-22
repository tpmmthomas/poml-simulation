"""Tests for blockchain state management."""

import struct
import time

from poml_sim.blockchain import Blockchain, create_genesis_block
from poml_sim.crypto import (
    block_fingerprint,
    evaluate_lottery,
    generate_keypair,
    sha256,
)
from poml_sim.types import Block, BlockHeader, InferenceResult, Query
from poml_sim.vrf import generate_vrf_keypair, vrf_eval


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


def _make_vrf_transcript(
    vrf_sk: bytes, seed: bytes, T: int
) -> list[tuple[bytes, bytes]]:
    return [vrf_eval(vrf_sk, seed + struct.pack(">I", t)) for t in range(1, T + 1)]


def _make_result(
    query_id: int,
    ciphertext: bytes,
    chain_binding: bytes,
    vrf_sk: bytes,
    seed: bytes,
    T: int,
) -> InferenceResult:
    return InferenceResult(
        query_id=query_id,
        output=[0.0] * 64,
        proof_bytes=b"fake_proof_bytes",
        seed_used=seed,
        chain_binding=chain_binding,
        ciphertext=ciphertext,
        vrf_transcript=_make_vrf_transcript(vrf_sk, seed, T),
    )


def _make_valid_block(
    chain: Blockchain,
    miner_pk: bytes,
    vrf_vk: bytes,
    vrf_sk: bytes,
    num_results: int = 1,
) -> Block:
    """Build a block that passes every validator check (lottery, binding, VRF)."""
    tip_hash = chain.get_tip_hash()
    difficulty = chain.difficulty
    T = chain.diffusion_steps
    fingerprint = block_fingerprint(tip_hash, [])

    queries = [_make_query(i) for i in range(num_results)]

    # Brute-force a list of ciphertexts whose cumulative hash wins the
    # lottery. With max difficulty this hits on the first try.
    for nonce in range(100_000):
        ciphertexts = [f"ct_{nonce}_{i}".encode() for i in range(num_results)]
        lottery_int, won = evaluate_lottery(fingerprint, ciphertexts, difficulty)
        if won:
            break
    else:
        raise RuntimeError("could not win lottery in 100k attempts")

    results: list[InferenceResult] = []
    for i, ct in enumerate(ciphertexts):
        binding = fingerprint if i == 0 else sha256(ciphertexts[i - 1])
        seed = sha256(b"seed", struct.pack(">I", i))
        results.append(_make_result(i, ct, binding, vrf_sk, seed, T))

    header = BlockHeader(
        block_height=chain.get_height() + 1,
        prev_hash=tip_hash,
        miner_pk=miner_pk,
        timestamp=time.time(),
        difficulty=difficulty,
        lottery_hash=lottery_int.to_bytes(32, "big"),
        miner_vrf_vk=vrf_vk,
    )
    return Block(header=header, queries=queries, results=results)


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
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        valid, reason = bc.validate_block(block)
        assert valid, reason

    def test_wrong_prev_hash(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        block.header.prev_hash = b"\xff" * 32
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "prev_hash" in reason

    def test_wrong_height(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        block.header.block_height = 5
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "height" in reason

    def test_empty_queries_rejected(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, _ = generate_vrf_keypair()
        header = BlockHeader(
            block_height=1,
            prev_hash=bc.get_tip_hash(),
            miner_pk=pk,
            timestamp=time.time(),
            difficulty=bc.difficulty,
            lottery_hash=b"\x00" * 32,
            miner_vrf_vk=vk,
        )
        block = Block(header=header)
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "no queries" in reason

    def test_tampered_vrf_output_rejected(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        # Flip a byte in the first VRF output y_1.
        y_t, pi_t = block.results[0].vrf_transcript[0]
        tampered_y = bytes([y_t[0] ^ 0xFF]) + y_t[1:]
        block.results[0].vrf_transcript[0] = (tampered_y, pi_t)
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "VRF" in reason

    def test_wrong_vrf_vk_rejected(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        # Swap in a different verification key.
        other_vk, _ = generate_vrf_keypair()
        block.header.miner_vrf_vk = other_vk
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "VRF" in reason

    def test_wrong_chain_binding_rejected(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk, num_results=2)
        block.results[1].chain_binding = b"\xff" * 32
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "chain binding" in reason

    def test_forged_lottery_hash_rejected(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        # Swap header's lottery_hash for something that doesn't match the ct.
        block.header.lottery_hash = b"\x00" * 32
        valid, reason = bc.validate_block(block)
        assert not valid
        assert "lottery_hash" in reason


class TestChainGrowth:
    def test_add_block(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        block = _make_valid_block(bc, pk, vk, sk)
        assert bc.add_block(block)
        assert bc.get_height() == 1

    def test_add_multiple_blocks(self):
        bc = Blockchain(2**256 - 1)
        pk, _ = generate_keypair()
        vk, sk = generate_vrf_keypair()
        for expected_height in range(1, 4):
            block = _make_valid_block(bc, pk, vk, sk)
            assert bc.add_block(block)
            assert bc.get_height() == expected_height
