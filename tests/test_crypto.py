"""Tests for crypto utilities."""

import struct

from poml_sim.crypto import (
    block_fingerprint,
    derive_seed,
    evaluate_lottery,
    generate_dummy_meta_proof,
    generate_keypair,
    hash_block,
    seed_to_noise,
    sha256,
    sign,
    verify_signature,
)
from poml_sim.types import Block, BlockHeader


class TestSHA256:
    def test_deterministic(self):
        assert sha256(b"hello") == sha256(b"hello")

    def test_different_inputs(self):
        assert sha256(b"hello") != sha256(b"world")

    def test_multipart(self):
        combined = sha256(b"hello", b"world")
        assert len(combined) == 32


class TestBlockFingerprint:
    def test_deterministic(self):
        prev = b"\x01" * 32
        fp1 = block_fingerprint(prev, [])
        fp2 = block_fingerprint(prev, [])
        assert fp1 == fp2

    def test_different_prev_hash(self):
        fp1 = block_fingerprint(b"\x01" * 32, [])
        fp2 = block_fingerprint(b"\x02" * 32, [])
        assert fp1 != fp2


class TestDeriveSeed:
    def test_deterministic(self):
        fp = b"\xaa" * 32
        commitment = b"\xbb" * 32
        s1 = derive_seed(fp, commitment, 0, b"\xcc" * 32, 1)
        s2 = derive_seed(fp, commitment, 0, b"\xcc" * 32, 1)
        assert s1 == s2
        assert len(s1) == 32

    def test_different_position(self):
        fp = b"\xaa" * 32
        s1 = derive_seed(fp, b"\xbb" * 32, 0, b"\xcc" * 32, 1)
        s2 = derive_seed(fp, b"\xbb" * 32, 0, b"\xcc" * 32, 2)
        assert s1 != s2

    def test_different_miner(self):
        fp = b"\xaa" * 32
        s1 = derive_seed(fp, b"\xbb" * 32, 0, b"\xcc" * 32, 1)
        s2 = derive_seed(fp, b"\xbb" * 32, 0, b"\xdd" * 32, 1)
        assert s1 != s2


class TestLottery:
    def test_easy_difficulty_always_wins(self):
        """With max difficulty (2^256 - 1), lottery always passes."""
        max_diff = 2**256 - 1
        fp = b"\x00" * 32
        proofs = [b"proof1", b"proof2"]
        _, won = evaluate_lottery(fp, proofs, max_diff)
        assert won is True

    def test_zero_difficulty_never_wins(self):
        """With difficulty 0, lottery never passes."""
        fp = b"\x00" * 32
        proofs = [b"proof1"]
        _, won = evaluate_lottery(fp, proofs, 0)
        assert won is False

    def test_deterministic(self):
        fp = b"\xaa" * 32
        proofs = [b"p1", b"p2"]
        diff = 2**255
        h1, w1 = evaluate_lottery(fp, proofs, diff)
        h2, w2 = evaluate_lottery(fp, proofs, diff)
        assert h1 == h2
        assert w1 == w2


class TestSeedToNoise:
    def test_length(self):
        seed = b"\x42" * 32
        noise = seed_to_noise(seed, 64)
        assert len(noise) == 64

    def test_deterministic(self):
        seed = b"\x42" * 32
        n1 = seed_to_noise(seed, 64)
        n2 = seed_to_noise(seed, 64)
        assert n1 == n2

    def test_range(self):
        seed = b"\x42" * 32
        noise = seed_to_noise(seed, 64)
        for v in noise:
            assert -1.0 <= v <= 1.0


class TestKeypairAndSign:
    def test_keypair_unique(self):
        pk1, sk1 = generate_keypair()
        pk2, sk2 = generate_keypair()
        assert pk1 != pk2
        assert sk1 != sk2

    def test_sign_verify(self):
        pk, sk = generate_keypair()
        data = b"test message"
        sig = sign(sk, data)
        assert verify_signature(pk, data, sig, sk) is True

    def test_wrong_key_fails(self):
        _, sk1 = generate_keypair()
        pk2, sk2 = generate_keypair()
        sig = sign(sk1, b"data")
        assert verify_signature(pk2, b"data", sig, sk2) is False


class TestMetaProof:
    def test_length(self):
        mp = generate_dummy_meta_proof()
        assert len(mp) == 64

    def test_unique(self):
        mp1 = generate_dummy_meta_proof()
        mp2 = generate_dummy_meta_proof()
        assert mp1 != mp2


class TestHashBlock:
    def test_deterministic(self):
        header = BlockHeader(
            block_height=1,
            prev_hash=b"\x00" * 32,
            miner_pk=b"\xaa" * 32,
            timestamp=1000.0,
            difficulty=100,
            lottery_hash=b"\xbb" * 32,
        )
        block = Block(header=header)
        h1 = hash_block(block)
        h2 = hash_block(block)
        assert h1 == h2
        assert len(h1) == 32
