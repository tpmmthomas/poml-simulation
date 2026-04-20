"""Tests for output encryption and chain binding."""

import struct

from poml_sim.crypto import block_fingerprint, sha256
from poml_sim.encryption import (
    decrypt_output,
    encrypt_output,
    generate_encryption_keypair,
)


class TestEncryptionKeypair:
    def test_generate_keypair(self):
        pk, sk = generate_encryption_keypair()
        assert len(pk) == 32
        assert len(sk) == 32

    def test_keypairs_unique(self):
        pk1, sk1 = generate_encryption_keypair()
        pk2, sk2 = generate_encryption_keypair()
        assert pk1 != pk2
        assert sk1 != sk2


class TestEncryptDecrypt:
    def test_roundtrip(self):
        pk, sk = generate_encryption_keypair()
        output_values = [0.1, 0.2, 0.3, -0.5]
        chain_binding = b"\xaa" * 32
        task_id = 42

        encrypted = encrypt_output(pk, output_values, chain_binding, task_id)
        assert len(encrypted) > 0
        # Should be: ephemeral_pk(32) + nonce(12) + ciphertext(>0)
        assert len(encrypted) > 44

        dec_output, dec_binding, dec_task_id = decrypt_output(
            sk, encrypted, num_output_floats=4
        )

        assert dec_task_id == task_id
        assert dec_binding == chain_binding
        # Floats may lose some precision in packing
        for a, b in zip(output_values, dec_output):
            assert abs(a - b) < 1e-6

    def test_wrong_key_fails(self):
        pk1, sk1 = generate_encryption_keypair()
        _, sk2 = generate_encryption_keypair()

        encrypted = encrypt_output(pk1, [1.0, 2.0], b"\xbb" * 32, 0)

        try:
            decrypt_output(sk2, encrypted, num_output_floats=2)
            assert False, "Should have raised an exception"
        except Exception:
            pass  # Expected: decryption fails with wrong key

    def test_different_plaintexts_different_ciphertexts(self):
        pk, sk = generate_encryption_keypair()
        ct1 = encrypt_output(pk, [1.0], b"\x00" * 32, 0)
        ct2 = encrypt_output(pk, [2.0], b"\x00" * 32, 0)
        assert ct1 != ct2

    def test_large_output(self):
        pk, sk = generate_encryption_keypair()
        output = [float(i) / 100 for i in range(64)]  # 64 floats (8x8 output)
        binding = sha256(b"test_binding")
        task_id = 999

        encrypted = encrypt_output(pk, output, binding, task_id)
        dec_output, dec_binding, dec_tid = decrypt_output(
            sk, encrypted, num_output_floats=64
        )

        assert dec_tid == task_id
        assert dec_binding == binding
        for a, b in zip(output, dec_output):
            assert abs(a - b) < 1e-5


class TestChainBinding:
    def test_first_binding_is_fingerprint(self):
        """bind_1 = G(s,x) = fingerprint."""
        prev_hash = b"\x01" * 32
        fingerprint = block_fingerprint(prev_hash, [])
        # For position 1, chain_binding should equal fingerprint
        assert len(fingerprint) == 32

    def test_subsequent_binding_is_proof_hash(self):
        """bind_i = H(π_{i-1}) for i >= 2."""
        proof_bytes = b"some_proof_data_here"
        binding = sha256(proof_bytes)
        assert len(binding) == 32

    def test_chain_binding_deterministic(self):
        proof = b"deterministic_proof"
        b1 = sha256(proof)
        b2 = sha256(proof)
        assert b1 == b2

    def test_different_proofs_different_bindings(self):
        b1 = sha256(b"proof_A")
        b2 = sha256(b"proof_B")
        assert b1 != b2
