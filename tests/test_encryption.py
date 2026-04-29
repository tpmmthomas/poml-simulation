"""Tests for randomized output encryption and VRF-based encryption randomness."""

import pytest

from poml_sim.crypto import block_fingerprint, sha256
from poml_sim.encryption import (
    decrypt_output,
    encrypt_output,
    generate_encryption_keypair,
)


class TestEncryptionKeypair:
    def test_generate_keypair(self):
        pk, sk = generate_encryption_keypair()
        # RSA-2048 keys serialize to a few hundred bytes of DER.
        assert len(pk) > 200
        assert len(sk) > 1000

    def test_keypairs_unique(self):
        pk1, sk1 = generate_encryption_keypair()
        pk2, sk2 = generate_encryption_keypair()
        assert pk1 != pk2
        assert sk1 != sk2


class TestEncryptDecrypt:
    def test_roundtrip(self):
        pk, sk = generate_encryption_keypair()
        output_values = [0.1, 0.2, 0.3, -0.5]
        enc_randomness = b"\xaa" * 32
        task_id = 42

        encrypted = encrypt_output(pk, output_values, enc_randomness, task_id)
        # Textbook RSA-2048: ciphertext is a whole number of 256-byte blocks.
        assert len(encrypted) > 0
        assert len(encrypted) % 256 == 0

        dec_output, dec_enc_randomness, dec_task_id = decrypt_output(
            sk, encrypted, num_output_floats=4
        )

        assert dec_task_id == task_id
        assert dec_enc_randomness == enc_randomness
        for a, b in zip(output_values, dec_output):
            assert abs(a - b) < 1e-6

    def test_wrong_key_fails(self):
        pk1, _ = generate_encryption_keypair()
        _, sk2 = generate_encryption_keypair()

        encrypted = encrypt_output(pk1, [1.0, 2.0], b"\xbb" * 32, 0)

        with pytest.raises(Exception):
            # Decrypting with a different key yields garbage; the 4-byte
            # length prefix will be nonsense, so downstream unpacking raises.
            decrypt_output(sk2, encrypted, num_output_floats=2)

    def test_different_plaintexts_different_ciphertexts(self):
        pk, _ = generate_encryption_keypair()
        ct1 = encrypt_output(pk, [1.0], b"\x00" * 32, 0)
        ct2 = encrypt_output(pk, [2.0], b"\x00" * 32, 0)
        assert ct1 != ct2

    def test_same_plaintext_same_ciphertext(self):
        """Determinism: repeated encryption with the same (pk, r_enc, y, taskID)
        yields identical ciphertext — required so the lottery hash is stable."""
        pk, _ = generate_encryption_keypair()
        values = [0.1 * i for i in range(64)]
        enc_randomness = b"\xab" * 32
        tid = 777
        ct1 = encrypt_output(pk, values, enc_randomness, tid)
        ct2 = encrypt_output(pk, values, enc_randomness, tid)
        ct3 = encrypt_output(pk, values, enc_randomness, tid)
        assert ct1 == ct2 == ct3

    def test_large_output(self):
        pk, sk = generate_encryption_keypair()
        output = [float(i) / 100 for i in range(64)]
        enc_randomness = sha256(b"test_enc_randomness")
        task_id = 999

        encrypted = encrypt_output(pk, output, enc_randomness, task_id)
        dec_output, dec_enc_randomness, dec_tid = decrypt_output(
            sk, encrypted, num_output_floats=64
        )

        assert dec_tid == task_id
        assert dec_enc_randomness == enc_randomness
        for a, b in zip(output, dec_output):
            assert abs(a - b) < 1e-5

    def test_ciphertext_length_is_multiple_of_modulus(self):
        pk, _ = generate_encryption_keypair()
        ct = encrypt_output(pk, [0.0] * 64, b"\x00" * 32, 1)
        assert len(ct) % 256 == 0


class TestChainBinding:
    def test_first_binding_is_fingerprint(self):
        """bind_1 = G(s,tx) = fingerprint."""
        fingerprint = block_fingerprint(b"\x01" * 32, [])
        assert len(fingerprint) == 32

    def test_subsequent_binding_is_proof_hash(self):
        """Revised paper: bind_i = H(pi_{i-1}) for i >= 2."""
        proof_bytes = b"some_proof_bytes_here"
        binding = sha256(proof_bytes)
        assert len(binding) == 32

    def test_chain_binding_deterministic(self):
        proof = b"deterministic_proof"
        assert sha256(proof) == sha256(proof)

    def test_different_proofs_different_bindings(self):
        assert sha256(b"proof_A") != sha256(b"proof_B")
