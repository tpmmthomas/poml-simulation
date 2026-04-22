"""Targeted tests for deterministic RSA encryption.

These are focused on the *determinism* property required by the revised
PoML paper: the lottery hash over ciphertexts must be uniquely determined
by (pk, plaintext), which rules out any randomised PKE scheme. Other
encryption behaviour is exercised by tests/test_encryption.py.
"""

import pytest

from poml_sim.encryption import (
    decrypt_output,
    encrypt_output,
    generate_encryption_keypair,
)


def _values(n: int) -> list[float]:
    return [0.01 * i - 0.5 for i in range(n)]


class TestDeterminism:
    def test_repeated_encryption_identical(self):
        pk, _ = generate_encryption_keypair()
        values = _values(64)
        bind = b"\x01" * 32
        tid = 42
        ciphertexts = {encrypt_output(pk, values, bind, tid) for _ in range(5)}
        assert len(ciphertexts) == 1

    def test_different_binding_differs(self):
        pk, _ = generate_encryption_keypair()
        ct1 = encrypt_output(pk, _values(8), b"\x00" * 32, 0)
        ct2 = encrypt_output(pk, _values(8), b"\x01" * 32, 0)
        assert ct1 != ct2

    def test_different_task_id_differs(self):
        pk, _ = generate_encryption_keypair()
        ct1 = encrypt_output(pk, _values(8), b"\x00" * 32, 0)
        ct2 = encrypt_output(pk, _values(8), b"\x00" * 32, 1)
        assert ct1 != ct2


class TestRoundtripSizes:
    @pytest.mark.parametrize("n", [1, 4, 16, 64, 128])
    def test_roundtrip(self, n):
        pk, sk = generate_encryption_keypair()
        values = _values(n)
        bind = b"\xab" * 32
        tid = n * 7
        ct = encrypt_output(pk, values, bind, tid)
        assert len(ct) % 256 == 0
        dec_vals, dec_bind, dec_tid = decrypt_output(sk, ct, n)
        assert dec_tid == tid
        assert dec_bind == bind
        for a, b in zip(values, dec_vals):
            assert abs(a - b) < 1e-6


class TestBadInput:
    def test_ciphertext_wrong_length(self):
        _, sk = generate_encryption_keypair()
        with pytest.raises(ValueError):
            decrypt_output(sk, b"\x00" * 100, num_output_floats=1)
