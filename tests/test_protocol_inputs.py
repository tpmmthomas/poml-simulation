"""Cryptographic plumbing and deterministic Gaussian/encryption regressions."""

import math
import pytest
from cryptography.exceptions import InvalidTag
from poml_sim.crypto import public_key
from poml_sim.protocol_inputs import (
    experimental_key,
    gaussian_vector,
    encryption_public_key,
    encrypt_output,
    decrypt_output,
)
from poml_sim.vrf import vrf_eval, vrf_verify


def test_vrf_is_deterministic_and_binds_key_input_and_output():
    key = experimental_key(42, "inference")
    z, proof = vrf_eval(key, b"seed")
    assert vrf_eval(key, b"seed") == (z, proof)
    assert vrf_verify(public_key(key), b"seed", z, proof)
    assert not vrf_verify(public_key(key), b"other", z, proof)
    assert not vrf_verify(public_key(experimental_key(42, "encryption")), b"seed", z, proof)
    assert not vrf_verify(public_key(key), b"seed", bytes(32), proof)
    assert not vrf_verify(public_key(key), b"seed", z, b"bad")


def test_gaussian_expansion_is_finite_deterministic_and_approximately_normal():
    values = gaussian_vector(b"seed", 20000)
    assert values == gaussian_vector(b"seed", 20000)
    assert values != gaussian_vector(b"other", 20000)
    assert all(math.isfinite(x) for x in values)
    assert abs(sum(values) / len(values)) < 0.04
    assert abs(sum(x * x for x in values) / len(values) - 1) < 0.05


def test_public_key_encryption_uses_explicit_coins_and_authenticates_tampering():
    secret = experimental_key(10, "recipient")
    public = encryption_public_key(secret)
    encrypted = encrypt_output(b"private output", public, b"a" * 32)
    assert encrypted == encrypt_output(b"private output", public, b"a" * 32)
    assert encrypted != encrypt_output(b"private output", public, b"b" * 32)
    assert decrypt_output(encrypted, secret) == b"private output"
    with pytest.raises(InvalidTag):
        decrypt_output(encrypted[:-1] + bytes([encrypted[-1] ^ 1]), secret)
    with pytest.raises(InvalidTag):
        decrypt_output(encrypted, experimental_key(11, "recipient"))
