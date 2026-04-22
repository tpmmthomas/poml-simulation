"""Verifiable Random Function used to derive per-inference noise.

The paper requires an unbiasable VRF. This module implements a
simulation-grade "sign-then-hash" VRF on top of RFC 8032 Ed25519:

    VRF.Gen()          -> (vk, sk)
    VRF.Eval(sk, x)    -> (y, pi)   where pi = Ed25519.sign(sk, x),
                                          y  = SHA256(pi)
    VRF.Vfy(vk, x, y, pi) -> Ed25519.verify(vk, x, pi) and y == SHA256(pi)

Ed25519 signatures are deterministic per (sk, message), which gives the
VRF uniqueness + pseudorandomness under the random-oracle model. It is
*not* a formally unbiasable VRF (a real deployment would use RFC 9381
ECVRF); this trade-off is documented in the simulation README.
"""

from __future__ import annotations

import struct

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)

from poml_sim.crypto import seed_to_noise, sha256

# Domain separator keeps VRF inputs from colliding with other SHA256 usage.
_VRF_DOMAIN = b"poml-vrf-v1\x00"


def generate_vrf_keypair() -> tuple[bytes, bytes]:
    """Return (verification_key, secret_key) as raw 32-byte Ed25519 keys."""
    sk_obj = Ed25519PrivateKey.generate()
    sk = sk_obj.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    vk = sk_obj.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return vk, sk


def vrf_eval(sk: bytes, x: bytes) -> tuple[bytes, bytes]:
    """Evaluate the VRF: return (y, pi) where y is pseudorandom 32 bytes."""
    sk_obj = Ed25519PrivateKey.from_private_bytes(sk)
    pi = sk_obj.sign(_VRF_DOMAIN + x)
    y = sha256(pi)
    return y, pi


def vrf_verify(vk: bytes, x: bytes, y: bytes, pi: bytes) -> bool:
    """Verify (y, pi) is the unique VRF output for x under vk."""
    try:
        Ed25519PublicKey.from_public_bytes(vk).verify(pi, _VRF_DOMAIN + x)
    except InvalidSignature:
        return False
    return y == sha256(pi)


def vrf_noise_schedule(
    sk: bytes,
    r_i: bytes,
    T: int,
    spatial: int,
) -> tuple[list[list[float]], list[tuple[bytes, bytes]]]:
    """Produce the paper's noise schedule U_i and VRF transcript Pi^VRF_i.

    For t = 1..T, evaluates the VRF on (r_i || t) and expands the resulting
    pseudorandom y_t into `spatial` floats in [-1, 1]. Returns the schedule
    U_i = [z_{i,1}, ..., z_{i,T}] and the transcript [(y_t, pi_t), ...].
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    schedule: list[list[float]] = []
    transcript: list[tuple[bytes, bytes]] = []
    for t in range(1, T + 1):
        y_t, pi_t = vrf_eval(sk, r_i + struct.pack(">I", t))
        schedule.append(seed_to_noise(y_t, spatial))
        transcript.append((y_t, pi_t))
    return schedule, transcript
