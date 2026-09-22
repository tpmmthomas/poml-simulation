"""Checkable deterministic Ed25519 VRF substitute for simulation.

This is sign-then-hash, not the paper's formally unbiasable VRF. The
simulator's reproducible keys must never be used as production secrets.
"""

import os
from .crypto import public_key, sha256, sign, verify_signature

DOMAIN = b"poml-vrf-v1\0"


def generate_vrf_keypair():
    """Return a random Ed25519 public key and its private seed."""
    secret = os.urandom(32)
    return public_key(secret), secret


def vrf_eval(secret, message):
    """Evaluate one indexed challenge by signing and hashing the signature."""
    proof = sign(secret, DOMAIN + message)
    return sha256(proof), proof


def vrf_verify(public, message, output, proof):
    """Reject malformed signatures and mismatching output digests."""
    return verify_signature(public, DOMAIN + message, proof) and output == sha256(proof)
