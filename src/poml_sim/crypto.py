"""Cryptographic utilities for PoML simulation.

Implements:
- Block fingerprint G(s,x) = SHA-256(prev_hash || hash(txns))
- Seed derivation r_i = H(G(s,x) || commitment_i || taskID_i || pk_m || i)
- Lottery evaluation H(G(s,x), (ct_1, ..., ct_i)) < D
- Block hashing for chaining
- Ed25519 signing stubs (HMAC-based for simplicity)
- Deterministic expansion of a 32-byte seed to float noise (reused by the
  VRF module to turn each VRF output y_t into model-input noise)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poml_sim.types import Block, Query, Transaction


def sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def block_fingerprint(prev_hash: bytes, transactions: list[Transaction]) -> bytes:
    """G(s,x) = SHA-256(prev_hash || hash(txns)).

    The 'fingerprint' of the block content before proofs are added.
    """
    txn_hash = sha256(
        b"".join(
            sha256(tx.sender, tx.receiver, struct.pack(">q", tx.amount))
            for tx in transactions
        )
        if transactions
        else b"\x00" * 32
    )
    return sha256(prev_hash, txn_hash)


def derive_seed(
    bind_i: bytes,
    commitment: bytes,
    task_id: int,
    miner_vk_sig: bytes,
) -> bytes:
    """r_i = H(bind_i || h_{u,i} || taskID_i || vk^sig_m).

    Per the revised paper: bind_i is G(s,tx) for the first query and
    H(pi_{i-1}) for subsequent ones; commitment is the query input hash
    h_{u,i} = H_zk(x_u || taskID); miner_vk_sig is the miner's identity
    verification key vk^sig_m.
    """
    return sha256(
        bind_i,
        commitment,
        struct.pack(">I", task_id),
        miner_vk_sig,
    )


def evaluate_lottery(
    fingerprint: bytes,
    ciphertexts_so_far: list[bytes],
    difficulty: int,
) -> tuple[int, bool]:
    """H_i = H(G(s,x), (ct_1, ..., ct_i)) < D.

    Paper's revised lottery: hashes the block fingerprint together with
    the ciphertext tuple Y_i produced so far, not the ZKPs. Returns
    (lottery_hash_as_int, won).
    """
    lottery_hash = sha256(fingerprint, *ciphertexts_so_far)
    hash_int = int.from_bytes(lottery_hash, "big")
    return hash_int, hash_int < difficulty


def hash_block(block: Block) -> bytes:
    """Compute the hash of a block for chaining."""
    header = block.header
    parts = [
        struct.pack(">I", header.block_height),
        header.prev_hash,
        header.miner_pk,
        header.miner_vrf_vk,
        struct.pack(">d", header.timestamp),
        header.difficulty.to_bytes(32, "big"),
        header.lottery_hash,
    ]
    # Include query commitments
    for q in block.queries:
        parts.append(q.commitment)
    # Include ciphertext, proof hash, and VRF transcript hashes per result
    for r in block.results:
        parts.append(r.ciphertext)
        parts.append(sha256(r.proof_bytes))
        for y_t, pi_t in r.vrf_transcript:
            parts.append(y_t)
            parts.append(sha256(pi_t))
    # Include transaction hashes
    for tx in block.transactions:
        parts.append(sha256(tx.sender, tx.receiver, struct.pack(">q", tx.amount)))
    return sha256(*parts)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a simple keypair (32-byte random key used as both sk and to derive pk)."""
    sk = os.urandom(32)
    pk = sha256(b"pk_derive", sk)
    return pk, sk


def sign(sk: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 signature (simplified stand-in for Ed25519)."""
    return hmac.new(sk, data, hashlib.sha256).digest()


def verify_signature(pk: bytes, data: bytes, signature: bytes, sk: bytes | None = None) -> bool:
    """Verify HMAC signature.

    In a real system this would use the pk directly.  For our HMAC stub we
    need the sk; in the simulation the coordinator holds all keys so this
    works.  When sk is None we optimistically return True (used during
    block validation where we trust the coordinator already checked).
    """
    if sk is None:
        return True
    expected = hmac.new(sk, data, hashlib.sha256).digest()
    return hmac.compare_digest(expected, signature)


def seed_to_noise(seed: bytes, spatial_size: int = 64) -> list[float]:
    """Derive deterministic noise values from a seed.

    Expands the 32-byte seed into `spatial_size` float values in [-1, 1]
    using iterated hashing.
    """
    values: list[float] = []
    current = seed
    while len(values) < spatial_size:
        current = sha256(current, struct.pack(">I", len(values)))
        # Take 4 bytes at a time and convert to float in [-1, 1]
        for i in range(0, 32, 4):
            if len(values) >= spatial_size:
                break
            val = struct.unpack(">I", current[i : i + 4])[0]
            values.append((val / (2**32 - 1)) * 2 - 1)
    return values[:spatial_size]
