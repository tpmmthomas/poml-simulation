"""Data types for PoML simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Query:
    """User inference query submitted to the mempool."""

    query_id: int
    user_pk: bytes  # signing public key
    task_id: int
    conditioning_input: list[float]
    commitment: bytes
    commitment_randomness: bytes
    fee: int
    signature: bytes
    encryption_pk: bytes = b""  # user's RSA public key (DER) for output encryption


@dataclass
class Transaction:
    """Simple account transfer transaction."""

    sender: bytes
    receiver: bytes
    amount: int
    signature: bytes


@dataclass
class InferenceResult:
    """Result of a single inference + proof generation."""

    query_id: int
    output: list[float]
    proof_bytes: bytes
    seed_used: bytes
    chain_binding: bytes = b""  # bind_i per paper §4.4 (H(pi_{i-1}) for i>=2)
    ciphertext: bytes = b""  # ct_i = Enc(pk_u, y || taskID; r_enc)
    # Pi^VRF_i: list of (z_t, pi_t) for t = 1..T binding the noise schedule
    # used by this inference. Verified per-block against header.miner_vrf_vk.
    vrf_transcript: list[tuple[bytes, bytes]] = field(default_factory=list)
    # Encryption VRF output and proof: r_enc, pi_enc from
    # VRF.Eval(sk_VRF_enc, seed || taskID). Verified against header.miner_enc_vrf_vk.
    enc_vrf_randomness: bytes = b""
    enc_vrf_proof: bytes = b""


@dataclass
class BlockHeader:
    """Block header containing metadata."""

    block_height: int
    prev_hash: bytes
    miner_pk: bytes
    timestamp: float
    difficulty: int
    lottery_hash: bytes
    # Miner's inference VRF verification key (Ed25519 raw). Self-declared per block.
    miner_vrf_vk: bytes = b""
    # Miner's encryption VRF verification key (Ed25519 raw). Self-declared per block.
    miner_enc_vrf_vk: bytes = b""


@dataclass
class Block:
    """Full block in the PoML chain."""

    header: BlockHeader
    queries: list[Query] = field(default_factory=list)
    results: list[InferenceResult] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)

    @property
    def block_hash(self) -> bytes:
        from poml_sim.crypto import hash_block

        return hash_block(self)
