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
    encryption_pk: bytes = b""  # X25519 public key for output encryption


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
    meta_proof_dummy: str
    seed_used: bytes
    chain_binding: bytes = b""  # bind_i per paper §4.4
    encrypted_output: bytes = b""  # ct_i = Enc(pk_u, y || bind_i || taskID)


@dataclass
class BlockHeader:
    """Block header containing metadata."""

    block_height: int
    prev_hash: bytes
    miner_pk: bytes
    timestamp: float
    difficulty: int
    lottery_hash: bytes


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
