"""Canonical encodings, signatures and hashes for the educational protocol."""

from dataclasses import asdict, is_dataclass
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def canonical(value) -> bytes:
    """Encode protocol objects unambiguously, including typed byte strings."""

    def convert(item):
        if is_dataclass(item):
            return convert(asdict(item))
        if isinstance(item, bytes):
            return {"bytes": item.hex()}
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(val) for val in item]
        return item

    return json.dumps(
        convert(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(*parts: bytes) -> bytes:
    """Hash an already unambiguous sequence of protocol fields."""
    return hashlib.sha256(b"".join(parts)).digest()


def public_key(secret: bytes) -> bytes:
    """Obtain an Ed25519 public key from its raw private seed."""
    return (
        Ed25519PrivateKey.from_private_bytes(secret)
        .public_key()
        .public_bytes(Encoding.Raw, PublicFormat.Raw)
    )


def sign(secret: bytes, message: bytes) -> bytes:
    """Sign a canonical transaction with Ed25519."""
    return Ed25519PrivateKey.from_private_bytes(secret).sign(message)


def verify_signature(public: bytes, message: bytes, signature: bytes) -> bool:
    """Reject malformed keys and invalid signatures without trusting a sender."""
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
