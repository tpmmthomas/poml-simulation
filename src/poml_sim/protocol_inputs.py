"""Public randomness expansion and deterministic-coin hybrid encryption."""

import hashlib
import math
import struct
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .crypto import canonical, sha256


def experimental_key(seed: int, domain: str) -> bytes:
    """Derive public experiment-only keys; these provide no secrecy."""
    return sha256(canonical(["poml-experimental-key", seed, domain]))


def gaussian_vector(seed: bytes, count: int) -> list[float]:
    """Expand a VRF output with SHAKE256 and unclipped Box-Muller normals."""
    if count < 1:
        raise ValueError("positive Gaussian dimension required")
    size = count + count % 2
    words = struct.unpack(
        f">{size}Q", hashlib.shake_256(b"poml-gaussian-v1" + seed).digest(size * 8)
    )
    result = []
    for index in range(0, size, 2):
        # Midpoints of a 52-bit grid exclude both zero and one in binary64.
        u = ((words[index] >> 12) + 0.5) / (1 << 52)
        v = ((words[index + 1] >> 12) + 0.5) / (1 << 52)
        radius = math.sqrt(-2 * math.log(u))
        result.extend((radius * math.cos(2 * math.pi * v), radius * math.sin(2 * math.pi * v)))
    return result[:count]


def encryption_public_key(secret: bytes) -> bytes:
    """Return a user's raw X25519 encryption key."""
    return (
        X25519PrivateKey.from_private_bytes(secret)
        .public_key()
        .public_bytes(Encoding.Raw, PublicFormat.Raw)
    )


def encrypt_output(plaintext: bytes, recipient_public: bytes, randomness: bytes) -> bytes:
    """Encrypt using X25519, HKDF and AES-GCM with uniquely derived coins.

    The host simulator sees the coins and witnesses; it does not claim the
    output privacy of the paper's full private NP relation.
    """
    ephemeral = X25519PrivateKey.from_private_bytes(sha256(b"ephemeral", randomness))
    recipient = X25519PublicKey.from_public_bytes(recipient_public)
    public = ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"poml-output-v1").derive(
        ephemeral.exchange(recipient)
    )
    nonce = sha256(b"nonce", randomness)[:12]
    return public + nonce + AESGCM(key).encrypt(nonce, plaintext, b"poml-output-v1")


def decrypt_output(ciphertext: bytes, secret: bytes) -> bytes:
    """Recover the response with the user's private encryption key."""
    user = X25519PrivateKey.from_private_bytes(secret)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"poml-output-v1").derive(
        user.exchange(X25519PublicKey.from_public_bytes(ciphertext[:32]))
    )
    return AESGCM(key).decrypt(ciphertext[32:44], ciphertext[44:], b"poml-output-v1")
