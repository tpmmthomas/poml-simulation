"""Canonical protocol bindings, deterministic experimental keys, and noise."""

import hashlib
import json
import math
import struct

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .vrf import vrf_eval


def canonical(value) -> bytes:
    """Encode metadata deterministically; binary sequences use length framing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def frame(value: bytes) -> bytes:
    """Make concatenated variable-length protocol values unambiguous."""
    return len(value).to_bytes(8, "big") + value


def digest(value: bytes) -> bytes:
    """Instantiate the experiment's 256-bit hash function."""
    return hashlib.sha256(value).digest()


def experimental_key(seed: int, domain: str) -> bytes:
    """Derive reproducible experiment-only keys, never production secrets."""
    return digest(canonical(["poml-experimental-key", seed, domain]))


def public_signing_key(secret: bytes) -> bytes:
    """Return the identity public key used in the per-query binding."""
    return (
        Ed25519PrivateKey.from_private_bytes(secret)
        .public_key()
        .public_bytes(Encoding.Raw, PublicFormat.Raw)
    )


def query_binding(
    bind: bytes, prompt_tokens: list[int], qid: str, vk: bytes
) -> tuple[bytes, bytes]:
    """Compute h_u and r_i using the query, miner, and preceding proof/block."""
    q = frame(qid.encode())
    hu = digest(canonical(prompt_tokens) + q)
    return hu, digest(bind + hu + q + vk)


def gaussian_noise(
    secret: bytes, r: bytes, n: int, cap: int, alpha: float, embedding_std: float
) -> tuple[list[list[float]], list[dict]]:
    """Expand inference VRFs to N Gaussian embedding rows; reserve cap draws too.

    The remaining cap VRFs supply independent 53-bit inverse-CDF uniforms;
    draws reserved for tokens after EOS remain unused.
    """
    if n < 2 or cap < 1 or alpha < 0 or embedding_std <= 0:
        raise ValueError("invalid noise dimensions or scale")
    noise, transcript = [], []
    for t in range(1, n + cap + 1):
        z, proof = vrf_eval(secret, r + t.to_bytes(8, "big"))
        transcript.append({"t": t, "z": z.hex(), "proof": proof.hex()})
        if t > n:
            continue
        words = struct.unpack(
            ">768Q", hashlib.shake_256(b"poml-gaussian-v1" + z).digest(768 * 8)
        )
        row = []
        for i in range(0, 768, 2):
            u = ((words[i] >> 11) + 0.5) / (1 << 53)
            v = ((words[i + 1] >> 11) + 0.5) / (1 << 53)
            radius = math.sqrt(-2 * math.log(u)) * alpha * embedding_std
            row.extend(
                [radius * math.cos(2 * math.pi * v), radius * math.sin(2 * math.pi * v)]
            )
        noise.append(row)
    return noise, transcript


def encrypt_output(plaintext: bytes, user_secret: bytes, randomness: bytes) -> bytes:
    """Use X25519/HKDF/AES-GCM with VRF-derived ephemeral randomness.

    Encryption is real and reproducible. Its relationship to proof outputs is
    checked by the honest host, not constrained inside the inference proof.
    """
    ephemeral = X25519PrivateKey.from_private_bytes(digest(b"ephemeral" + randomness))
    recipient = X25519PrivateKey.from_private_bytes(user_secret).public_key()
    public = ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"poml-output-v1"
    ).derive(ephemeral.exchange(recipient))
    nonce = digest(b"nonce" + randomness)[:12]
    return public + nonce + AESGCM(key).encrypt(nonce, plaintext, b"poml-output-v1")


def decrypt_output(ciphertext: bytes, user_secret: bytes) -> bytes:
    """Recover an experimental response for end-to-end consistency checks."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    user = X25519PrivateKey.from_private_bytes(user_secret)
    key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"poml-output-v1"
    ).derive(user.exchange(X25519PublicKey.from_public_bytes(ciphertext[:32])))
    return AESGCM(key).decrypt(ciphertext[32:44], ciphertext[44:], b"poml-output-v1")


def decoding_uniforms(transcript: list[dict], prompt_length: int) -> list[int]:
    """Extract one exact 53-bit uniform from each reserved decoding VRF."""
    return [
        int.from_bytes(bytes.fromhex(row["z"])[:8], "big") >> 11
        for row in transcript[prompt_length:]
    ]
