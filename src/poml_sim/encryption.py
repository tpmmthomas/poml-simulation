"""Output encryption for PoML simulation.

Implements public-key encryption of inference outputs using
X25519 key exchange + ChaCha20-Poly1305 AEAD, matching the paper's
ct_i = Enc(pk_u, y_i || bind_i || taskID).
"""

from __future__ import annotations

import struct

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def generate_encryption_keypair() -> tuple[bytes, bytes]:
    """Generate an X25519 keypair for output encryption.

    Returns (public_key_bytes, private_key_bytes).
    """
    sk = X25519PrivateKey.generate()
    pk = sk.public_key()
    return (
        pk.public_bytes_raw(),
        sk.private_bytes_raw(),
    )


def encrypt_output(
    recipient_pk_bytes: bytes,
    output_values: list[float],
    chain_binding: bytes,
    task_id: int,
) -> bytes:
    """Encrypt y_i || bind_i || taskID under the user's public key.

    Uses ephemeral X25519 key exchange + HKDF + ChaCha20-Poly1305.
    Returns: ephemeral_pk (32 bytes) || nonce (12 bytes) || ciphertext.
    """
    # Build plaintext: output floats || chain_binding || taskID
    plaintext = struct.pack(f">{len(output_values)}f", *output_values)
    plaintext += chain_binding
    plaintext += struct.pack(">I", task_id)

    # Ephemeral key exchange
    ephemeral_sk = X25519PrivateKey.generate()
    ephemeral_pk = ephemeral_sk.public_key()
    recipient_pk = X25519PublicKey.from_public_bytes(recipient_pk_bytes)

    shared_secret = ephemeral_sk.exchange(recipient_pk)

    # Derive symmetric key via HKDF
    symmetric_key = HKDF(
        algorithm=SHA256(),
        length=32,
        salt=None,
        info=b"poml-output-encryption",
    ).derive(shared_secret)

    # Encrypt with ChaCha20-Poly1305
    import os
    nonce = os.urandom(12)
    aead = ChaCha20Poly1305(symmetric_key)
    ciphertext = aead.encrypt(nonce, plaintext, None)

    return ephemeral_pk.public_bytes_raw() + nonce + ciphertext


def decrypt_output(
    recipient_sk_bytes: bytes,
    encrypted_data: bytes,
    num_output_floats: int,
) -> tuple[list[float], bytes, int]:
    """Decrypt an encrypted output.

    Returns (output_values, chain_binding, task_id).
    """
    # Parse: ephemeral_pk (32) || nonce (12) || ciphertext
    ephemeral_pk_bytes = encrypted_data[:32]
    nonce = encrypted_data[32:44]
    ciphertext = encrypted_data[44:]

    # Key exchange
    recipient_sk = X25519PrivateKey.from_private_bytes(recipient_sk_bytes)
    ephemeral_pk = X25519PublicKey.from_public_bytes(ephemeral_pk_bytes)

    shared_secret = recipient_sk.exchange(ephemeral_pk)

    symmetric_key = HKDF(
        algorithm=SHA256(),
        length=32,
        salt=None,
        info=b"poml-output-encryption",
    ).derive(shared_secret)

    aead = ChaCha20Poly1305(symmetric_key)
    plaintext = aead.decrypt(nonce, ciphertext, None)

    # Parse plaintext: floats || binding (32 bytes) || taskID (4 bytes)
    float_size = num_output_floats * 4
    output_bytes = plaintext[:float_size]
    chain_binding = plaintext[float_size : float_size + 32]
    task_id_bytes = plaintext[float_size + 32 : float_size + 36]

    output_values = list(struct.unpack(f">{num_output_floats}f", output_bytes))
    task_id = struct.unpack(">I", task_id_bytes)[0]

    return output_values, chain_binding, task_id
