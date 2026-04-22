"""Deterministic public-key encryption for PoML inference outputs.

The paper's revised protocol requires Enc to be deterministic: for a fixed
user public key and plaintext (y_i || bind_i || taskID), the ciphertext
must be unique so that the lottery hash H(G(s,x), (ct_1, ..., ct_i))
is itself uniquely determined by the inference outputs.

Implementation: textbook RSA-2048. We generate a standard RSA-2048 key
pair via `cryptography`, then perform encryption/decryption as
`c = m^e mod n` / `m = c^d mod n` using Python's built-in `pow()`. The
plaintext is chunked into 245-byte blocks (safely below the 256-byte
modulus size), with a 4-byte big-endian length prefix so the final
chunk's zero padding can be stripped on decryption.

Security note: textbook RSA is one-way under the RSA assumption but is
*not* IND-CPA secure (malleable, deterministic). That is exactly what we
want for simulation purposes here — IND-CPA would preclude determinism.
Not suitable for production use.
"""

from __future__ import annotations

import struct

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

_RSA_KEY_BITS = 2048
_RSA_MODULUS_BYTES = _RSA_KEY_BITS // 8  # 256
# Leave at least one byte of headroom so every chunk as a big-endian
# integer is strictly less than n. 245 is generous and keeps arithmetic
# friendly.
_RSA_PLAINTEXT_BLOCK = 245
_LENGTH_PREFIX = 4


def generate_encryption_keypair() -> tuple[bytes, bytes]:
    """Generate an RSA-2048 keypair. Returns (pk_bytes, sk_bytes) as DER."""
    sk_obj = rsa.generate_private_key(public_exponent=65537, key_size=_RSA_KEY_BITS)
    pk_bytes = sk_obj.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    sk_bytes = sk_obj.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pk_bytes, sk_bytes


def _load_public_numbers(pk_bytes: bytes) -> tuple[int, int]:
    pk = serialization.load_der_public_key(pk_bytes)
    nums = pk.public_numbers()
    return nums.n, nums.e


def _load_private_numbers(sk_bytes: bytes) -> tuple[int, int]:
    sk = serialization.load_der_private_key(sk_bytes, password=None)
    nums = sk.private_numbers()
    return nums.public_numbers.n, nums.d


def _serialize_plaintext(
    output_values: list[float],
    chain_binding: bytes,
    task_id: int,
) -> bytes:
    plaintext = struct.pack(f">{len(output_values)}f", *output_values)
    plaintext += chain_binding
    plaintext += struct.pack(">I", task_id)
    return plaintext


def _parse_plaintext(
    plaintext: bytes, num_output_floats: int
) -> tuple[list[float], bytes, int]:
    float_size = num_output_floats * 4
    output_bytes = plaintext[:float_size]
    chain_binding = plaintext[float_size : float_size + 32]
    task_id_bytes = plaintext[float_size + 32 : float_size + 36]
    output_values = list(struct.unpack(f">{num_output_floats}f", output_bytes))
    task_id = struct.unpack(">I", task_id_bytes)[0]
    return output_values, chain_binding, task_id


def encrypt_output(
    recipient_pk_bytes: bytes,
    output_values: list[float],
    chain_binding: bytes,
    task_id: int,
) -> bytes:
    """Deterministically encrypt y || bind || taskID under an RSA pk."""
    n, e = _load_public_numbers(recipient_pk_bytes)

    plaintext = _serialize_plaintext(output_values, chain_binding, task_id)
    # Length prefix lets the decryptor strip zero-padding on the final
    # chunk without ambiguity.
    framed = struct.pack(">I", len(plaintext)) + plaintext

    # Zero-pad to a whole number of plaintext blocks.
    pad = (-len(framed)) % _RSA_PLAINTEXT_BLOCK
    framed += b"\x00" * pad

    parts: list[bytes] = []
    for off in range(0, len(framed), _RSA_PLAINTEXT_BLOCK):
        m_int = int.from_bytes(framed[off : off + _RSA_PLAINTEXT_BLOCK], "big")
        c_int = pow(m_int, e, n)
        parts.append(c_int.to_bytes(_RSA_MODULUS_BYTES, "big"))
    return b"".join(parts)


def decrypt_output(
    recipient_sk_bytes: bytes,
    encrypted_data: bytes,
    num_output_floats: int,
) -> tuple[list[float], bytes, int]:
    """Inverse of encrypt_output. Returns (values, chain_binding, task_id)."""
    n, d = _load_private_numbers(recipient_sk_bytes)

    if len(encrypted_data) == 0 or len(encrypted_data) % _RSA_MODULUS_BYTES != 0:
        raise ValueError(
            f"ciphertext length {len(encrypted_data)} is not a multiple of "
            f"{_RSA_MODULUS_BYTES}"
        )

    framed_parts: list[bytes] = []
    for off in range(0, len(encrypted_data), _RSA_MODULUS_BYTES):
        c_int = int.from_bytes(encrypted_data[off : off + _RSA_MODULUS_BYTES], "big")
        m_int = pow(c_int, d, n)
        framed_parts.append(m_int.to_bytes(_RSA_PLAINTEXT_BLOCK, "big"))
    framed = b"".join(framed_parts)

    (plaintext_len,) = struct.unpack(">I", framed[:_LENGTH_PREFIX])
    plaintext = framed[_LENGTH_PREFIX : _LENGTH_PREFIX + plaintext_len]
    return _parse_plaintext(plaintext, num_output_floats)
