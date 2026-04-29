"""Simulated randomized public-key encryption for PoML inference outputs.

The paper's revised protocol requires Enc to be randomized (IND-CPA), with
encryption randomness r_enc derived from a dedicated encryption VRF:
  r_enc, pi_enc = VRF.Eval(sk_VRF_enc, seed_i || taskID)
The ciphertext ct_i = Enc(pk_u, y_i || taskID; r_enc) binds the output to
the VRF-derived randomness, making it unique and verifiable.

Implementation: textbook RSA-2048 with r_enc embedded in the plaintext.
Since textbook RSA has no internal randomness parameter, we include the
32-byte VRF output r_enc as a prefix in the plaintext:
  plaintext = r_enc (32B) || y_i (floats) || taskID (4B)
This makes the ciphertext uniquely determined by (pk_u, r_enc, y_i, taskID),
and r_enc is recoverable on decryption for ZK statement verification.

Note: textbook RSA is NOT IND-CPA secure in the standard sense (no semantic
security) and is not suitable for production use. The r_enc prefix gives
the ciphertext dependence on the VRF randomness as required by the paper.
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
    enc_randomness: bytes,
    task_id: int,
) -> bytes:
    # Layout: r_enc (32B) || y (floats) || taskID (4B)
    plaintext = enc_randomness
    plaintext += struct.pack(f">{len(output_values)}f", *output_values)
    plaintext += struct.pack(">I", task_id)
    return plaintext


def _parse_plaintext(
    plaintext: bytes, num_output_floats: int
) -> tuple[list[float], bytes, int]:
    # r_enc is the first 32 bytes.
    enc_randomness = plaintext[:32]
    float_size = num_output_floats * 4
    output_bytes = plaintext[32 : 32 + float_size]
    task_id_bytes = plaintext[32 + float_size : 32 + float_size + 4]
    output_values = list(struct.unpack(f">{num_output_floats}f", output_bytes))
    task_id = struct.unpack(">I", task_id_bytes)[0]
    return output_values, enc_randomness, task_id


def encrypt_output(
    recipient_pk_bytes: bytes,
    output_values: list[float],
    enc_randomness: bytes,
    task_id: int,
) -> bytes:
    """Encrypt y || taskID under pk_u with VRF-derived randomness r_enc.

    Plaintext layout: r_enc (32B) || y (floats) || taskID (4B).
    The r_enc value is included so decryptors can verify the VRF relationship.
    """
    n, e = _load_public_numbers(recipient_pk_bytes)

    plaintext = _serialize_plaintext(output_values, enc_randomness, task_id)
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
    """Inverse of encrypt_output. Returns (values, enc_randomness, task_id)."""
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
