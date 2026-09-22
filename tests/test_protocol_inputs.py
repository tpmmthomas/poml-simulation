"""Query/miner/proof binding, deterministic noise, and actual response encryption."""

from poml_sim.protocol_inputs import (
    digest,
    experimental_key,
    query_binding,
    public_signing_key,
    gaussian_noise,
    decoding_uniforms,
    encrypt_output,
    decrypt_output,
)
from poml_sim.vrf import vrf_verify


def test_query_binding_changes_with_every_protocol_binding():
    vk = public_signing_key(experimental_key(1, "sig"))
    _, r = query_binding(digest(b"parent"), [1, 2], "qid", vk)
    variants = [
        (digest(b"other"), [1, 2], "qid", vk),
        (digest(b"parent"), [1, 3], "qid", vk),
        (digest(b"parent"), [1, 2], "other", vk),
        (digest(b"parent"), [1, 2], "qid", b"x" * 32),
    ]
    assert all(query_binding(*v)[1] != r for v in variants)


def test_gaussian_and_decoding_randomness_reproduce_and_verify():
    secret = experimental_key(1, "inf")
    r = digest(b"attempt")
    noise, vrfs = gaussian_noise(secret, r, 3, 4, 0.05, 0.1437)
    assert (noise, vrfs) == gaussian_noise(secret, r, 3, 4, 0.05, 0.1437)
    assert len(noise) == 3 and all(len(row) == 768 for row in noise)
    assert len(set(decoding_uniforms(vrfs, 3))) == 4
    assert noise != gaussian_noise(secret, digest(b"other"), 3, 4, 0.05, 0.1437)[0]
    vk = public_signing_key(secret)
    for row in vrfs:
        assert vrf_verify(
            vk,
            r + row["t"].to_bytes(8, "big"),
            bytes.fromhex(row["z"]),
            bytes.fromhex(row["proof"]),
        )


def test_encryption_binds_proved_payload_and_vrf_randomness():
    user = experimental_key(3, "user")
    payload = b"tokens and logits\x00" * 100
    ct = encrypt_output(payload, user, digest(b"encryption"))
    assert decrypt_output(ct, user) == payload
    assert ct == encrypt_output(payload, user, digest(b"encryption"))
    assert ct != encrypt_output(payload, user, digest(b"different challenge"))
