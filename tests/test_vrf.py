"""Tests for the Ed25519 sign-then-hash VRF used to derive inference noise."""

from poml_sim.vrf import (
    generate_vrf_keypair,
    vrf_eval,
    vrf_noise_schedule,
    vrf_verify,
)


class TestKeygen:
    def test_keypair_shape(self):
        vk, sk = generate_vrf_keypair()
        assert len(vk) == 32
        assert len(sk) == 32

    def test_keypairs_unique(self):
        vk1, sk1 = generate_vrf_keypair()
        vk2, sk2 = generate_vrf_keypair()
        assert vk1 != vk2
        assert sk1 != sk2


class TestEvalVerify:
    def test_eval_deterministic(self):
        _, sk = generate_vrf_keypair()
        y1, pi1 = vrf_eval(sk, b"some_input")
        y2, pi2 = vrf_eval(sk, b"some_input")
        assert y1 == y2
        assert pi1 == pi2

    def test_verify_accepts_valid(self):
        vk, sk = generate_vrf_keypair()
        y, pi = vrf_eval(sk, b"msg")
        assert vrf_verify(vk, b"msg", y, pi)

    def test_verify_rejects_wrong_message(self):
        vk, sk = generate_vrf_keypair()
        y, pi = vrf_eval(sk, b"msg")
        assert not vrf_verify(vk, b"other", y, pi)

    def test_verify_rejects_wrong_vk(self):
        _, sk = generate_vrf_keypair()
        other_vk, _ = generate_vrf_keypair()
        y, pi = vrf_eval(sk, b"msg")
        assert not vrf_verify(other_vk, b"msg", y, pi)

    def test_verify_rejects_tampered_output(self):
        vk, sk = generate_vrf_keypair()
        y, pi = vrf_eval(sk, b"msg")
        tampered = bytes([y[0] ^ 0xFF]) + y[1:]
        assert not vrf_verify(vk, b"msg", tampered, pi)

    def test_verify_rejects_tampered_proof(self):
        vk, sk = generate_vrf_keypair()
        y, pi = vrf_eval(sk, b"msg")
        tampered = bytes([pi[0] ^ 0xFF]) + pi[1:]
        assert not vrf_verify(vk, b"msg", y, tampered)


class TestNoiseSchedule:
    def test_shape(self):
        _, sk = generate_vrf_keypair()
        U, transcript = vrf_noise_schedule(sk, b"\x00" * 32, T=4, spatial=64)
        assert len(U) == 4
        assert all(len(row) == 64 for row in U)
        assert len(transcript) == 4

    def test_range(self):
        _, sk = generate_vrf_keypair()
        U, _ = vrf_noise_schedule(sk, b"\x00" * 32, T=3, spatial=64)
        for row in U:
            for v in row:
                assert -1.0 <= v <= 1.0

    def test_deterministic(self):
        _, sk = generate_vrf_keypair()
        U1, t1 = vrf_noise_schedule(sk, b"abc" + b"\x00" * 29, T=2, spatial=32)
        U2, t2 = vrf_noise_schedule(sk, b"abc" + b"\x00" * 29, T=2, spatial=32)
        assert U1 == U2
        assert t1 == t2

    def test_per_step_transcript_verifies(self):
        """Each (y_t, pi_t) from the schedule must verify against the vk."""
        import struct
        vk, sk = generate_vrf_keypair()
        r_i = b"\xab" * 32
        _, transcript = vrf_noise_schedule(sk, r_i, T=3, spatial=16)
        for t, (y_t, pi_t) in enumerate(transcript, start=1):
            assert vrf_verify(vk, r_i + struct.pack(">I", t), y_t, pi_t)
