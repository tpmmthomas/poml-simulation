"""The hashing baseline returns real threshold-valid double-SHA-256 headers."""

import hashlib
from poml_sim.pow import run_pow


def test_actual_pow_races_produce_verifiable_parent_bound_headers():
    result = run_pow(blocks=2, miners=2, target=0.02, calibration_seconds=0.05)
    parent = bytes(32).hex()
    assert result["hashes_per_second"] > 0
    for row in result["records"]:
        header = bytes.fromhex(row["header"])
        digest = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        assert len(header) == 80 and digest.hex() == row["winning_digest"]
        assert int.from_bytes(digest, "little") < int(result["difficulty"])
        assert row["parent"] == parent
        assert row["block_time"] > 0 and row["hashes"] > 0
        parent = row["winning_digest"]
