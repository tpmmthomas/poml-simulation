"""Campaign boundary coverage and failure handling for structural ledgers."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_deepprove_work import load_ledger, validation_pairs


def test_campaign_covers_every_proof_profile_and_padding_neighbor():
    pairs = validation_pairs()
    totals = {n + k for n, k in pairs}
    assert {3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64} <= totals
    assert (63, 1) in pairs and (2, 62) in pairs
    assert len(pairs) > len(set(pairs))


def test_incomplete_ledger_is_rejected(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"verified": True}) + "\n" + '{"verif')
    with pytest.raises(ValueError, match="Incomplete"):
        load_ledger(ledger)


def test_failed_proof_cannot_be_loaded_as_verified(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"verified": False}) + "\n")
    with pytest.raises(ValueError, match="Unverified"):
        load_ledger(ledger)
