"""Regression coverage for exact-prompt DeepProve timing association."""

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.build_real_llm_trace_bank import _parse_proofs, _boundary_stop_tokens  # noqa: E402


def _proof_files(directory):
    trials = [
        dict(prompt_length=2, output_length=1, prompt_tokens=tokens)
        for tokens in ([1, 2], [3, 4])
    ]
    ledger = [
        dict(
            verified=True,
            meta=dict(
                backend="cuda",
                trial=i,
                n=2,
                k=1,
                tokens=t["prompt_tokens"],
                setup_max=3,
            ),
        )
        for i, t in enumerate(trials, 1)
    ]
    (directory / "deepprove_ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ledger)
    )
    with (directory / "deepprove_timings.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "min_user_len",
                "max_context",
                "inference_time",
                "prove_full",
                "verify_full",
                "proof_size",
            ],
        )
        writer.writeheader()
        for duration in (100, 200):
            writer.writerow(
                dict(
                    min_user_len=2,
                    max_context=3,
                    inference_time=10,
                    prove_full=duration,
                    verify_full=1,
                    proof_size=50,
                )
            )
    return trials


def test_same_shape_different_prompts_keep_separate_measured_times(tmp_path):
    trials = _proof_files(tmp_path)
    measurements = _parse_proofs(tmp_path, trials)
    assert [row["duration"] for row in measurements] == [0.11, 0.21]
    assert [row["proof_trial"] for row in measurements] == [1, 2]


def test_wrong_prompt_cannot_supply_a_matching_shape_timing(tmp_path):
    trials = _proof_files(tmp_path)
    trials[1]["prompt_tokens"] = [1, 2]
    with pytest.raises(ValueError, match="mismatch"):
        _parse_proofs(tmp_path, trials)


def test_sentence_stop_rule_is_public_and_prompt_independent():
    class Tokenizer:
        def get_vocab(self):
            return {str(i): i for i in range(6)}

        def decode(self, tokens):
            return ["hello", ".", "?", "word!", "\n", "word"][tokens[0]]

    assert _boundary_stop_tokens(Tokenizer()) == [1, 2, 3, 4]
