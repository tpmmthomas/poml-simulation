#!/usr/bin/env python3
"""Independently audit a completed live run's proof files, C, bindings and hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from poml_sim.gpt2_work import reference_counts, weighted_cost  # noqa: E402
from poml_sim.lottery import scaled_complexity, evaluate_tickets  # noqa: E402
from poml_sim.protocol_inputs import (  # noqa: E402
    canonical,
    digest,
    frame,
    query_binding,
    public_signing_key,
    experimental_key,
)


def audit_chain(directory, schedule, scale, weights=None):
    """Rebuild every logical attempt's proof/ciphertext prefix independently."""
    summary = json.loads((directory / "summary.json").read_text())
    blocks = json.loads((directory / "blocks.json").read_text())
    attempts = json.loads((directory / "attempts.json").read_text())
    checked = 0
    prior_block = None
    for block in blocks:
        payload = block["block"]
        if prior_block is not None and payload["parent"] != prior_block:
            raise ValueError("successive blocks are not hash-linked")
        prior_block = block["block_hash"]
        if digest(canonical(payload)).hex() != block["block_hash"]:
            raise ValueError("block hash differs from retained block manifest")
        base = digest(
            b"poml-G-v1" + bytes.fromhex(payload["parent"]) + canonical([[], [], []])
        )
        prefixes, previous = {}, {}
        rows = sorted(
            [r for r in attempts if r["height"] == block["height"]],
            key=lambda r: r["attempt_id"],
        )
        for row in rows:
            source_directory = Path(row["proof_directory"])
            replayed = row.get("fresh_proof") is False
            artifact_directory = Path(row.get("attempt_directory", source_directory))
            request_path = (
                artifact_directory / "simulated_request.json"
                if replayed
                else source_directory / "request.json"
            )
            request = json.loads(request_path.read_text())
            n, k = len(request["prompt_tokens"]), len(row["output_tokens"])
            counts = reference_counts(n, k, schedule)["combined"]
            raw = sum(counts.values())
            weighted = weighted_cost(counts, weights) if weights is not None else raw
            if weights is not None and (
                row.get("weighted_complexity") != weighted
                or row.get("complexity_weights_sha256")
                != digest(canonical(weights)).hex()
            ):
                raise ValueError("weighted operation cost or schedule digest mismatch")
            if row["raw_complexity"] != raw or row["complexity"] != scaled_complexity(
                weighted, scale
            ):
                raise ValueError("recorded C differs from scaled appendix reference")
            miner = row["miner"]
            bind = previous.get(miner, base)
            vk = public_signing_key(
                experimental_key(summary["seed"], f"miner:{miner}:sig")
            )
            hu, challenge = query_binding(
                bind, request["prompt_tokens"], row["qid"], vk
            )
            if (bind.hex(), hu.hex(), challenge.hex()) != (
                row["bind"],
                row["h_u"],
                row["r"],
            ):
                raise ValueError("query/miner/preceding-proof binding mismatch")
            proof = (source_directory / "proof.bin").read_bytes()
            if hashlib.sha256(proof).hexdigest() != row["proof_sha256"]:
                raise ValueError("proof file differs from the chained digest")
            previous[miner] = digest(proof)
            ciphertext = (artifact_directory / "ciphertext.bin").read_bytes()
            if digest(ciphertext).hex() != row["ciphertext_sha256"]:
                raise ValueError("ciphertext file digest mismatch")
            prefixes[miner] = prefixes.get(miner, b"") + frame(ciphertext)
            result = evaluate_tickets(
                base + prefixes[miner], row["complexity"], int(row["difficulty"])
            )
            for key in (
                "tickets_evaluated",
                "winning_ticket",
                "winning_hash",
                "prefix_sha256",
            ):
                if result[key] != row[key]:
                    raise ValueError(f"literal lottery mismatch: {key}")
            checked += 1
    return {
        "blocks": len(blocks),
        "verified_attempt_bindings_hashes_and_complexities": checked,
        "replayed_attempts": sum(row.get("fresh_proof") is False for row in attempts),
        "fresh_attempts": sum(row.get("fresh_proof") is True for row in attempts),
        "proof_binding_scope": "source proof digest is chained; replay challenge is audited for current binding, encryption and lottery but is not re-proved",
    }


def main():
    """Audit completed stage data; no inference or proof timing is regenerated."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--schedule", type=Path, default=ROOT / "config/gpt2_reference_schedule.json"
    )
    args = p.parse_args()
    calibration = json.loads((args.input / "preparation/calibration.json").read_text())
    schedule = json.loads(args.schedule.read_text())
    report = {}
    for directory in [
        args.input / "liveness",
        *sorted((args.input / "cherry_pick").glob("*-*")),
    ]:
        if (directory / "COMPLETED").exists():
            report[str(directory.relative_to(args.input))] = audit_chain(
                directory, schedule, calibration["scale"], calibration.get("weights")
            )
    if not report:
        raise ValueError("no completed live chain stages")
    (args.input / "protocol_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
