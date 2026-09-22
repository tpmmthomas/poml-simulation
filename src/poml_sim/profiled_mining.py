"""Replay genuine measured pairs while computing bindings, encryption and hashes.

An archived proof attests only its original challenge. Reusing its output and
duration under a new simulated challenge is an explicit empirical approximation.
"""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from .gpt2_work import reference_counts, weighted_cost
from .live_mining import append_json
from .lottery import evaluate_tickets, scaled_complexity
from .protocol_inputs import (
    canonical,
    digest,
    frame,
    experimental_key,
    public_signing_key,
    query_binding,
    gaussian_noise,
    decoding_uniforms,
    encrypt_output,
    decrypt_output,
)
from .vrf import vrf_eval


def file_digest(path: Path) -> str:
    """Hash a retained artifact without loading an arbitrarily large file."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class MeasuredBank:
    """Validated per-prompt outcomes with deterministic, private per-query assignment."""

    def __init__(
        self, pool: list[dict], records: list[dict], schedule: dict, scale: dict
    ):
        self.records = {q["query_id"]: [] for q in pool}
        prompts = {q["query_id"]: q for q in pool}
        identities = set()
        for row in records:
            q = prompts[row["query_id"]]
            identity = (row["query_id"], row["replicate"])
            if identity in identities:
                raise ValueError("duplicate measurement replicate")
            identities.add(identity)
            if not all(
                row.get(k) is True
                for k in ("verified", "fresh_inference", "fresh_proof")
            ):
                raise ValueError("only fresh verified measurements may enter the bank")
            if not math.isfinite(row["duration"]) or row["duration"] <= 0:
                raise ValueError("measurement duration must be positive and finite")
            directory = Path(row["proof_directory"])
            request = json.loads((directory / "request.json").read_text())
            if (
                request["prompt_tokens"] != q["prompt_tokens"]
                or request["max_output"] != q["max_output_length"]
                or row["prompt_sha256"] != q["prompt_sha256"]
            ):
                raise ValueError("measurement does not match the registered prompt/cap")
            k = row["output_length"]
            if len(row["output_tokens"]) != k or not 1 <= k <= q["max_output_length"]:
                raise ValueError("measurement has an invalid realised output length")
            raw = sum(
                reference_counts(q["prompt_length"], k, schedule)["combined"].values()
            )
            if row["raw_complexity"] != raw or row["complexity"] != scaled_complexity(
                raw, scale
            ):
                raise ValueError(
                    "measurement complexity differs from the frozen schedule"
                )
            if file_digest(directory / "proof.bin") != row["proof_sha256"]:
                raise ValueError("archived proof digest mismatch")
            logits = directory / "logits.i64"
            if logits.stat().st_size != k * 50257 * 8:
                raise ValueError("archived logits have an invalid size")
            if row.get("logits_sha256") and file_digest(logits) != row["logits_sha256"]:
                raise ValueError("archived logits digest mismatch")
            # Verify the saved output bytes against the original end-to-end record.
            if digest(self.plaintext(row, row["qid"])).hex() != row["output_sha256"]:
                raise ValueError("archived tokens/logits digest mismatch")
            self.records[row["query_id"]].append(row)
        if any(not rows for rows in self.records.values()):
            raise ValueError("each prompt needs at least one verified measurement")
        for rows in self.records.values():
            rows.sort(key=lambda r: r["replicate"])
        self.sha256 = digest(canonical(records)).hex()

    @staticmethod
    @lru_cache(maxsize=16)
    def logits(directory: str) -> bytes:
        """Cache a bounded number of immutable public-logit artifacts."""
        return (Path(directory) / "logits.i64").read_bytes()

    def plaintext(self, row: dict, qid: str) -> bytes:
        """Construct the actual response payload with the current query identifier."""
        return frame(
            canonical(
                {
                    "qid": qid,
                    "tokens": row["output_tokens"],
                    "logits_scale": row["logits_scale"],
                    "logits_shape": row["logits_shape"],
                }
            )
        ) + self.logits(str(row["proof_directory"]))

    def sample(self, query: dict, seed: int) -> dict:
        """Assign one recorded pair to a qid, identically for every miner.

        Selection policies never receive the sampled record or its realised K.
        Independent qids may reuse a template; collision identity remains qid.
        """
        key = canonical(
            ["poml-measured-query-v1", seed, query["query_id"], query["request_id"]]
        )
        rows = self.records[query["query_id"]]
        return rows[int.from_bytes(digest(key), "big") % len(rows)]


class ProfiledWork:
    """Compute auxiliary protocol work now, using a fixed measured service time."""

    def __init__(
        self, bank, ready, schedule, scale, directory, seed, alpha, weights=None
    ):
        self.bank, self.schedule, self.scale = bank, schedule, scale
        self.weights = weights
        self.directory, self.seed, self.alpha = Path(directory), seed, alpha
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prover = SimpleNamespace(
            ready={**ready, "execution": "profiled", "bank_sha256": bank.sha256}
        )
        if weights is not None:
            # Binding the entire frozen schedule into genesis prevents changing
            # operation weights between blocks while retaining the same chain.
            self.prover.ready["complexity_weights_sha256"] = digest(
                canonical(weights)
            ).hex()
        self.executions = 0

    def execute(self, query, miner, bind, lottery_base, ciphertext_prefix, difficulty):
        """Schedule a recorded pair; never call a prover or reuse a saved lottery."""
        start = time.perf_counter()
        index = self.executions
        self.executions += 1
        location = self.directory / f"attempt-{index:07d}"
        location.mkdir(exist_ok=False)
        source = self.bank.sample(query, self.seed)
        qid = query["request_id"]
        identity = experimental_key(self.seed, f"miner:{miner}:sig")
        inf = experimental_key(self.seed, f"miner:{miner}:inf")
        enc = experimental_key(self.seed, f"miner:{miner}:enc")
        hu, r = query_binding(
            bind, query["prompt_tokens"], qid, public_signing_key(identity)
        )
        noise, transcript = gaussian_noise(
            inf,
            r,
            query["prompt_length"],
            query["max_output_length"],
            self.alpha,
            self.prover.ready["embedding_std"],
        )
        (location / "simulated_request.json").write_text(
            json.dumps(
                {
                    "prompt_tokens": query["prompt_tokens"],
                    "noise": noise,
                    "uniforms": decoding_uniforms(transcript, query["prompt_length"]),
                    "max_output": query["max_output_length"],
                    "notice": "New challenge is not applied to the archived inference/proof.",
                }
            )
            + "\n"
        )
        counts = reference_counts(
            query["prompt_length"], source["output_length"], self.schedule
        )
        raw = sum(counts["combined"].values())
        weighted = (
            weighted_cost(counts["combined"], self.weights)
            if self.weights is not None
            else raw
        )
        complexity = scaled_complexity(weighted, self.scale)
        z_enc, proof_enc = vrf_eval(enc, r + frame(qid.encode()))
        plaintext = self.bank.plaintext(source, qid)
        ciphertext = encrypt_output(
            plaintext, experimental_key(self.seed, f"user:{qid}"), z_enc
        )
        if (
            decrypt_output(ciphertext, experimental_key(self.seed, f"user:{qid}"))
            != plaintext
        ):
            raise RuntimeError("replayed response encryption did not round-trip")
        (location / "ciphertext.bin").write_bytes(ciphertext)
        prefix = ciphertext_prefix + frame(ciphertext)
        lottery = evaluate_tickets(lottery_base + prefix, complexity, difficulty)
        row = {
            **lottery,
            "attempt_id": index,
            "miner": miner,
            "query_id": query["query_id"],
            "qid": qid,
            "prompt_length": query["prompt_length"],
            "prompt_sha256": query["prompt_sha256"],
            "output_length": source["output_length"],
            "output_tokens": source["output_tokens"],
            "stop_reason": source["stop_reason"],
            "raw_complexity": raw,
            "complexity": complexity,
            "weighted_complexity": weighted,
            "complexity_weights_sha256": self.prover.ready.get(
                "complexity_weights_sha256"
            ),
            "reference_counts": counts,
            "duration": source.get(
                "inference_proof_seconds",
                source["inference_seconds"] + source["proof_seconds"],
            ),
            "profiled_inference_seconds": source["inference_seconds"],
            "profiled_proof_seconds": source["proof_seconds"],
            "profiled_service_seconds": source["duration"],
            "inference_seconds": 0.0,
            "proof_seconds": 0.0,
            "verification_seconds": 0.0,
            "host_seconds": time.perf_counter() - start,
            "bind": bind.hex(),
            "h_u": hu.hex(),
            "r": r.hex(),
            "inference_vrf": transcript,
            "encryption_vrf": {"z": z_enc.hex(), "proof": proof_enc.hex()},
            "ciphertext_sha256": digest(ciphertext).hex(),
            "output_sha256": digest(plaintext).hex(),
            "proof_sha256": source["proof_sha256"],
            "source_output_sha256": source["output_sha256"],
            "source_logits_sha256": source.get("logits_sha256"),
            "proof_directory": source["proof_directory"],
            "attempt_directory": str(location),
            "source_replicate": source["replicate"],
            "source_challenge": source["r"],
            "source_verified": True,
            "verified": False,
            "proof_bound_to_current_challenge": False,
            "fresh_inference": False,
            "fresh_proof": False,
            "lottery": "literal_sha256",
            "difficulty": str(difficulty),
            "bank_sha256": self.bank.sha256,
        }
        (location / "attempt.json").write_text(json.dumps(row, indent=2) + "\n")
        append_json(self.directory / "executed.jsonl", row)
        return {**row, "ciphertext_prefix": prefix}
