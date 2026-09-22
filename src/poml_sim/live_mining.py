"""Protocol-bound fresh work and zero-delay event simulation for live experiments."""

import heapq
import hashlib
from operator import itemgetter
import json
from pathlib import Path
import random
import shutil
import time

from .gpt2_work import reference_counts
from .lottery import evaluate_tickets, scaled_complexity, LIMIT
from .protocol_inputs import (
    canonical,
    digest,
    frame,
    experimental_key,
    public_signing_key,
    query_binding,
    gaussian_noise,
    encrypt_output,
    decrypt_output,
    decoding_uniforms,
)
from .vrf import vrf_eval


def append_json(path: Path, value: dict) -> None:
    """Persist a completed record immediately for auditing interrupted campaigns."""
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


class FreshWork:
    """Create each actual inference/proof from the current query and miner binding."""

    def __init__(
        self,
        prover,
        schedule: dict,
        scale: dict,
        directory: Path,
        seed: int,
        alpha: float,
    ):
        self.prover, self.schedule, self.scale = prover, schedule, scale
        self.directory, self.seed, self.alpha = directory, seed, alpha
        directory.mkdir(parents=True, exist_ok=True)
        self.executions = 0

    def execute(
        self,
        query: dict,
        miner: int,
        bind: bytes,
        lottery_base: bytes,
        ciphertext_prefix: bytes,
        difficulty: int,
    ) -> dict:
        """Never substitute a bank trace, cached output, or a cached proof."""
        if shutil.disk_usage(self.directory).free < 2 * 1024**3:
            raise RuntimeError(
                "less than 2 GiB free: free disk space before resuming this campaign"
            )
        index = self.executions
        self.executions += 1
        location = self.directory / f"attempt-{index:07d}"
        start = time.perf_counter()
        identity = experimental_key(self.seed, f"miner:{miner}:sig")
        inf = experimental_key(self.seed, f"miner:{miner}:inf")
        enc = experimental_key(self.seed, f"miner:{miner}:enc")
        qid = query["request_id"]
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
        result = self.prover.run(
            {
                "request_id": f"attempt-{index:07d}",
                "prompt_tokens": query["prompt_tokens"],
                "noise": noise,
                "uniforms": decoding_uniforms(transcript, query["prompt_length"]),
                "max_output": query["max_output_length"],
                "mode": "prove",
                "output_directory": str(location.resolve()),
            }
        )
        counts = reference_counts(
            query["prompt_length"], result["output_length"], self.schedule
        )
        raw = sum(counts["combined"].values())
        complexity = scaled_complexity(raw, self.scale)
        z_enc, proof_enc = vrf_eval(enc, r + frame(qid.encode()))
        output = (
            frame(
                canonical(
                    {
                        "qid": qid,
                        "tokens": result["output_tokens"],
                        "logits_scale": result["logits_scale"],
                        "logits_shape": result["logits_shape"],
                    }
                )
            )
            + (location / "logits.i64").read_bytes()
        )
        user = experimental_key(self.seed, f"user:{qid}")
        ciphertext = encrypt_output(output, user, z_enc)
        if decrypt_output(ciphertext, user) != output:
            raise RuntimeError(
                "encrypted response differs from proved inference output"
            )
        (location / "ciphertext.bin").write_bytes(ciphertext)
        prefix = ciphertext_prefix + frame(ciphertext)
        lottery = evaluate_tickets(lottery_base + prefix, complexity, difficulty)
        duration = time.perf_counter() - start - result["verification_seconds"]
        row = {
            **result,
            **lottery,
            "attempt_id": index,
            "miner": miner,
            "query_id": query["query_id"],
            "qid": qid,
            "prompt_length": query["prompt_length"],
            "prompt_sha256": query["prompt_sha256"],
            "raw_complexity": raw,
            "rounding_error_tickets": complexity
            - raw * self.scale["numerator"] / self.scale["denominator"],
            "complexity": complexity,
            "reference_counts": counts,
            "duration": duration,
            "inference_proof_seconds": result["inference_seconds"] + result["proof_seconds"],
            "host_seconds": time.perf_counter() - start,
            "miner_public_keys": {
                "identity": public_signing_key(identity).hex(),
                "inference_vrf": public_signing_key(inf).hex(),
                "encryption_vrf": public_signing_key(enc).hex(),
            },
            "output_sha256": digest(output).hex(),
            "logits_sha256": hashlib.sha256(
                (location / "logits.i64").read_bytes()
            ).hexdigest(),
            "bind": bind.hex(),
            "h_u": hu.hex(),
            "r": r.hex(),
            "inference_vrf": transcript,
            "encryption_vrf": {"z": z_enc.hex(), "proof": proof_enc.hex()},
            "ciphertext_sha256": digest(ciphertext).hex(),
            "proof_directory": str(location),
            "fresh_inference": True,
            "fresh_proof": True,
            "difficulty": str(difficulty),
            "lottery": "literal_sha256",
        }
        (location / "attempt.json").write_text(json.dumps(row, indent=2) + "\n")
        append_json(self.directory / "executed.jsonl", row)
        # Binary prefixes are runtime state only; the ciphertext files reproduce them.
        return {**row, "ciphertext_prefix": prefix}


def select_query(
    pool: list[dict], seen: set[str], policy: str, rng: random.Random
) -> dict | None:
    """Select without replacement using only frozen, publicly profiled attributes."""
    eligible = [q for q in pool if q["request_id"] not in seen]
    if not eligible:
        return None
    if policy == "uniform":
        return rng.choice(eligible)
    if policy in {"profiled-short", "profiled-long"}:
        key = itemgetter("profile_mean_output_length")
        value = (min if policy == "profiled-short" else max)(map(key, eligible))
    elif policy == "shortest-prompt":
        key = itemgetter("prompt_length")
        value = min(map(key, eligible))
    else:
        raise ValueError(f"unknown selection policy {policy}")
    return rng.choice([q for q in eligible if key(q) == value])


class LiveChain:
    """Independent virtual miners, with every scheduled attempt physically executed.

    The host serializes GPU jobs. Their measured service times define virtual
    completion events; already computed in-flight jobs can be logically canceled.
    These jobs remain in the audit and host-cost totals, never in completed work.
    """

    def __init__(
        self,
        work: FreshWork,
        pool: list[dict],
        miners: int,
        difficulty: int,
        seed: int,
        policy: str = "uniform",
    ):
        if miners < 1 or not pool or not 0 < difficulty <= LIMIT:
            raise ValueError("invalid live chain configuration")
        self.work, self.templates, self.miners = work, pool, miners
        self.difficulty, self.policy = difficulty, policy
        self.rng = random.Random(seed)
        self.pending, self.next_qid = [], 0
        self.height = 0
        self.parent = digest(
            canonical(
                {
                    "genesis": True,
                    "seed": seed,
                    "setup": {
                        k: v
                        for k, v in work.prover.ready.items()
                        if k not in {"setup_seconds", "event"}
                    },
                    "scale": work.scale,
                    "difficulty": str(difficulty),
                }
            )
        )
        self.replenish()

    def replenish(self):
        """Model sufficient demand by issuing new benchmark-backed query objects."""
        for q in self.templates:
            self.pending.append({**q, "request_id": f"query-{self.next_qid:09d}"})
            self.next_qid += 1

    def block(self) -> tuple[dict, list[dict]]:
        """Produce one adopted block and return all logical completion/cancellation records."""
        base = digest(b"poml-G-v1" + self.parent + canonical([[], [], []]))
        states = [
            {"seen": set(), "bind": base, "prefix": b"", "chain": []}
            for _ in range(self.miners)
        ]
        events, completed = [], []

        def schedule(miner, start):
            state = states[miner]
            policy = self.policy if miner == 0 else "uniform"
            query = select_query(self.pending, state["seen"], policy, self.rng)
            if query is None:
                self.replenish()
                query = select_query(self.pending, state["seen"], policy, self.rng)
            state["seen"].add(query["request_id"])
            result = self.work.execute(
                query, miner, state["bind"], base, state["prefix"], self.difficulty
            )
            heapq.heappush(events, (start + result["duration"], miner, result))

        for miner in range(self.miners):
            schedule(miner, 0.0)
        while events:
            finish, miner, result = heapq.heappop(events)
            prefix = result.pop("ciphertext_prefix")
            result.update(
                completion_time=finish,
                height=self.height + 1,
                logical_status="completed",
            )
            completed.append(result)
            states[miner]["chain"].append(result)
            if result["winning_ticket"] is not None:
                winner = miner
                break
            states[miner]["prefix"] = prefix
            states[miner]["bind"] = bytes.fromhex(result["proof_sha256"])
            schedule(miner, finish)
        canonical_ids = {r["qid"] for r in states[winner]["chain"]}
        credited = set(canonical_ids)
        collision = response = 0
        for result in completed:
            if result["miner"] == winner:
                result["disposition"] = "winning_prefix"
            elif result["qid"] in credited:
                collision += result["complexity"]
                result["disposition"] = "collision"
            else:
                response += result["complexity"]
                credited.add(result["qid"])
                result["disposition"] = "response_eligible"
        canceled = []
        for end, miner, row in events:
            row.pop("ciphertext_prefix")
            row.update(
                completion_time=end,
                height=self.height + 1,
                logical_status="canceled",
                disposition="unfinished",
            )
            canceled.append(row)
        total = sum(r["complexity"] for r in completed)
        payload = {
            "parent": self.parent.hex(),
            "height": self.height + 1,
            "empty_transactions": [[], [], []],
            "miner": winner,
            "proof_chain": [
                {
                    "qid": r["qid"],
                    "proof_sha256": r["proof_sha256"],
                    "ciphertext_sha256": r["ciphertext_sha256"],
                    "attempt_id": r["attempt_id"],
                }
                for r in states[winner]["chain"]
            ],
            "winning_ticket": completed[-1]["winning_ticket"],
        }
        # Content-addressed proof/ciphertext references fix the block encoding;
        # all referenced bytes are retained alongside the experiment.
        self.parent = digest(canonical(payload))
        self.height += 1
        self.pending = [q for q in self.pending if q["request_id"] not in credited]
        while len(self.pending) < len(self.templates):
            self.replenish()
        summary = {
            "height": self.height,
            "block_hash": self.parent.hex(),
            "block": payload,
            "block_time": finish,
            "winner": winner,
            "adopted": True,
            "completed_attempts": len(completed),
            "canceled_attempts": len(canceled),
            "total_completed_complexity": total,
            "collision_complexity": collision,
            "response_complexity": response,
            "winning_complexity": total - collision - response,
            "wasted_work_pct": 100 * collision / total,
            "host_seconds": sum(r["host_seconds"] for r in completed + canceled),
        }
        return summary, completed + canceled
