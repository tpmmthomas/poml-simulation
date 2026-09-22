"""Event tests use a labelled test double; actual campaigns never accept trace banks."""

from types import SimpleNamespace
import random

from poml_sim.live_mining import LiveChain, select_query


def pool():
    return [
        {
            "query_id": f"q{i}",
            "request_id": f"q{i}",
            "prompt_length": n,
            "profile_mean_output_length": k,
        }
        for i, (n, k) in enumerate([(8, 3), (16, 1), (4, 5), (32, 2)])
    ]


def test_policies_use_frozen_attributes_and_do_not_repeat_queries():
    rows = pool()
    assert (
        select_query(rows, set(), "profiled-short", random.Random(1))["query_id"]
        == "q1"
    )
    assert (
        select_query(rows, set(), "profiled-long", random.Random(1))["query_id"] == "q2"
    )
    assert (
        select_query(rows, set(), "shortest-prompt", random.Random(1))["query_id"]
        == "q2"
    )
    assert (
        select_query(rows, {"q1"}, "profiled-short", random.Random(1))["query_id"]
        == "q3"
    )
    assert (
        select_query(rows, {q["request_id"] for q in rows}, "uniform", random.Random(1))
        is None
    )


class TestOnlyWork:
    """Controlled outcomes to exercise fresh scheduling and proof-chain bindings."""

    __test__ = False

    def __init__(self):
        self.prover = SimpleNamespace(ready={"test_double": True})
        self.scale = {"numerator": 1, "denominator": 1}
        self.calls = []

    def execute(self, q, miner, bind, base, prefix, difficulty):
        i = len(self.calls)
        self.calls.append((q, miner, bind, base, prefix, difficulty))
        # Miner 0 completes two linked attempts and wins; miner 1 is canceled.
        return {
            "attempt_id": i,
            "miner": miner,
            "qid": q["request_id"],
            "complexity": 1000,
            "duration": 1 if miner == 0 else 10,
            "host_seconds": 1,
            "winning_ticket": 1 if i >= 2 else None,
            "proof_sha256": f"{i + 1:064x}",
            "ciphertext_sha256": f"{i + 10:064x}",
            "ciphertext_prefix": prefix + b"real-in-tests-only",
        }


def test_every_scheduled_attempt_is_executed_and_canceled_work_is_separate():
    work = TestOnlyWork()
    chain = LiveChain(work, pool(), 2, 1, 41)
    first_parent = chain.parent
    block, events = chain.block()
    assert len(work.calls) == 3
    assert work.calls[2][2] == bytes.fromhex("01".zfill(64))
    assert work.calls[2][4] == b"real-in-tests-only"
    assert block["completed_attempts"] == 2 and block["canceled_attempts"] == 1
    assert block["total_completed_complexity"] == 2000
    assert block["winning_complexity"] == 2000
    assert block["collision_complexity"] == 0
    assert len(block["block"]["proof_chain"]) == 2
    assert chain.parent != first_parent
    assert sum(e["logical_status"] == "canceled" for e in events) == 1
    second, _ = chain.block()
    assert second["height"] == 2
    assert second["block"]["parent"] == block["block_hash"]


def test_winning_prefix_receives_credit_before_duplicate_responses(monkeypatch):
    import poml_sim.live_mining as mining

    def first_available(pool, seen, policy, rng):
        return next(q for q in pool if q["request_id"] not in seen)

    monkeypatch.setattr(mining, "select_query", first_available)

    class CollisionWork(TestOnlyWork):
        def execute(self, *args):
            result = super().execute(*args)
            result["duration"] = [1, 1.5, 1, 10][result["attempt_id"]]
            result["winning_ticket"] = 1 if result["attempt_id"] == 2 else None
            return result

    chain = LiveChain(CollisionWork(), pool(), 2, 1, 41)
    block, events = chain.block()
    assert block["completed_attempts"] == 3
    assert block["winning_complexity"] == 2000
    assert block["collision_complexity"] == 1000
    assert block["response_complexity"] == 0
    assert block["wasted_work_pct"] == 100 / 3
    assert [e["disposition"] for e in events].count("collision") == 1
