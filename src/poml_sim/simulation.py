"""Zero-delay event races with fresh backend work and explicit cancellation.

Physical GPU/prover jobs run serially on the host. Independent virtual miners
advance by measured service durations; physical work computed for canceled
virtual attempts is recorded separately, not charged as a completed pair.
"""

import heapq
import random
from .crypto import sha256
from .lottery import evaluate_lottery
from .system import Block, Transactions, fingerprint


def run_race(system, *, seed, max_events=10000):
    """Mine a block, freezing only response transactions available at its start."""
    rng = random.Random(seed)
    parent = system.tip
    parent_state = system.state
    responses, reserved = [], set()
    for response in system.pending_responses:
        if response.query.qid in reserved:
            continue
        try:
            system.validate_response(response, parent_state, parent_state.height + 1)
        except ValueError:
            continue  # Expired or already settled responses leave the mempool.
        responses.append(response)
        reserved.add(response.query.qid)
    transactions = Transactions(responses=tuple(responses))
    base = fingerprint(parent, transactions)
    candidates = []
    for query in system.pending.values():
        if query.qid in reserved:
            continue
        try:
            system.validate_query(query, parent_state, parent_state.height + 1)
            system.affordable_cap(query)
        except ValueError:
            continue
        candidates.append(query)
    if not candidates:
        raise ValueError("no eligible mining query (response-only blocks are not allowed)")
    states = [{"used": set(), "chain": [], "binding": base} for _ in system.miners]
    queue, completed, executed = [], [], []
    sequence = 0

    def start(miner, time):
        nonlocal sequence
        state = states[miner]
        available = [q for q in candidates if q.qid not in state["used"]]
        if not available:
            return
        query = rng.choice(available)
        state["used"].add(query.qid)
        response, duration = system.execute(query, miner, state["binding"])
        sequence += 1
        heapq.heappush(queue, (time + duration, rng.getrandbits(64), sequence, miner, response))
        executed.append(duration)

    for miner in range(len(states)):
        start(miner, 0.0)
    winner, finish = None, 0.0
    while queue and len(completed) < max_events:
        finish, _, _, miner, response = heapq.heappop(queue)
        completed.append((finish, miner, response))
        state = states[miner]
        state["chain"].append(response)
        _, won = evaluate_lottery(
            base,
            tuple(r.ciphertext for r in state["chain"]),
            system.difficulty,
            response.complexity,
        )
        if won:
            winner = miner
            break
        state["binding"] = sha256(response.proof)
        start(miner, finish)
    block = None
    if winner is not None:
        block = Block(
            parent, system.miners[winner].public[0], transactions, tuple(states[winner]["chain"])
        )
        if not system.accept_block(block):
            raise RuntimeError("fresh honest block did not extend the canonical chain")
        # Completed losing responses become available for the next frozen tx
        # list. They cannot be inserted retroactively into the winning block.
        system.pending_responses = [
            r for _, _, r in completed if r.query.qid not in system.state.completed
        ]
    seen, first_work = set(), 0
    for _, _, response in completed:
        if response.query.qid not in seen:
            first_work += response.complexity
            seen.add(response.query.qid)
    total = sum(r.complexity for _, _, r in completed)
    return {
        "adopted": winner is not None,
        "termination_reason": "adopted"
        if winner is not None
        else "event_limit"
        if queue
        else "query_pool_exhausted",
        "height": system.height,
        "block_hash": block.digest.hex() if block else None,
        "parent_hash": parent.hex(),
        "block_time": finish,
        "winner": winner,
        "completed_pairs": len(completed),
        "canceled_attempts": len(queue),
        "physical_pair_seconds": sum(executed),
        "completed_complexity": total,
        "first_completed_complexity": first_work,
        "duplicate_complexity": total - first_work,
        "wasted_work_fraction": (total - first_work) / total if total else None,
        "winning_queries": len(block.chain) if block else 0,
        "response_transactions": len(responses) if block else 0,
        "response_eligible": len(system.pending_responses) if block else 0,
        "attempts": [
            {
                "completion_time": time,
                "miner": miner,
                "qid": r.query.qid.hex(),
                "complexity": r.complexity,
                "output_length": r.output_length,
            }
            for time, miner, r in completed
        ],
    }


def run_chain(system, *, blocks=50, pool_size=256, seed=42, prompts=None, max_output=16):
    """Maintain sufficient signed query demand and report every race outcome."""
    if blocks < 1 or pool_size < 1:
        raise ValueError("positive block count and pool size required")
    prompts = prompts or ["A useful machine learning inference query"]
    issued = 0
    results = []
    for index in range(blocks):
        completed = system.state.completed
        # Retain query objects for forks but bound the active working set.
        system.pending = {qid: q for qid, q in system.pending.items() if qid not in completed}
        needed = max(0, pool_size - len(system.pending))
        for _ in range(needed):
            system.submit_query(prompts[issued % len(prompts)], max_output=max_output)
            issued += 1
        row = run_race(system, seed=seed + index)
        results.append(row)
        if not row["adopted"]:
            break
    return results
