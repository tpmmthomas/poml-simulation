"""Protocol regressions covering bindings, eligibility, settlement and forks."""

from dataclasses import replace
import copy
import pytest
from poml_sim.backends import SmokeBackend
from poml_sim.crypto import canonical, public_key, sha256, sign
from poml_sim.lottery import LIMIT
from poml_sim.protocol_inputs import decrypt_output, experimental_key
from poml_sim.simulation import run_chain, run_race
from poml_sim.system import (
    Block,
    MinerKeys,
    PoMLSystem,
    Registration,
    Transactions,
    Transfer,
    fingerprint,
)


@pytest.fixture
def system():
    return PoMLSystem(SmokeBackend(), miners=2, difficulty=LIMIT)


def candidate(system, query=None, miner=0, transactions=Transactions(), parent=None):
    query = query or system.submit_query("hello world")
    parent = parent or system.tip
    response, _ = system.execute(query, miner, fingerprint(parent, transactions))
    return Block(parent, system.miners[miner].public[0], transactions, (response,))


def test_real_signatures_and_unique_task_ids_bind_queries(system):
    a, b = system.submit_query("same"), system.submit_query("same")
    assert a.qid != b.qid and (a.task_id, b.task_id) == (1, 2)
    for bad in (
        replace(a, signature=b"bad"),
        replace(a, inputs=(1,)),
        replace(a, max_fee=a.max_fee + 1),
        replace(a, qid=b.qid),
    ):
        with pytest.raises(ValueError):
            system.validate_query(bad, system.state, 1)


def test_fee_cap_stops_decoding_before_over_budget(system):
    query = system.submit_query("hi", max_fee=12, max_output=10)
    assert system.affordable_cap(query) == 2  # C=2+K, phi=3.
    response = candidate(system, query).chain[0]
    assert response.cap == 2 and response.output_length <= 2
    too_small = system.submit_query("hi", max_fee=8)
    with pytest.raises(ValueError, match="afford"):
        system.affordable_cap(too_small)


def test_settlement_burns_fees_without_minting_block_rewards(system):
    query = system.submit_query("accounting")
    initial = sum(system.state.balances.values())
    block = candidate(system, query)
    cost = block.chain[0].complexity
    assert system.accept_block(block)
    assert system.state.burned == cost
    assert system.state.balances[block.producer] == 2 * cost
    assert sum(system.state.balances.values()) + system.state.burned == initial
    assert query.qid in system.state.completed
    with pytest.raises(ValueError, match="completed"):
        system.validate_query(query, system.state, 2)


def test_ciphertext_decrypts_to_the_bound_query_output(system):
    response = candidate(system).chain[0]
    plaintext = decrypt_output(response.ciphertext, system.user_secrets[response.query.user])
    assert response.query.qid.hex().encode() in plaintext
    assert b'"output"' in plaintext


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", b"x" * 32),
        ("ciphertext", b"wrong"),
        ("complexity", 999),
        ("proof", b"wrong"),
        ("randomness", ()),
        ("encryption_vrf", (b"x" * 32, b"y" * 64)),
        ("output_length", 999),
    ],
)
def test_rejects_tampered_responses_even_when_solver_resigns(system, field, value):
    block = candidate(system)
    bad = replace(block.chain[0], **{field: value})
    bad = replace(bad, signature=sign(system.miners[0].identity, bad.message()))
    with pytest.raises(ValueError):
        system.validate_block(replace(block, chain=(bad,)))


def test_frozen_transactions_and_previous_proof_bind_every_mining_seed(system):
    q1, q2 = system.submit_query("first"), system.submit_query("second")
    first = candidate(system, q1)
    wrong, _ = system.execute(q2, 0, fingerprint(system.tip, Transactions()))
    with pytest.raises(ValueError, match="binding"):
        system.validate_block(replace(first, chain=first.chain + (wrong,)))
    correct, _ = system.execute(q2, 0, sha256(first.chain[0].proof))
    system.validate_block(replace(first, chain=first.chain + (correct,)))
    registration_keys = MinerKeys.deterministic(42, 20)
    registration = Registration(
        registration_keys.public,
        sign(registration_keys.identity, canonical(registration_keys.public)),
    )
    with pytest.raises(ValueError, match="binding"):
        system.validate_block(
            replace(first, transactions=Transactions(registrations=(registration,)))
        )


def test_response_seed_need_not_match_current_block_but_query_settles_once(system):
    losing = candidate(system, miner=1).chain[0]
    transactions = Transactions(responses=(losing,))
    winning = candidate(system, transactions=transactions)
    cost = losing.complexity
    system.accept_block(winning)
    assert system.state.balances[losing.solver] == cost
    assert system.state.balances[winning.producer] == cost + 2 * winning.chain[0].complexity
    assert len(system.state.completed) == 2


def test_rejects_duplicate_query_within_block_and_across_response_chain(system):
    block = candidate(system)
    with pytest.raises(ValueError, match="duplicate"):
        system.validate_block(replace(block, chain=block.chain * 2))
    response = block.chain[0]
    transactions = Transactions(responses=(response,))
    mining, _ = system.execute(response.query, 0, fingerprint(system.tip, transactions))
    with pytest.raises(ValueError, match="duplicate"):
        system.validate_block(Block(system.tip, mining.solver, transactions, (mining,)))


def test_registration_activates_after_acceptance_and_rejects_key_reuse(system):
    keys = MinerKeys.deterministic(42, 5)
    tx = Registration(keys.public, sign(keys.identity, canonical(keys.public)))
    block = candidate(system, transactions=Transactions(registrations=(tx,)))
    assert keys.public[0] not in system.state.miners
    system.accept_block(block)
    assert keys.public[0] in system.state.miners
    with pytest.raises(ValueError, match="registered"):
        system.validate_block(candidate(system, transactions=Transactions(registrations=(tx,))))


def test_balances_are_checked_after_each_debit_and_invalid_blocks_are_atomic(system):
    query = system.submit_query("balance")
    user_secret = experimental_key(system.seed, "user:user:sig")
    recipient = public_key(experimental_key(1, "recipient"))
    tx = Transfer(query.user, recipient, system.state.balances[query.user], 0)
    tx = replace(tx, signature=sign(user_secret, tx.message()))
    block = candidate(system, query, transactions=Transactions(transfers=(tx,)))
    before = copy.deepcopy(system.state)
    with pytest.raises(ValueError, match="remaining balance"):
        system.accept_block(block)
    assert system.state == before


def test_expiry_and_insufficient_upfront_balance_reject_eligibility(system):
    query = system.submit_query("expire", expires=1)
    with pytest.raises(ValueError, match="expired"):
        system.validate_query(query, system.state, 2)
    system.state.balances[query.user] = 0
    with pytest.raises(ValueError, match="balance"):
        system.validate_query(query, system.state, 1)


def test_longest_branch_wins_and_first_received_breaks_equal_height_ties(system):
    q1, q2, q3 = [system.submit_query(str(i)) for i in range(3)]
    a, b = candidate(system, q1), candidate(system, q2, miner=1)
    assert system.accept_block(a)
    assert not system.accept_block(b)
    assert system.tip == a.digest
    c = candidate(system, q3, miner=1, parent=b.digest)
    assert system.accept_block(c)
    assert q1.qid not in system.state.completed and q2.qid in system.state.completed


def test_rejects_empty_chain_unknown_parent_and_unregistered_producer(system):
    block = candidate(system)
    for bad in (
        replace(block, chain=()),
        replace(block, parent=b"missing"),
        replace(block, producer=b"unknown"),
    ):
        with pytest.raises(ValueError):
            system.validate_block(bad)


def test_chain_smoke_is_reproducible_and_counts_cancellation():
    one = PoMLSystem(SmokeBackend(), miners=3, difficulty=LIMIT)
    two = PoMLSystem(SmokeBackend(), miners=3, difficulty=LIMIT)
    first = run_chain(one, blocks=3, pool_size=20)
    second = run_chain(two, blocks=3, pool_size=20)
    assert first == second
    assert one.height == 3
    assert all(row["canceled_attempts"] == 2 for row in first)


def test_losing_responses_are_frozen_only_in_a_later_block(system):
    losing = candidate(system, miner=1).chain[0]
    system.pending_responses = [losing]
    system.submit_query("next")
    result = run_race(system, seed=1)
    assert result["response_transactions"] == 1
    assert losing.query.qid in system.state.completed
