"""Paper-aligned protocol objects, trusted-host proof relation and ledger.

Signatures, VRFs, ciphertexts and lotteries are real. The complete PoML relation
is checked by the trusted simulator host, wrapping the model-only ZK proofs;
this is an explicit simulation abstraction, not a private NIZK construction.
"""

from dataclasses import dataclass, field, replace
import copy
import math

from .crypto import canonical, public_key, sha256, sign, verify_signature
from .lottery import LIMIT, evaluate_lottery
from .protocol_inputs import experimental_key, encryption_public_key, encrypt_output
from .vrf import vrf_eval, vrf_verify


@dataclass(frozen=True)
class MinerKeys:
    """Three independently domain-separated experiment keys per miner."""

    identity: bytes
    inference: bytes
    encryption: bytes

    @classmethod
    def deterministic(cls, seed, index):
        """Create reproducible test keys, never production secrets."""
        return cls(
            *(experimental_key(seed, f"miner:{index}:{kind}") for kind in ("sig", "inf", "enc"))
        )

    @property
    def public(self):
        """Return the public triple registered in chain state."""
        return tuple(public_key(key) for key in (self.identity, self.inference, self.encryption))


@dataclass(frozen=True)
class Query:
    """Signed on-chain query metadata plus the locally available off-chain input."""

    qid: bytes
    user: bytes
    encryption_pk: bytes
    task_id: int
    inputs: tuple
    commitment: bytes
    expires: int
    max_fee: int
    max_output: int
    signature: bytes = b""

    def message(self):
        """Bind all public fields; raw inputs are committed and sent off-chain."""
        return canonical(
            {
                "qid": self.qid,
                "user": self.user,
                "encryption_pk": self.encryption_pk,
                "task_id": self.task_id,
                "commitment": self.commitment,
                "expires": self.expires,
                "max_fee": self.max_fee,
                "max_output": self.max_output,
            }
        )


@dataclass(frozen=True)
class Registration:
    """Self-signed binding of a miner's three distinct public keys."""

    keys: tuple[bytes, bytes, bytes]
    signature: bytes = b""


@dataclass(frozen=True)
class Transfer:
    """Signed ordinary account transfer with a replay-preventing nonce."""

    sender: bytes
    recipient: bytes
    amount: int
    nonce: int
    signature: bytes = b""

    def message(self):
        """Return the canonical transfer signing payload."""
        return canonical(replace(self, signature=b""))


@dataclass(frozen=True)
class Response:
    """Completed query response, reusable after a miner loses the block race."""

    query: Query
    solver: bytes
    seed: bytes
    randomness: tuple[tuple[bytes, bytes], ...]
    encryption_vrf: tuple[bytes, bytes]
    ciphertext: bytes
    complexity: int
    output_length: int
    cap: int
    proof: bytes
    signature: bytes = b""

    def message(self):
        """Bind every response field to the registered solver identity."""
        return canonical(replace(self, signature=b""))


@dataclass(frozen=True)
class Transactions:
    """A frozen block transaction list in the paper's prescribed order."""

    transfers: tuple[Transfer, ...] = ()
    registrations: tuple[Registration, ...] = ()
    responses: tuple[Response, ...] = ()


@dataclass(frozen=True)
class Block:
    """A parent-linked block carrying a nonempty mining proof chain."""

    parent: bytes
    producer: bytes
    transactions: Transactions
    chain: tuple[Response, ...]

    @property
    def digest(self):
        """Hash all consensus fields, including proof and ciphertext bindings."""
        return sha256(canonical(self))


def fingerprint(parent: bytes, transactions: Transactions) -> bytes:
    """Compute G(s,tx) after freezing all transactions."""
    return sha256(b"poml-G-v1", parent, canonical(transactions))


def query_seed(binding: bytes, query: Query, miner: bytes) -> bytes:
    """Bind the current proof-chain position, input, qid and miner identity."""
    return sha256(binding, query.commitment, query.qid, miner)


@dataclass
class State:
    """Parent-specific account, registration and query-completion state."""

    balances: dict[bytes, int]
    miners: dict[bytes, tuple[bytes, bytes, bytes]]
    completed: set[bytes] = field(default_factory=set)
    nonces: dict[bytes, int] = field(default_factory=dict)
    burned: int = 0
    height: int = 0


class PoMLSystem:
    """Validate blocks and adopt the longest valid branch, first received on ties."""

    def __init__(self, backend, *, miners=4, difficulty=LIMIT // 100, seed=42, fees=(1, 1, 1)):
        if (
            type(miners) is not int
            or miners < 1
            or type(difficulty) is not int
            or not 1 <= difficulty <= LIMIT
        ):
            raise ValueError("positive miner count and valid difficulty required")
        if len(fees) != 3 or any(type(fee) is not int or fee < 0 for fee in fees) or not sum(fees):
            raise ValueError(
                "burn, solve and include fees must be nonnegative integers with positive total"
            )
        self.backend, self.seed, self.difficulty, self.fees = backend, seed, difficulty, tuple(fees)
        self.miners = [MinerKeys.deterministic(seed, index) for index in range(miners)]
        self.genesis = sha256(
            canonical(
                {
                    "model": backend.identity,
                    "difficulty": difficulty,
                    "fees": fees,
                    "miners": [m.public for m in self.miners],
                }
            )
        )
        self.tip = self.genesis
        self.states = {self.genesis: State({}, {m.public[0]: m.public for m in self.miners})}
        self.blocks: dict[bytes, Block] = {}
        self.pending: dict[bytes, Query] = {}
        self.pending_responses: list[Response] = []
        self.user_secrets = {}
        self.task_counters = {}
        # SECURITY: these are trusted-host execution receipts, not adversarially
        # sound NIZK proofs. Model adapters verify their underlying proofs first.
        self._receipts: dict[bytes, bytes] = {}
        self.executions = []

    @property
    def state(self):
        """Return the current canonical chain's state."""
        return self.states[self.tip]

    @property
    def height(self):
        """Return the number of adopted non-genesis blocks."""
        return self.state.height

    def submit_query(self, prompt, *, user_id="user", max_fee=10**24, max_output=16, expires=10**9):
        """Sign and validate a fresh query, with monotonically increasing taskID."""
        if type(max_output) is not int or max_output < 1:
            raise ValueError("max_output must be positive")
        secret = experimental_key(self.seed, f"user:{user_id}:sig")
        encryption_secret = experimental_key(self.seed, f"user:{user_id}:enc")
        user = public_key(secret)
        task_id = self.task_counters.get(user, 0) + 1
        self.task_counters[user] = task_id
        inputs = self.backend.prepare_input(prompt)
        qid = sha256(user, task_id.to_bytes(8, "big"))
        query = Query(
            qid,
            user,
            encryption_public_key(encryption_secret),
            task_id,
            inputs,
            sha256(canonical(inputs), qid),
            expires,
            max_fee,
            max_output,
        )
        query = replace(query, signature=sign(secret, query.message()))
        if user not in self.user_secrets:
            if self.height:
                raise ValueError("new users must be funded at genesis in this simulator")
            self.state.balances[user] = 10**30
            self.user_secrets[user] = encryption_secret
        self.validate_query(query, self.state, self.height + 1)
        self.pending[qid] = query
        return query

    def validate_query(self, query, parent_state, height):
        """Check the paper's eligibility conditions against parent state."""
        if (
            any(
                type(x) is not int
                for x in (query.task_id, query.expires, query.max_fee, query.max_output)
            )
            or min(query.task_id, query.max_fee, query.max_output) < 1
        ):
            raise ValueError("malformed query integers")
        if query.task_id >= 2**64:
            raise ValueError("taskID must fit in 64 bits")
        if (
            len(query.user) != 32
            or len(query.encryption_pk) != 32
            or query.encryption_pk == bytes(32)
        ):
            raise ValueError("malformed query keys")
        if query.qid != sha256(query.user, query.task_id.to_bytes(8, "big")):
            raise ValueError("qid does not match user and taskID")
        if query.commitment != sha256(canonical(query.inputs), query.qid):
            raise ValueError("input commitment mismatch")
        if not verify_signature(query.user, query.message(), query.signature):
            raise ValueError("invalid query signature")
        if query.qid in parent_state.completed or height > query.expires:
            raise ValueError("query completed or expired")
        if parent_state.balances.get(query.user, 0) < query.max_fee:
            raise ValueError("insufficient available balance for maximum fee")

    def affordable_cap(self, query):
        """Find the largest safe EOS/cap bound under the public fee schedule."""
        cap = self.backend.max_output(query.inputs, query.max_output)
        affordable = 0
        for k in range(1, cap + 1):
            if sum(self.fees) * self.backend.complexity(query.inputs, k) > query.max_fee:
                break
            affordable = k
        if affordable == 0:
            raise ValueError("query cannot afford even one output")
        return affordable

    def execute(self, query, miner_index, binding):
        """Run fresh VRF-driven inference, verify its model proof and encrypt output."""
        keys = self.miners[miner_index]
        cap = self.affordable_cap(query)
        seed = query_seed(binding, query, keys.public[0])
        randomness = tuple(
            vrf_eval(keys.inference, seed + t.to_bytes(8, "big"))
            for t in range(1, self.backend.randomness_count(query.inputs, cap) + 1)
        )
        result = self.backend.run(query.inputs, tuple(z for z, _ in randomness), cap)
        if (
            not 1 <= result.output_length <= cap
            or not math.isfinite(result.duration)
            or result.duration <= 0
        ):
            raise ValueError("backend returned invalid output length or duration")
        if result.complexity != self.backend.complexity(query.inputs, result.output_length):
            raise ValueError("backend complexity disagrees with public schedule")
        encryption_vrf = vrf_eval(keys.encryption, seed + query.qid)
        plaintext = canonical({"qid": query.qid, "output": result.output})
        ciphertext = encrypt_output(plaintext, query.encryption_pk, encryption_vrf[0])
        proof = canonical(
            {
                "relation": "trusted-host-composite",
                "model_proof": sha256(result.proof),
                "input": query.commitment,
                "output": sha256(result.output),
                "seed": seed,
                "complexity": result.complexity,
                "ciphertext": sha256(ciphertext),
            }
        )
        response = Response(
            query,
            keys.public[0],
            seed,
            randomness,
            encryption_vrf,
            ciphertext,
            result.complexity,
            result.output_length,
            cap,
            proof,
        )
        self._receipts[sha256(proof)] = sha256(response.message())
        response = replace(response, signature=sign(keys.identity, response.message()))
        self.executions.append(
            {
                "qid": query.qid.hex(),
                "prompt_sha256": sha256(canonical(query.inputs)).hex(),
                "prompt_length": len(query.inputs),
                "output_length": result.output_length,
                "complexity": result.complexity,
                "duration": result.duration,
                "proof_sha256": sha256(result.proof).hex(),
                **result.metadata,
            }
        )
        return response, result.duration

    def validate_response(self, response, parent_state, height):
        """Check solver registration, both VRFs, complexity and the host receipt."""
        self.validate_query(response.query, parent_state, height)
        keys = parent_state.miners.get(response.solver)
        if keys is None or not verify_signature(
            response.solver, response.message(), response.signature
        ):
            raise ValueError("unregistered solver or invalid response signature")
        if (
            response.cap != self.affordable_cap(response.query)
            or not 1 <= response.output_length <= response.cap
        ):
            raise ValueError("invalid termination cap or output length")
        if response.complexity != self.backend.complexity(
            response.query.inputs, response.output_length
        ):
            raise ValueError("invalid public complexity")
        if len(response.randomness) != self.backend.randomness_count(
            response.query.inputs, response.cap
        ):
            raise ValueError("wrong inference randomness schedule length")
        for t, (z, proof) in enumerate(response.randomness, 1):
            if not vrf_verify(keys[1], response.seed + t.to_bytes(8, "big"), z, proof):
                raise ValueError("invalid inference VRF")
        if not vrf_verify(keys[2], response.seed + response.query.qid, *response.encryption_vrf):
            raise ValueError("invalid encryption VRF")
        if self._receipts.get(sha256(response.proof)) != sha256(response.message()):
            raise ValueError("invalid trusted-host PoML relation receipt")

    def _settle(self, response, producer, state):
        query = response.query
        fee = sum(self.fees) * response.complexity
        if (
            query.qid in state.completed
            or fee > query.max_fee
            or fee > state.balances.get(query.user, 0)
        ):
            raise ValueError("duplicate settlement or fee exceeds remaining balance/cap")
        state.balances[query.user] -= fee
        state.burned += self.fees[0] * response.complexity
        for recipient, rate in ((response.solver, self.fees[1]), (producer, self.fees[2])):
            state.balances[recipient] = (
                state.balances.get(recipient, 0) + rate * response.complexity
            )
        state.completed.add(query.qid)

    def validate_block(self, block):
        """Return an updated snapshot only if every block condition passes."""
        if block.parent not in self.states:
            raise ValueError("unknown block parent")
        parent = self.states[block.parent]
        if block.producer not in parent.miners or not block.chain:
            raise ValueError("producer is not active or mining chain is empty")
        qids = [r.query.qid for r in (*block.transactions.responses, *block.chain)]
        if len(qids) != len(set(qids)):
            raise ValueError("duplicate query in block")
        state = copy.deepcopy(parent)
        state.height += 1
        for tx in block.transactions.transfers:
            if (
                tx.amount <= 0
                or tx.nonce != state.nonces.get(tx.sender, 0)
                or not verify_signature(tx.sender, tx.message(), tx.signature)
            ):
                raise ValueError("invalid account transfer")
            if state.balances.get(tx.sender, 0) < tx.amount:
                raise ValueError("transfer exceeds balance")
            state.balances[tx.sender] -= tx.amount
            state.balances[tx.recipient] = state.balances.get(tx.recipient, 0) + tx.amount
            state.nonces[tx.sender] = tx.nonce + 1
        used_keys = {key for triple in parent.miners.values() for key in triple}
        for tx in block.transactions.registrations:
            if (
                len(tx.keys) != 3
                or any(len(key) != 32 for key in tx.keys)
                or len(set(tx.keys)) != 3
                or used_keys.intersection(tx.keys)
            ):
                raise ValueError("malformed or already registered miner keys")
            if not verify_signature(tx.keys[0], canonical(tx.keys), tx.signature):
                raise ValueError("invalid miner registration signature")
            used_keys.update(tx.keys)
            state.miners[tx.keys[0]] = tx.keys
        for response in block.transactions.responses:
            self.validate_response(response, parent, state.height)
            self._settle(response, block.producer, state)
        base = fingerprint(block.parent, block.transactions)
        binding = base
        for response in block.chain:
            self.validate_response(response, parent, state.height)
            if response.solver != block.producer or response.seed != query_seed(
                binding, response.query, block.producer
            ):
                raise ValueError("mining seed/solver binding mismatch")
            self._settle(response, block.producer, state)
            binding = sha256(response.proof)
        _, won = evaluate_lottery(
            base,
            tuple(r.ciphertext for r in block.chain),
            self.difficulty,
            block.chain[-1].complexity,
        )
        if not won:
            raise ValueError("block did not win complexity-weighted lottery")
        return state

    def accept_block(self, block):
        """Store a valid branch and change the tip only for a strictly longer chain."""
        state = self.validate_block(block)
        key = block.digest
        self.blocks[key], self.states[key] = block, state
        if state.height > self.height:
            self.tip = key
            return True
        return False
