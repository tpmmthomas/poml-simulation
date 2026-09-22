# Protocol-to-simulation audit

**Historical audit/design:** superseded for current execution by [the live protocol implementation](llm_live_protocol.md). The descriptions below concern the earlier measurement-bank replay code.

Audited 2026-09-15 against `.scratch/PoML_paper_draft/4_protocols.tex`, the
GPT-2 complexity appendix (`appendix.tex`, `app:llm-complexity`), the LLM
simulation, trace builder, sampler and actual DeepProve benchmark. This audit
concerns `run_llm_experiments.py`, not the retained U-Net/EZKL node software.
Having cryptographic modules elsewhere in this repository does not mean this
driver calls them.

## Direct answers

**Complexity: yes, the appendix reference formula with every operation weight
set to one.** `complexity_for` calls `reference_counts`, sums its inference
and proof operation counts and obtains an integer C(N,K). The reference
schedule contains the padded proof profiles. C depends on the realised K,
not the profiling estimate, a fixed per-query charge, or measured seconds.
Trace replay normally reads this precomputed C rather than recalculating it
at every event. The archived 4,096 traces were independently recomputed:
**zero mismatches**. Independently transcribing the appendix equation and
its table coefficients also matched all 1,953 admissible N,K pairs with
2 <= N, 1 <= K, N+K <= 64.

The appendix leaves operation weights symbolic. The simulation chooses w=1;
it does not claim empirically calibrated relative gas weights. It covers the
appendix's inference/proving operation model, not all auxiliary work listed
in the main protocol (application VRFs, output encryption, signatures and
virtual-ticket hash evaluation). It is a deterministic reference charge,
not an exact counter of every value-dependent proof execution.

**Actual hash lottery: no.** `simulate_race` executes:

```python
win_probability = -math.expm1(C * math.log1p(-p))
if rng.random() < win_probability:
    # adopt this completion
```

This is the Bernoulli probability 1-(1-p)^C for at least one successful
independent virtual ticket, where p plays the role D/2^256. The calibrated
floating-point p drives simulation; the hexadecimal integer D is reported
but is not consumed by a hashing loop. Python's seeded PRNG supplies the
draw. No ciphertext prefix, H(G(s,tx)||Y||j), successful ticket index or
256-bit threshold comparison is constructed. Floating-point calculations
approximate the probability; this is not bit-for-bit execution of Algorithm 1.

This is a useful acceleration for arrival-time and selection simulations
*under the independent random-oracle assumption*. It assumes the valid-ticket
budget, unpredictability and independence that the protocol reduction needs;
it cannot experimentally establish them. It also excludes ticket-hashing
runtime. For scale, an archived N=2,K=16 trace has C=11,326,616,910; literally
hashing every virtual ticket would introduce substantial additional work.
The current measured clock does not account for that cost.

## LLM workload mismatch that must be resolved

The LLM-specific subsection of `4_protocols.tex` defines randomness as Gaussian
perturbations of the initial prompt embeddings, one vector per input token.
It stops at EOS or the fee-derived K_max, and identifies the output as the
pre-decoding logits with constrained decoding to delivered tokens.

The current bank sampler instead varies token-sampling randomness without
embedding perturbation, saves sampled tokens rather than the required logits,
and the revised sentence-completion mode adds punctuation/newline stopping.
Its cap is a CLI parameter, not calculated from live fees. Consequently, the
profiling pilot establishes predictability for this *different* GPT-2 workload.
It does not establish predictability for the paper's specified LLM model.

Before an overnight run intended to evaluate that model, align profiling and
held-out generation with its embedding-noise law, decoding/output contract,
and EOS/fee-derived cap. Then remeasure whether K varies enough to support
short/long selection. Alternatively, explicitly amend the public model and
termination specification in the paper and evaluate that declared variant.
The benchmark pipeline is implemented and smoke-tested, but this alignment
has not been implemented; replacing the dataset alone does not resolve it.

## Mapping to the complete protocol

| Protocol part | What the LLM experiment does | Omitted or assumed |
| --- | --- | --- |
| Initialization/model/CRS | Fixed GPT-2 model and public reference schedule; real DeepProve setup for the measurement batch | No genesis block binding these values, no PoML CRS or miner registration state |
| Miner identities and three key pairs | Integer miner IDs and configured homogeneous rates | No registered signing/inference/encryption key triples; no registration validation or churn |
| Query submission | Benchmark prompt templates, logical query IDs, N and cap | No signed query transactions, qid=H(user key||taskID), input-binding proof, off-chain retrieval or network delivery |
| Eligibility | Queries selected without replacement; sufficient demand in Exp 1–2; finite Q in Exp 3 | Well-formedness, signatures, balances, expiry and previous-chain completion checks are assumed, not executed |
| Frozen transaction list | None is built | No tx freeze, transaction mutation test, or binding G(s,tx) |
| Inference seed r_i | Reproducible profiling/evaluation seed strings; replay samples per-query traces | No H(bind_i||h_u||qid||miner key), no parent-block or H(previous proof) dependency |
| Inference randomness | Real GPT-2 bank sampling uses the repository's Ed25519 sign-then-hash construction and inverse-CDF token selection | Seed-derived keys are experimental, not fixed registered miner VRF keys; messages differ from r_i||t; no protocol VRF validation; embedding perturbation absent |
| Model inference | Real Hugging Face GPT-2 on GPU during bank construction | Each simulated attempt resamples a finite bank; no neural execution or activation trace at each event |
| Termination and delivered output | Revised sampler saves tokens and uses EOS/punctuation/newline/CLI cap; old bank used EOS/cap | Paper specifies EOS/fee-derived cap and pre-decoding logits with constrained decoding; neither that output contract nor termination is enforced by the measured proof |
| Complexity omega | Appendix C(N,K), unit weights, actual sampled K | Not an auxiliary-crypto-inclusive charge or measured wall-time fit; C is not proved in a PoML statement |
| Encryption | None | No encryption VRF, ciphertext, user decryption, encryption cost or privacy experiment |
| Inference proof | Real CUDA DeepProve inference, proof and verification for the same input prompt and fixed K during bank construction | The prover runs its own decoder; the separate Hugging Face sampled output is not constrained. The full PoML relation (input binding, R, C, encryption) is not proved |
| Sequential proof chain | One active attempt per miner; next attempt begins on completion | No proof/ciphertext prefix objects or cryptographic predecessor binding; independence is imposed by fresh replay draws |
| Virtual-ticket lottery | One aggregated Bernoulli draw per completion, weighted by C | No actual H evaluation, ticket index, hash-cost clock, grinding, repeated-input consistency or proof-to-ticket binding |
| Publish/adopt block | Earliest winning event ends the interval; active work is cancelled | No serialized block, gossip, validation delay, competing branches or block broadcast |
| Longest-chain rule | Ordered independent block intervals | No actual hash-linked chain, forks, reorgs, withheld blocks, common-prefix or chain-quality measurement |
| Validation conditions 1–8 | No block validator runs | All transaction, key, seed, VRF, PoML proof, ticket and fee validity checks are assumed |
| RESPONSE_QUERY | Exp 3 credits one useful completion per request ID, after prioritizing the entire winning prefix | No response transaction/proof verification, actual inclusion timing, fees, capacity or expiry; eventual eligibility is assumed |
| Fees and rewards | Block counts/shares, completed work and yield per time | No balances, burn/solve/include transfers or economic profit accounting; block share is not total protocol fee revenue |
| Adversary | One miner changes query-selection policy | No alternative inference/proving algorithm, caching/reuse attack, proof forgery, key grinding, withholding or network attack |

The simulator resolves equal-time completions in event-queue insertion order
and immediately stops at the first winner. It assumes independently provisioned
miners and does not simulate concurrent GPU contention. Unfinished complexity
is the full scheduled charge of cancelled attempts, not a measurement of
partially expended GPU work.

## Archived measurements versus the revised builder

The completed historical campaign used repeated/truncated versions of one
base text, synthetic expected-length categories, 4,096 real GPT-2 rollouts
and 57 CUDA proof timings shared by N,K shape. Those selection results cannot
establish resistance to meaningful input cherry-picking. The archived Exp 3
also mixed pooled K,T draws with a separately assigned N.

The revised builder uses distinct cached WikiText-2 prompt windows, independent
profiling and evaluation seeds, and one proof timing per held-out trial.
Its replay preserves each prompt's joint N,K,C,T association. These changes
repair workload selection and timing association; they do not supply any of
the omitted cryptographic protocol components in the table.

The canonical accounting audit found a further error: only the final winning
pair had been marked canonical. Algorithm 1 includes the winning miner's
entire prefix. The code now credits that whole prefix before responses. If
competing completions for an earlier winning-prefix query have different K
and therefore C, this changes the collision percentage, not just its labels.
A deterministic regression example changed waste from 180/260 to 160/260
work units. Old collision heatmaps therefore need simulation reruns; proof
measurements can be reused for this accounting-only correction. The new
benchmark campaign includes the corrected rule.

Responses produced during an interval cannot appear in that candidate's
already-frozen tx list. Exp 3's credit means *eligible for later submission*,
not included in the winning block. This is appropriate for the requested
collision-only metric only under the stated eventual-inclusion assumption.

## Supported conclusions and recommended scope

- Exp 1 measures simulated block intervals under an empirical workload and
  an idealized complexity-weighted lottery. It does not show full-protocol
  deployment liveness or validation throughput.
- Exp 2 can measure whether the tested profile-based selection policies gain
  simulated block yield with the chosen C and measured runtimes. It cannot
  prove general attack resilience or correct cryptographic ticket binding.
- Exp 3 estimates completed duplicate work assuming all eligible unique
  responses remain usable. It does not measure transaction inclusion or fees.

Retain the accelerated lottery for the large statistical simulation, but name
it explicitly. A separate small protocol conformance implementation is needed
to test actual bindings, H(.)<D, prefix construction and validator rejection
paths. Proving the full PoML NP statement additionally requires changes to
the prover circuit; simply hashing replay records would not close that gap.
Full-protocol timing would also need auxiliary cryptography, hashing and
validation costs. The historical paper wording should not claim those tests.

## Code pointers

- `src/poml_sim/llm_simulation.py`: `complexity_for`, `sample_trace`,
  `simulate_race` (probability draw, prefix/response accounting).
- `src/poml_sim/gpt2_work.py`: `inference_counts`, `proof_axis_macs`,
  `reference_counts`; `config/gpt2_reference_schedule.json`.
- `experiments/build_real_llm_trace_bank.py`: `_rollouts`, `_prove_batch`,
  `_parse_proofs`, `main` (explicit `prover_limitation` metadata).
- `experiments/gpt2_experiments.py`: `vrf_uniforms`, `generate_trace`.
- `src/poml_sim/vrf.py`: experimental Ed25519 sign-then-hash construction.
- `.scratch/deep-prove/zkml/src/bin/bench/llm.rs`: `run_elements`, `prove`,
  `verify_fixed_length`, ledger emission after successful verification.
