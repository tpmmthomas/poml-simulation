# Publication implementation and paper audit

The `fc27` branch replaces the historical experiment workspace with a standalone
teaching system and the experiments present in the current manuscript. Its
purpose is to expose PoML's mechanics faithfully and make implementation limits
inspectable. It is not a production consensus implementation or a claim that
all archived paper measurements have been reproduced.

## Source of truth and agreed choices

Audited against `.scratch/PoML_paper_draft`, read through `main.tex`, whose active
evaluation is **`7_experiments_llm.tex`**, not the older diffusion evaluation file.
The local draft is not redistributed. SHA-256 snapshots of the relevant sources:

| Source | SHA-256 |
| --- | --- |
| `main.tex` | `d9cdac036e105ac203f25fe2bc841703557f88e503c6417f68203c992a16dede` |
| `4_protocols.tex` | `9f1357c3d02404cd4c3f29bf54793926f9a04a0845cbfd76c79f6dc76d949d13` |
| `7_experiments_llm.tex` | `6a6856475a1b5eeb378e812c5e236177e93e9c09fc1adc02ffad22bd227bd66c` |
| `appendix.tex` | `686f639d202d2ca9a5336f5b6708b13e56236061827ae8c6fd749b3a0b617535` |

Author clarifications incorporated here: tiny U-Net/EZKL for the diffusion
simulator; full SD only in the appendix; common reverse noise for perturbed
DDPM trajectories; nonnegative runtime weight fitting with a uniform default;
MIT for original project code. The single-hash lottery is taken from the latest
protocol, superseding historical enumeration of C hashes.

## Protocol mapping

| Paper condition | Implementation |
| --- | --- |
| Registered `(vk, vk_inf, vk_enc)` | `MinerKeys`, `Registration`; self-signed distinct keys, activation in later blocks |
| Unique task and committed input | `Query`, `submit_query`, `validate_query`; qid, signed commitment, fee cap, expiry, parent balance |
| Frozen parent and transactions | `Transactions`, `fingerprint`; changes to tx invalidate the mining seed |
| Per-query proof chain | `query_seed`; first uses G(parent,tx), later seeds use H(previous proof) |
| Independent inference and encryption randomness | Indexed inference VRF and separate encryption VRF over seed/qid |
| Result encryption | Deterministic-coin X25519/HKDF/AES-GCM to the recipient public key |
| Complexity-aware termination | `affordable_cap`; output/EOS length and public count validation |
| Single-hash lottery | `lottery.py`; SHA-256 over G and the full ciphertext prefix, last completed pair's C |
| Account state update | Transfers, registrations, response transactions, mining responses; every debit checked atomically |
| No duplicate settlement/reward inflation | Distinct query IDs within/across tx and mining chain; completion set; burn/solve/include only |
| Completed losing work | Deferred response transactions retain their original seed; no rebinding to a new block |
| Chain choice | Parent-specific snapshots, longest valid chain, first received on equal height |

`system.py` separates validation from `simulation.py`, which supplies the honest
race driver. Fees default to one unit each for burn, solver and includer. Demo
users receive genesis balances; no funding transaction generator is supplied.
The driver freezes previously available responses before starting the race,
selects queries uniformly without replacement per miner, cancels remaining
virtual jobs at adoption, and replenishes the query pool between blocks.

For q=2^256 and positive finite C, threshold rounding uses
`q - ceil((q-D)^C / q^(C-1))`. Small cases use exact integers; large cases use
increasing-precision directed bounds until their integer results agree. This
avoids cancellation at tiny difficulty and avoids rounding a probability below
one up to certain acceptance. D=q is exposed for bounded integration tests.

## Trusted-host boundary

Both real backends generate and verify genuine **model** proofs. No complete
private PoML NP relation is implemented by either underlying circuit. The host
checks the input commitment, VRF-derived execution inputs, complexity, output
encryption and transcript, then records a canonical receipt. Ledger validators
in the same simulator accept only previously checked receipts. A detached node
cannot independently validate that receipt as a PoML NIZK. Host-visible input
witnesses, experiment keys, VRF values and public DeepProve logits provide no
PoML confidentiality guarantee.

The experimental VRF is deterministic Ed25519 sign-then-hash, not a standardized
ECVRF; SHA-256 stands in for the paper's ZK-friendly commitment hash. Public
canonical JSON encodings make simulator hashes deterministic; they are not a
wire-protocol specification. DeepProve setup uses reproducible test randomness,
not a production setup ceremony. EZKL uses its downloaded SRS and fixed weights.

GPT-2 injects one noise row per original prompt token, reserves N+cap indexed
VRF outputs, and exposes generated tokens plus pre-decoding logits. The patched
worker proves quantized inference and performs a composite inverse-CDF check
with 53-bit uniforms and 48-bit probability masses. The float32 compatibility
experiment has ordinary float64 probability accumulation, so it is not asserted
to be bit-identical to the prover. Reference C counts the appendix's original
graph and is an explicit approximation for the extra perturbation/sampling work.

The tiny U-Net has 11,401 seeded random weights, a 2×8×8 input and 1×8×8 output.
Channel 0 is seed-derived Gaussian noise; channel 1 is synthetic conditioning
from the query text. There is no text encoder, training, timestep embedding,
reverse diffusion chain or useful-image-quality claim. Its fixed shape gives
C=1. Full Stable Diffusion with actual DDPM recurrence is separate from proving.

## Experiment alignment and remaining abstractions

The liveness command offers fresh proofs with the full protocol and measured
service-time replay. Both use zero-delay virtual miners; physical prover jobs
are serialized to share a GPU. Incomplete virtual jobs may already have consumed
physical GPU time: that cost is logged separately from completed work. Setup,
verification, communication, signature and encryption latency are excluded from
virtual inference/proof service times. Difficulty calibration is approximate,
not a promise of exactly 300 seconds. Actual hashing and the optional Poisson
baseline are distinct output modes.

The wasted-work replay preserves each sampled `(T,C)` pair and credits the
first completion for each query. It uses a pooled timing distribution independent
of query identity, fixed query pools and a uniform 256-bit threshold draw. This
matches the paper's lottery distribution under its random-oracle model without
pretending to produce ciphertexts/proofs for all 36,000 races. Exhaustion remains
visible; only adopted races enter the reported conditional means.

The appendix diffusion implementation now uses 50 DDPM steps instead of the
historical 20-step PNDM run. It compares pre-denoiser latents x_t, not internal
U-Net layers, and does not establish the paper's computational independence
assumption as a theorem. The GPT-2 activation experiment now decodes independently
instead of feeding both runs a shared clean continuation. It is EOS-aware and
reports actual comparison counts. Benchmark scoring details unspecified by the
manuscript are explicit in [the run guide](../experiments.md).

The operation campaign exposes the requested 40 distinct pairs/24 lengths.
The shipped reference B_P is retained from verified historical instrumentation;
it is not represented as a newly executed full 40-pair campaign. The calculator
covers every supported N/K. Runtime fitting is a documented calibration method,
with prompt-grouped held-out evaluation and uniform default weights. The weights
are underidentified when features are collinear, so rank is reported.

## Packaging and migration

Core code lives in `src/poml_sim/`; the reference operation matrix moved from
`config/` into package data so an installed wheel works outside the source tree.
Python extras isolate optional ML/prover dependencies, pinned to tested versions.
The wheel provides the simulator library/CLI; the source distribution also
includes experiment commands, setup scripts, patches, model source and docs.

Historical alternate experiments, live/profiled mining variants, timing-only
SD studies, strategic-selection studies, result-specific plotters, old config,
tracked egg-info and superseded docs/tests were removed. Existing ignored local
results, downloaded models, paper sources and environment data were preserved.
Historical commands/imports are not compatible with this branch: use the new
[README](../../README.md) and [experiment entry points](../../experiments/README.md).
No historical paper table or plot is silently copied into fresh output.
