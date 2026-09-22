# Live GPT-2 PoML experiments

Experiments 1 and 2 first build a genuine measurement bank: each WikiText-2
prompt is run with perturbed GPT-2 inference and a full CUDA DeepProve proof
under several independent challenges. The chain simulations then replay one
recorded inference+proof duration per query while recomputing bindings,
complexity, encryption, and literal SHA-256 virtual tickets for every attempt.
Experiment 3 retains the authorized empirical-duration simulation.

The bank is the only stage that invokes the prover; the replay campaign is the
paper's execution path. Historical experiment numbers do not describe this
implementation. Do not reuse them as results of the revised experiments.

An existing bank can also calibrate nonuniform operation weights and replay
Experiment 2 without new measurements; see the
[runtime-weighting guide](runtime_weighted_selection.md).

## Run

From the repository root, in the existing `.venv` with the cached GPT-2 and
WikiText-2 data:

```bash
.venv/bin/python scripts/prepare_deepprove_protocol.py
.venv/bin/python experiments/run_live_llm_experiments.py all \
  --device cuda:1 --output /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/live-protocol
```

This is a foreground command suitable for tmux. No `nohup` is used. The second
command performs calibration and all three experiments. Repeat it with
`--resume` after interruption. Configuration, source and binary hashes must
match; a changed implementation requires a new campaign directory.

To run stages separately, use the same arguments/output and replace `all` with
`calibrate`, `liveness`, `cherry-pick`, or `wasted-work`. After the first stage,
append `--resume`. Calibration and the measured bank are reused by Experiments 1
and 2. Repeating a completed stage skips it; producing additional independent
samples requires a new campaign/seed.

Defaults: 32 distinct prompts of lengths 8/16/24/32; output cap 32; eight
independent profiling draws and two held-out calibration proofs per prompt;
relative embedding noise alpha=0.05; temperature=1, top-k=50, top-p=0.95;
four virtual miners; target 300 seconds; Experiment 1 produces 50 blocks;
Experiment 2 produces 10 blocks per policy and seed, with three seeds.
Experiment 3 uses the former 1–2–5 M/Q grids, Q>M, 100 seeds, and targets
300/600/900 seconds. `--workers 8` parallelizes its CPU cells.

The expensive stage is the one-time bank (32 prompts × 2 full proof pairs, in
addition to eight inference-only profiling draws). Once it is complete, the
170-block replay and Experiment 3 are CPU/event work; they do not invoke GPT-2
or DeepProve again. The bank duration is hardware-dependent and is recorded in
`preparation/bank_manifest.json`. Smaller `--blocks`, `--selection-blocks`, and
`--seeds` remain useful for pilots, with correspondingly weak statistical power.
The source defaults preserve the 50-block Experiment 1.
Reserve roughly 120–150 GB for a conservative full-run artifact budget. Bulk
workspace artifacts now use `/mnt/nas/thomas_work/poml-sim`, with about 14 TB
available at migration time. Existing workspace paths resolve there through
symlinks; see the [storage map](nas_artifact_storage.md).

## What is implemented

| Component | Execution and limits |
| --- | --- |
| Queries | Distinct token windows from distinct non-heading WikiText-2 test rows; exact tokens, row/offset, content hashes and dataset fingerprint recorded. The pending pool is replenished with new query IDs under the sufficient-demand assumption. |
| Randomness | Separate miner identity/inference/encryption keys. r binds the previous block/actual proof digest, input commitment, qid and miner identity. The existing Ed25519 sign-then-hash VRF remains an experimental substitute for a formally unbiasable ECVRF. |
| Embedding perturbation | N Gaussian 768-dimensional vectors from the first N VRF outputs, scaled by alpha times the public embedding table's population SD. A new public noise tensor is added inside the proved graph. Quantization rounds the supplied noise to the model's fixed scale. Generated token embeddings receive zero perturbation. |
| Sampling | Remaining reserved VRF outputs provide independent 53-bit uniforms. Temperature/top-k/top-p and inverse-CDF sampling use proved logits; ties sort by token ID. Exponential masses are rounded to 48-bit fixed-point values; CDF selection uses integer arithmetic. EOS or the configured cap terminates generation; punctuation does not. |
| Inference/proof agreement | Cached autoregressive generation is checked against a final full-sequence trace. The composite verifier checks that the public proof input contains the prompt and every sampled token in order, and that the noise, sampling and stopping are consistent. |
| Proof | Genuine DeepProve proof of the quantized GPT-2 graph, including the noise addition and logits. Sampling is an explicit check in the composite verifier over public logits. It is not a private sampling proof operator or the full PoML NP relation. |
| Encryption | Actual X25519/HKDF/AES-GCM encryption of tokens/logits/qid, using independent encryption-VRF-derived randomness. Decryption consistency is checked. Encryption is outside the proof, as authorized. Public logits mean this experiment does not establish output privacy. |
| Complexity | Appendix reference counts evaluated at the actual N,K. All operation weights receive one common calibrated rational scale, followed by rounding the total. This preserves the relative reference schedule up to rounding. It is an approved reference-weight assumption, not newly audited exact operation counts for the modified graph. |
| Lottery, Exp. 1–2 | Every j=1..C is actually hashed as SHA256(G(parent,empty transactions) || framed ciphertext prefix || uint64_be(j)); threshold comparison is integer H<D. The last successful j is retained, matching Algorithm 1. |
| Chain | Winner publishes a linked block manifest with the full winning miner proof prefix. Proof/ciphertext references are content-addressed and all referenced files are retained. No propagation delay or forks. |
| Responses/collisions | Full winning prefix receives credit first, then one eligible response per other qid. Only duplicate qid work is a completed collision. Response eligibility/settlement is modeled without transaction serialization, fees or validation. |
| Execution clock | The bank's measured inference+proof intervals model independently provisioned virtual miners. Auxiliary encryption, hashing and event bookkeeping are executed during replay but have zero simulated delay. Jobs scheduled ahead but still active at adoption are canceled only in virtual time and are recorded separately. Host runtime is not chain time. |
| Exp. 3 | N,K,C,duration records are resampled from actual isolated calibration proofs. Large pools reuse benchmark templates under distinct qids. The default aggregate lottery samples the equivalent probability of at least one winning ticket; no ciphertexts or proofs are regenerated. `--collision-lottery literal` explicitly enumerates hashes against simulated binding bytes. |
| Approved omissions | Miner registration, query/block validity checks, transaction lists, transfers, encryption constraints inside ZKP, network propagation. No Byzantine-security or privacy conclusion follows from these experiments. |

## Complexity scale and calibration

For raw reference count C0 and fixed rational lambda=a/b, the lottery count is
`max(1, floor(lambda*C0 + 1/2))`. Calibration chooses lambda so that the median
profiling count maps to `--ticket-target 10000`. The scale is frozen before
held-out proof timing and all policy comparisons; miners cannot choose it.
`complexity_scale.json` records a,b, observed ranges, median and rounding rule.
`attempt.json` records both raw C0 and scaled C plus reference component counts.
Integer rounding changes each un-clamped value by at most half a ticket.

The difficulty is tuned against empirical renewal times and checked on a
separate set of simulation seeds. Experiments 1–2 use the frozen bank only for
the expensive inference/proof result and service duration; bindings, current
response encryption and every lottery hash are newly computed. Scaling C also recalibrates D; it is
not an attempt to preserve an old unscaled difficulty.

## Reports and reproducibility

- `campaign.json` records arguments, Python source digests, binary and schedule
  digests; `source_snapshot/` retains the sources and versioned patches.
- `dataset.json`, `prompts.json`, preparation profiles and `pool.json` preserve
  benchmark provenance and frozen rankings. Profiling randomness uses a
  separate key/domain from calibration and live mining.
- `prover/` retains the trusted verifier setup, decoding contract, handshake,
  stderr and stdout. The actual prover context is persisted and reused on restart. `scripts/deepprove_protocol.patch` applies on top of the
  pinned work-accounting patches. Zstd 0.13.3, already in DeepProve's dependency
  graph, is promoted to a runtime dependency for statement compression.
- Every bank record retains request/noise/uniforms, logits, response text, the
  actual proof, compressed public statement, and verification result. Every
  replay attempt retains its new request/noise/uniforms, ciphertext, hash
  result, VRF transcripts, N,K, raw/scaled C, recorded service time, and a
  pointer to the source proof record; replay proofs are deliberately not
  re-generated for each event.
- `executed.jsonl` includes replay work in abandoned/unfinished blocks.
  Adopted-block checkpoints retain logical completed/canceled classifications,
  query state and selection RNG state. Separate counters distinguish allocated
  attempts from replayed attempts, including abandoned blocks. Restart discards
  incomplete attempts and deterministically replays the unfinished block; it
  never relabels old work as a fresh measurement.
- Experiment 1 reports block and completed-pair mean, median, SD, CV, extrema,
  percentiles and a box-and-whisker figure with no points. The accompanying
  PoW comparison is explicitly an exponential-time model, not real PoW mining.
- Experiment 2 reports each policy/seed's block share, complexity share,
  blocks/hour, tickets/busy-second, mean K, and block yield relative to the
  uniform baseline. Pooled relative rates retain zero-win baseline seeds; intervals are omitted when too many bootstrap denominators are zero. The one-time inference-only profiling and full measurement-bank costs are reported separately. Bootstrap intervals resample independent chains; with
  only three seeds these remain exploratory. A zero-win baseline yields an
  undefined relative ratio, never an invented finite advantage.
- Experiment 3 writes per-seed/per-cell metrics and three heatmaps, with mean
  collision complexity percentage and sample SD. Failed cells are counted;
  adopted-only statistics are explicitly conditional on adoption.

If nearly every prompt reaches the cap, short/long profiling cannot identify a
meaningful length distinction. Report that limitation; do not interpret it as
resistance to cherry-picking. The shortest-prompt policy remains a distinct
N-based deviation. Historical artificial length labels are not used.

## Verification and practical limits

The bounded three-experiment campaign completed with 19 fresh verified CUDA proofs. A separate N=32,K=32 proof passed in 3.38 s inference plus 104.78 s proving; controlled early EOS and saved-setup restart checks also passed. A real small CUDA proof of perturbed stochastic GPT-2 passed the composite
verifier; the sampled logits and the final trace agreed exactly. Decoder tests
cover random interval selection, filtering, and rejection of inconsistent
uniforms, tokens, noise, or premature termination. The latest full Python suite
passed 236 tests, including pooling and artifact-migration regressions; three
Rust decoder/verifier tests passed. These are qualification results, not
statistical paper measurements.

Public proof statements are Zstd-compressed because uncompressed padded field
tensors are large. Proofs and encrypted logits still consume substantial disk.
The worker stops with a clear resumable error below 2 GiB free; it does not
delete unrelated experiment data. Check projected disk usage before the full
campaign. Literal ticket enumeration also makes Experiment 3 slower than its
old aggregate probability calculation.

Fresh runs need not reproduce identical wall-clock times or proof bytes. Those
can change the next proof-bound challenge and event ordering. The archive
retains the actual challenges, proof/ciphertext bytes and completion events
needed to audit the reported run. Deterministic experimental keys and recorded
seeds support independent reproduction of the method, not a promise of
bit-identical fresh physical campaigns. DeepProve's test SRS and the existing
sign-then-hash VRF are experimental cryptographic assumptions, not a production
setup or a cryptographic security evaluation.


## Independent checks

After calibration or a live run, `experiments/check_live_llm_protocol.py`
reverifies a saved composite proof and tests altered public bindings.
`experiments/audit_live_llm_run.py --input <campaign>` independently recomputes
raw/scaled complexity, proof-chain/query bindings and every lottery hash in
completed chain stages. `experiments/check_live_llm_backend.py` provides bounded
restart, EOS and context-64 tests. EOS qualification intentionally chooses a
uniform inside the EOS interval under a separate full-vocabulary decoder; this
is an edge-case test and is excluded from benchmark data.
