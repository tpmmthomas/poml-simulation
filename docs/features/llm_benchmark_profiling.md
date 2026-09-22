# Benchmark-backed PoML experiments with measured length profiles

**Historical audit/design:** superseded for current execution by [the live protocol implementation](llm_live_protocol.md). The descriptions below concern the earlier measurement-bank replay code.

The three paper experiments now use distinct WikiText-2 benchmark prompts,
real GPT-2 sampling, and separately measured CUDA DeepProve proofs. Experiment
2 estimates each prompt's expected output length from independent profiling
runs. The single-base-text campaign and its assigned length categories are
superseded; its numerical conclusions must not be used for this design.

## Workload alignment pending

The [protocol audit](llm_protocol_simulation_audit.md) found that this candidate
uses token-sampling randomness and sentence stopping, whereas the paper's LLM
instantiation uses prompt-embedding perturbation and EOS/fee-derived stopping.
The pipeline is smoke-tested, but **do not use an overnight run as evidence for
the paper's specified model until those contracts are aligned**. The command
below runs the explicitly described candidate workload; it is not a complete
PoML protocol implementation.

## Run the candidate campaign

From the repository root, inside tmux, using the existing `.venv` and compiled
CUDA prover:

```bash
.venv/bin/python experiments/run_llm_benchmark_campaign.py \
  --device cuda:1 \
  --output experiments/results/llm_poml/benchmark-campaign \
  --paper-output .scratch/PoML_paper_draft/plots/llm_benchmark
```

Use a fresh output directory. To continue an interrupted campaign, repeat the
same command with `--resume`. Profiling rollouts and completed proof batches
are checkpointed. A partially finished proof batch is retried; completed
batches are validated and reused. Incomplete simulation stages are archived
and rerun deterministically. Source/argument changes invalidate a trace-build
resume. The campaign writes `COMPLETED` only after all stages and figures.

The default bank has 128 prompts, 16 profiling seeds and four independent
held-out evaluation seeds per prompt: 2,048 profiling rollouts, 512 evaluation
rollouts, and 512 measured proof trials plus 16 batch warm-ups. At 75 seconds
per proof, proof computation alone takes about 11 hours; setup, verification,
and longer contexts add time. This is a planning estimate, not a deadline.
`--queries 64` halves the bank cost; it also reduces benchmark diversity.
`--evaluation-replicates 8` improves the empirical per-prompt distribution but
doubles proving cost. `--profile-replicates` changes inference cost only.

`--smoke` uses a small bank and grids. To test all simulation/reporting stages
without new proofs, supply `--trace-bank <completed-benchmark-bank> --smoke`.
All stages display tqdm progress; exact commands are in `commands.jsonl`.
GPU indices follow PyTorch's visible-device indexing; the builder maps the
selected device to DeepProve's CUDA visibility.

## Dataset decision

Use the already downloaded `Salesforce/wikitext`, `wikitext-2-raw-v1`, test
split. Select 128 distinct non-heading rows with a fixed seed and one contiguous
random token window from each. Balance prompt lengths over 8, 16, 24 and 32
tokens. No token padding, repeated base sentence, or invented output labels
are used. Save exact text, token IDs, row indices, token offsets, dataset
fingerprint/cache hash and token hashes. These are benchmark-derived language
continuation requests, not a benchmark accuracy evaluation.

Other cached datasets were inspected:

| Candidate | Finding | Decision |
| --- | --- | --- |
| WikiText-2 | Natural language continuation; enough distinct cached rows | Primary workload, compatible with base GPT-2 |
| Resisting Correction | Labels identify correction/next-word choices, not generated length | No reliable length label; possible later external workload |
| BIG-bench repeat/copy logic | 32 tasks, with specified copying targets | Too small as primary; target length does not guarantee base GPT-2 follows instructions |
| PIQA, ARC, HellaSwag, LAMBADA | Mostly choices or short expected answers | Available extensions; labels do not establish free-generation length |
| LongBench (new candidate) | Long-context task evaluation | Poor fit to the current audited 64-token context |

Primary sources: [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext),
[Inverse Scaling release](https://github.com/inverse-scaling/prize/tree/main/data-release),
[BIG-bench repeat/copy logic](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/repeat_copy_logic),
[LongBench](https://github.com/THUDM/LongBench).
[Response Length Perception and Sequence Scheduling](https://arxiv.org/abs/2305.13144)
studies response-length prediction in another model/workload. It motivates
measuring predictability here; it does not validate a GPT-2 predictor.

## Length profiling and generation contract

For each fixed prompt q, generate R_p=16 outputs with independently derived
profiling seeds. Freeze `mean(K_profile[q])` as the adversary's estimate.
Only then generate R_e=4 held-out outputs with a disjoint seed namespace.
Policies cannot inspect the held-out K before selecting a query. Mining
attempts resample the held-out traces with replacement, so the large campaign
remains a discrete-event simulation. There is no claim of fresh neural-network
execution at every simulated attempt.

The public workload is **sentence completion**: temperature 1, EOS or the
first generated token ending in `.`, `!`, `?` (ignoring trailing spaces/quotes),
or containing a newline; hard cap K=32. Every prompt uses this same rule.
The actual stopping-token list and its digest are saved. It is a simple
punctuation rule, not a semantic sentence parser: abbreviations may stop it.
N+K stays within the audited 64-token reference schedule. `--stop-rule eos`
is available as a control, but its predictiveness must be assessed separately.

This contract was selected after an EOS-only pilot with 32 prompts, cap 16,
and eight profiling seeds produced identical profiling means of 16 for every
prompt (98.44% of held-out generations hit the cap). That experiment could
not meaningfully distinguish short and long prompts.

A separate CUDA sentence-completion pilot used 32 prompts, 16 profiling and
8 evaluation seeds. Its held-out Spearman correlation was 0.692; the shortest
profiled quartile averaged 7.83 held-out tokens versus 22.05 for the longest;
MAE was 3.89 tokens and 16.80% hit the cap. These are pilot diagnostics, **not
mining results or evidence against selection attacks**. The full campaign
must use its own saved diagnostics. A flat or poorly predictive profile cannot
support a strong negative conclusion.

## Real measurements and their scope

Each held-out prompt/replicate gets its own CUDA DeepProve inference/proving
measurement. No timing is reused merely because two trials share (N,K).
The builder checks CUDA backend, verified proof, exact input token IDs, trial
position, N, K and timing CSV agreement. It rejects mismatches. The trace
clock is DeepProve `inference_time + prove_full`; setup, quantization, warm-up
and verification are excluded from the mining clock and retained in raw logs.
A common largest-context warm-up gives every batch the same setup bound.

**Proof limitation:** the current DeepProve benchmark executes its own
fixed-length decoder on the same input prompt for the sampled K. It does not
constrain the separate stochastic GPT-2 output-token sequence or the paper's
challenge-dependent embedding perturbation. These are measured workload
proxies for a simulation, not a full PoML protocol implementation.

Complexity remains `complexity_for` → `reference_counts` → sum of operation counts,
using `config/gpt2_reference_schedule.json` and the existing unit gas weights.
The profiling mean never replaces the realised K in C(N,K). Serial throughput
calibration uses sum(C)/sum(T), not the biased average of per-attempt C/T.
The nominal block target is 300 seconds; observed target tracking is reported.

## Experiment 1: honest chain

Use 16 homogeneous, independently provisioned miners, uniform selection and
50 adopted blocks. Miners work without replacement in each pool generation;
if exhausted, another generation of benchmark-backed request IDs becomes
available. Work/time from earlier generations remain in the same interval.
After a win, cancel active attempts and begin the next block. Network delay,
forks, and concurrent contention on the calibration GPU are not simulated.

Report adopted blocks, mean, sample SD, median, quartiles, min/max, attempts,
completed/unfinished complexity, target deviation and calibrated probability.
The boxplot has no individual points; whiskers use 1.5 IQR and the full range
is reported numerically. There is no matched PoW campaign in this design.

## Experiment 2: profiled query selection

One miner deviates while the other 15 select uniformly. All use identical
hardware rates, the same theoretical C, stopping rule, and fixed difficulty.

| Policy | Selection from remaining requests | Question |
| --- | --- | --- |
| uniform | Uniform random | Honest baseline |
| profiled-short | Smallest measured profiling mean K | Can short expected generation earn blocks faster? |
| profiled-long | Largest measured profiling mean K | Can longer work yield more tickets efficiently? |
| shortest-prompt | Smallest public N | Can reduced input-processing work help? |

Profile ties are broken randomly with reproducible seeds, independently of N.
Shortest-prompt ties use query ID. This is a finite-pool profiling adversary:
it pays for repeat access to known prompts; it is not a predictor of previously
unseen prompts. No reference answers, synthetic category labels, or held-out
outputs enter selection.

Run 20 seed groups × 500 intervals per policy, with sufficient demand.
Preserve paired seed-group comparisons. Report:

- prediction MAE, rank correlation, cap fraction, per-prompt means/SDs, and
  held-out separation of the shortest/longest profiled quartiles;
- **blocks per simulated hour and its ratio to uniform**, plus completed
  complexity per elapsed second and selected completed K/duration;
- block share, completed-complexity share and their ratio; the last ratio is
  secondary because proportional reward per work does not establish no speed gain;
- block-time/adoption statistics, incomplete work, and profiling inference cost;
- steady-state yield and yield with the one-off measured profiling cost
  charged to each seed group's campaign duration.

Use seed-group percentile-bootstrap 95% intervals, conditional on this finite
prompt/trace bank (10,000 resamples in the runner; 20,000 in the paper renderer).
These intervals do not include uncertainty from drawing a new benchmark bank
or new proof measurements. Do not treat a non-significant result as proof of
security against all inputs, alternative algorithms, or hardware.

## Experiment 3: collision-only wasted complexity

Preserve the original accelerated strategy: M=10..10,000, Q=20..100,000 on
1–2–5 grids; only Q>M (75 cells), targets 300/600/900 seconds, 100 seeds/cell.
Each miner uses an independent uniform permutation without replacement. There
is no pool replenishment within this finite-pool experiment. Report exhaustion
explicitly; cell means condition on adoption and include zero-waste trials.

Large pools resample the real benchmark prompt templates into distinct request
identifiers. They are **not** 100,000 independently measured prompts. Each
request retains its source prompt's joint (N,K,C,T) distribution, avoiding
mismatched pooled duration and input length. Collisions mean the same request
identifier, including its request identity, not simply identical prompt text.

Credit the entire winning miner’s proof prefix and one eligible RESPONSE_QUERY per other
completed query ID. Only additional completed attempts for those same IDs
are wasted: 100×(completed − canonical − eligible-response C)/completed C.
Unfinished attempts are reported separately and excluded. All eligible
responses are assumed includable; no fee/capacity/propagation model is added.
Write three heatmaps (mean ± sample SD), displayed side by side in the paper.
A mean of heatmap cells is not a network-wide waste rate.

## Artifacts, implementation and migration

- `src/poml_sim/llm_benchmark.py`: prompt extraction, profiles, seed domains,
  held-out diagnostics; `llm_simulation.py`: selection and event simulation.
- `experiments/build_real_llm_trace_bank.py`: checkpointed real rollouts/proofs.
- `experiments/run_llm_benchmark_campaign.py`: one-command campaign orchestration.
- `experiments/run_llm_experiments.py`: calibration and the three simulations.
- `experiments/plot_llm_paper_results.py`: report-only vector/PNG figures.

The bank saves exact prompts/outputs/seeds, frozen profiles, diagnostics,
source/model/package/GPU provenance, dataset/schedule/input hashes, every
proof ledger and timing row, batch commands/logs, and completion markers.
Each experiment saves CLI/config/source digests, per-seed/block data, reports
and figures. Keep the complete output directory together for reproduction.

Legacy single-prompt trace banks are deliberately rejected. All three paper
experiments need a fresh benchmark campaign; replaying old traces cannot
repair their input design. Synthetic fixtures require `--synthetic-smoke`
and are unavailable for Experiment 2. Historical collision-accounting notes
remain useful explanations, not commands for the current benchmark bank.

For the full protocol mapping, including the aggregated Bernoulli lottery and
omitted bindings/encryption/validation, read the
[protocol audit](llm_protocol_simulation_audit.md).
