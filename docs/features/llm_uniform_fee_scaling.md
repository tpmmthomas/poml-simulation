# Query-pool scaling under uniform fees with LLM measurements

The paper's LLM subsection now follows the original diffusion experiment's
question, paragraph order and analysis: how the ratio of pending queries to
miners controls duplicate completed work at 300, 600 and 900 second targets.
The original `.scratch/PoML_paper_draft/7_experiments.tex` is preserved.

## Method and changes from the previous LLM draft

`experiments/run_llm_uniform_fee_scaling.py` implements the homogeneous
occupancy experiment beside the existing experiment drivers. Its input is the
88 verified GPT-2/DeepProve records in
`experiments/results/llm_poml/profiled-replay-varied-k/preparation/timings.jsonl`.
No new inference or proofs are needed. The new CPU simulation is saved
separately under `experiments/results/llm_poml/uniform-fee-scaling/`.

| Aspect | Original design restored for the LLM experiment |
| --- | --- |
| Workload | All identifiers have the same pooled empirical distribution. Draw a measured `(duration, raw complexity)` pair independently with replacement for each attempt; preserve the pairing. |
| Work units | Unscaled unit-weight reference counts for each measured N,K, checked against the recorded operation ledger. The previous rounded ticket scale is unnecessary. |
| Query selection | Each homogeneous miner independently draws a uniform permutation without replacement, with at most one active attempt. No replenishment. |
| Fees | The same fee for every eligible identifier; no fee-based selection or rejection. |
| Difficulty | `floor(2**256 * E[T] / (M * tau * E[C]))`, the original throughput calibration expressed per complexity unit. |
| Lottery | A seeded Bernoulli draw with probability `1 - (1 - D/2**256)**C` at each completion. No new proof, ciphertext or ticket hash generation. |
| Adoption | Adopt the first winning completion immediately. Seeded random tie ordering follows the original experiment; cancel pending attempts. |
| Waste | Credit the first completion of each identifier. Sum the complexity of every later completion, including a duplicate winner, divided by all completed complexity. |
| Grid | The original 1–2–5 M/Q grids, now including **all 120 cells** at each target, as requested. 100 independent deterministic seeds per cell: 36,000 races. |
| Statistics | Arithmetic mean of per-race percentages and sample SD (`ddof=1`), conditional on adoption. Exhaustion is retained separately; zero-waste adopted races remain included. |
| Figures | Original `RdYlGn_r` palette: green at 0%, red at 100%, mean and ±SD in every measured cell, shared color scale. Asterisks mark excluded exhausted races. |

Pooled measurements deliberately make identifiers exchangeable: an identifier
does not retain a specific benchmark text or shape. This is a homogeneous
workload abstraction, not a claim that all actual prompts have identical cost
distributions. Independently drawing a duration and a complexity would lose
their measured correlation, so entire pairs are resampled together.

The former LLM experiment assigned a fixed template to each query and credited
the winning miner's whole prefix before eligible responses. Those results
remain archived. First-completion credit matches the original occupancy metric;
it does not simulate which response a blockchain eventually pays for. With
constant complexity it reduces exactly to `100 * (N-U)/N`. With variable
complexity, replacing an earlier pair with a later winning pair changes the
metric and is intentionally avoided here.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  experiments/run_llm_uniform_fee_scaling.py \
  --input experiments/results/llm_poml/profiled-replay-varied-k \
  --output experiments/results/llm_poml/uniform-fee-scaling \
  --workers 8
```

Use a fresh output directory for a new run. Append `--resume` after an
interruption; input measurements, simulation sources, seeds and design must
match. Per-cell JSON checkpoints are atomic. The campaign retains source
snapshots, input hashes and a measurement-bank copy, plus `runs.csv`, `cells.csv`,
`summary.json` and PDF/PNG figures. `COMPLETED` is written only after validation
and figure generation succeed. Generated data remains gitignored.

To regenerate only the paper's three heatmaps:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/plot_llm_paper_results.py \
  --collisions experiments/results/llm_poml/uniform-fee-scaling \
  --collisions-only --output .scratch/PoML_paper_draft/plots
```

This writes `llm_collision_{300,600,900}s.{pdf,png}` and
`llm_uniform_fee_statistics.json`, leaving other experiment figures alone.
For a complete figure rebuild, add the same `--collisions` argument to the
baseline/weighted command in [the paper-results guide](llm_paper_results.md).
Without it, that historical command still reproduces the archived
query-conditioned results; it should not replace the current paper's heatmaps.

Regression checks cover count-metric equivalence at constant complexity,
variable-cost duplicates including a winning duplicate, exhaustion, canceled
attempts, deterministic seeds, complete diagonal/lower-triangle coverage,
input validation, safe resume, sample SD and the plotted annotations/palette.
There are no new dependencies or changes to Experiments 1–2.

## Completed results

| Target | Races | Adopted | Exhausted |
| --- | --- | --- | --- |
| 300 s | 12,000 | 11,999 | 1 |
| 600 s | 12,000 | 11,954 | 46 |
| 900 s | 12,000 | 11,876 | 124 |

All 360 cells have adopted observations. Exhaustion affects 25 target-specific
cells, corresponding to 13 distinct `(M,Q)` configurations. The empirical pair
duration is 95.39 ± 8.76 seconds. At 300 seconds, mean waste ranges from
39.53–42.48% for Q/M=2, 10.56–14.18% for Q/M=10, and 1.39–1.74% for Q/M=100.
Every tested Q/M=100 cell remains below 5% at all three targets. The paper
retains the original examples `(1000,2000)` and `(100,10000)` and updates their
means/SDs directly from the new run.

An independent check recomputed every mean and sample SD from all 36,000
rows, verified the first/duplicate complexity and query-count partitions,
and confirmed that each failed race completed exactly M×Q attempts.
The 20 focused regression checks and Ruff lint/format checks pass. The full
267-test suite passed 266 tests; its legacy EZKL E2E test missed the fixed
180-second mining deadline while Rayon was capped at four threads and the
campaign was also running. Repeating that unchanged test separately with
normal Rayon settings passed in 181.70 seconds including shutdown. Thus all
267 tests have passed across the full run and that isolated retry.
The updated paper compiles; existing missing references in the liveness and
appendix sections and the duplicated `eq:online-depth-bound` label are outside
this subsection. Its own equations, figure and subfigure references resolve.
