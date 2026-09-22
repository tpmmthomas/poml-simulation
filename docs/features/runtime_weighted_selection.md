# Runtime-weighted Experiment 2

`experiments/fit_runtime_selection.py` fits operation weights using an existing
verified bank and reruns Experiment 2 with the weights frozen. It never creates
a prover, invokes GPT-2, generates a proof, or updates output-length profiles.
The original campaign and its 88 inference/proof measurements remain unchanged.

## Method

The explanatory vector is exactly `reference_counts(N, K)["combined"]` from
`poml_sim.gpt2_work`, using **realised** output length K. The target is the
recorded inference-plus-proof time, excluding verification and auxiliary host
work. Every existing pair is retained; neither block shares nor policy labels
enter the regression. Zero operation weights are allowed; negative weights are
forbidden. No extra intercept or prompt-specific correction is added.

For reference counts x, coefficients beta minimize
`mean((x / s @ beta - measured_seconds)^2) + lambda * sum(beta^2)`, subject to
`beta >= 0`. Each column scale s is its maximum over the public supported
domain `N >= 2, K >= 1, N + K <= 64`. These structural scales use no timing
observations. A training-fold mean is unsuitable for rare MSM-size counts:
a column constant in one fold can jump at a held-out padding boundary.

Four outer folds hold out entire prompt hashes, keeping all stochastic
replicates and capped variants together. Within each training fold, four inner
prompt folds select lambda from `1e-6, 1e-4, 0.01, 0.1, 1, 10` by squared error.
The uniform-weight control fits one global multiplier on the same training rows.
The final schedule selects lambda by four-fold validation on the full bank and
then fits all 88 rows. This final replay is an **in-bank sensitivity experiment**;
the nested held-out prediction results are reported separately.

The 47 operation features have rank six across the 17 measured N,K shapes.
The regression therefore does not uniquely identify physical per-operation
latencies. The nonnegative regularized solution is an empirical reference
schedule for this prover/hardware/workload, without a portability guarantee.
For the current bank, nested held-out MAPE falls from 10.119% to 3.951%, and
MAE from 9.099 to 3.805 seconds. Short-cap coverage remains limited: 24 added
pairs from eight base prompts, alongside 64 original pairs from 32 prompts.

## Deterministic mining calculation

Seconds-per-operation weights are rounded into nonnegative integers at
`10^15` units per second. Each replay computes their dot product with the
reference count vector using integer arithmetic, applies one frozen rational
scale, and rounds half upward with a minimum of one ticket. The scale maps
the bank median to 10,000 tickets. Difficulty is recalibrated to a 300-second
target with the same renewal simulator and checked using independent seeds.

`ProfiledWork` logs the original uniform raw count, weighted count, final C,
and schedule digest. The digest enters genesis. Archived proof records keep
their original complexity and are validated under their original scale;
their stored C is never substituted for the new mining calculation.

The repeat uses the source campaign's four miners, four policies, three seeds
and 50 blocks per policy/seed: 600 blocks. Each attempt still recomputes current
bindings, encryption, and every SHA-256 ticket. Recorded inference/proof time
is its simulated service time. Source proofs continue to attest the original
measurement challenge, as documented in the [protocol guide](llm_live_protocol.md).
Four independent chain jobs can run concurrently; per-chain RNGs, output files
and checkpoints are independent of thread scheduling.

## Command and artifacts

From the configured workspace:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/fit_runtime_selection.py \
  --input /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/profiled-replay-varied-k \
  --output /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/runtime-weighted-selection \
  --workers 4
```

Use `--resume` for the same output and unchanged sources. `--fit-only` stops
after fitting and validation. Source proof artifacts are referenced in place;
new ciphertexts and replay logs are retained under the NAS output directory.

| Artifact | Content |
| --- | --- |
| `runtime_weights.json`, `operation_weights.csv` | Frozen weights, scale, normalization, candidate errors and nested-fold details |
| `validation_predictions.csv`, `held_out_predictions.{png,pdf}` | Every pair's actual time, fold, and predictions |
| `preparation/source_calibration.json` | Original bank-validation scale and provenance |
| `preparation/calibration_inputs.json`, `calibration.json` | Recomputed costs and independently checked difficulty |
| `campaign.json`, `environment.json`, `source_snapshot/` | Exact configuration, source/input hashes and executable source copies |
| `cherry_pick/` | Chain checkpoints, literal-hash attempt records, policy/seed metrics and figure |
| `comparison.{csv,json}`, `report.md` | Uniform versus fitted schedule, block shares, ticket throughput and relative yield |

For a full-range prediction plot, per-shape errors and a counterfactual that
changes C on the identical historical selected attempts, run:

```bash
.venv/bin/python experiments/report_runtime_selection.py \
  --input /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/runtime-weighted-selection
```

This produces `runtime_predictions.{png,pdf}`, `shape_diagnostics.json` and
`historical_attempt_reweighting.json`, with a retained postprocessing source.
Use `runtime_predictions` for the figure; it includes the entire prediction
range and an output-length color key. No chain work is repeated.
After the selection stage finishes, it also writes `effective_work_rates.json`:
ticket rates per full chain second and time lost to cancellation, alongside the
completed-pair ticket/time metric. These distinguish weighting mismatch from
block-boundary effects.

The command `experiments/audit_live_llm_run.py --input <output>` independently
reconstructs weighted C, query/proof bindings, ciphertext prefixes and every
literal lottery. This reads existing proofs without generating any new ones.

The symmetric block-share expectation is 25%. Improved ticket/time equality
does not force observed shares to equal 25%: finite samples, completion races
and cancellation can still produce differences. Three chains per policy give
limited seed-level statistical precision; use the retained intervals and
throughput diagnostics rather than claiming universal selection resistance.

## Completed comparison

The 600-block rerun completed with 8,761 scheduled attempts (6,961 completed,
1,800 canceled). All 84,415,573 literal tickets, source-proof/ciphertext bindings
and weighted complexities passed the independent audit. Original input and
mining source hashes were unchanged. New inference/proof pairs: **zero**.

| Policy | Uniform-weight block share | Fitted-weight block share | Original completed-ticket rate / uniform policy | Fitted completed-ticket rate / uniform policy |
| --- | --- | --- | --- | --- |
| Uniform | 32/150 = 21.33% | 35/150 = 23.33% | 1.0000 | 1.0000 |
| Profiled-short | 30/150 = 20.00% | 39/150 = 26.00% | 0.8666 | 0.9851 |
| Profiled-long | 42/150 = 28.00% | 30/150 = 20.00% | 1.0802 | 1.0149 |
| Shortest-prompt | 40/150 = 26.67% | 35/150 = 23.33% | 0.9058 | 1.0063 |

The weighting-related imbalance is much smaller. Reweighting the identical
historical selected attempts, without changing selections or times, also
reduces the long-policy excess from 8.02% to 1.07%. This separates the direct
effect of weights from changes in query history and fresh lottery outcomes.

The result does not establish that all selection benefits vanish. Including
time spent on interrupted attempts, fitted effective ticket rates differ from
uniform by +1.03% for profiled-short, -0.30% for profiled-long and +2.16% for
shortest-prompt. Profiled-short loses 9.52% of its chain time to cancellation,
compared with uniform's 11.77%. Weighting completed jobs cannot eliminate this
completion-time effect.

Profiled-short's observed pooled block yield is still 1.179× uniform, with the
existing paired seed-bootstrap interval [1.055, 1.351]. Report this residual
result rather than claiming no advantage. The interval has only three
independent seed-level chains per policy; lottery variation and scheduling
effects cannot be cleanly separated at this sample size. The 47 fitted
coefficients are also not uniquely identified primitive latencies.

Validation: 63 focused regression/unit tests passed; Ruff lint and formatting
passed. The README's obsolete description of per-attempt GPU proving was
corrected to describe the measurement-bank replay path.
