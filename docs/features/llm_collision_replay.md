# Collision-only Experiment 3 and recovery of saved campaigns

> Historical note: the benchmark redesign rejects the old single-text bank.
> A subsequent protocol audit also corrected canonical credit to include the
> entire winning miner prefix; see [the audit](llm_protocol_simulation_audit.md)
> and [current campaign guide](llm_benchmark_profiling.md). Earlier heatmaps
> must not be reused as results of that corrected benchmark design.

The archived `overnight-20260915-062744/wasted_work` run counted every
completed non-winning pair as wasted. It did not credit unique completed
queries that could be submitted as responses, so its reported 96–99% means
are not collision waste. Aggregate cell CSVs cannot reveal which query IDs
were duplicated; the simulation must be replayed from the saved seeds/traces.

`experiments/run_llm_experiments.py wasted-work --replay-from <directory>`
restores the archived design and writes a separate corrected campaign.
The [experiment guide](llm_poml_experiments.md) gives the complete command.
No real inference/proofs, Experiment 1, or Experiment 2 need rerunning.

## Accounting and simulation

One completed pair is useful per exact query identifier. The canonical winner
takes priority for its query; the first completed pair for each other query
is credited as an eligible response. Only additional completed pairs of those
same queries are wasted, weighted by their own `C(N,K)`. Responses have ideal
eligibility/inclusion; no network, capacity, or fee model is asserted.
Unfinished work is excluded. True pool exhaustion is reported separately.

The old simulator silently stopped at 100,000 completions. Corrected runs have
no event cap. For an archived replay, the first 100,000 events still reproduce
the old adoption counts, mean completed/unfinished complexity, no-response
waste mean/SD, and difficulty. Each cell must match those saved aggregates
before corrected results are accepted. A replay mismatch fails the run.
The number of trials that previously reached this cap is separately reported.

To speed replay, the runner caches the few distinct complexity values and the
fee bounds per prompt length. Empirical trace order, multiplicities, random
draws, pool composition and seed derivation are preserved. Parallel workers
operate on independent cells. The existing pooled `(K, duration)` distribution
is retained; assigning a query's `N` does not make the sampled time a new
measurement of that exact `(N,K)` pair.

## Artifacts and migration

- `runs.csv`: every seed's canonical, response, collision, and total work,
  number of completed/unique queries, block time, and termination reason.
- `cells.csv`: conditional mean/SD, adoption/exhaustion counts and replay checks.
- `manifest.json`: seeds, CLI, input/source hashes, sampling and response policy.
- `summary.json` and `report.md`: labelled equal-cell averages and limitations.
- Three annotated green-to-red heatmaps, each exported as PNG and PDF.
- `COMPLETED`: created after CSVs, summaries and all figures succeed.

Results are flushed per cell and existing nonempty output directories are
rejected. Original results and original paper figures are preserved. The
replacement paper section uses `plots/llm_collision_work/` for its new figures,
with an explicit placeholder until the rerun is available. Existing model
dependencies are sufficient; this change adds no dependency.

Regression tests check zero waste for unique responses, weighted duplicates,
canonical priority, unchanged event streams, caching equivalence, event-limit
versus actual exhaustion, archived replay validation, and output preservation.
