# Paper results from the completed LLM campaigns

The query-pool subsection and heatmaps are now superseded by the
[uniform-fee full-grid rerun](llm_uniform_fee_scaling.md). It restores the
original homogeneous method, first-completion duplicate accounting and
mean ± sample SD annotations. The archived results below describe the earlier
query-conditioned campaign; Experiments 1–2 retain their existing sources.

The active evaluation in `.scratch/PoML_paper_draft/main.tex` now includes
`7_experiments_llm.tex`. The original diffusion section, `7_experiments.tex`,
is preserved. The abstract, introduction's experimental contribution and
reference-complexity appendix describe the measurement-bank replay and
runtime-weight comparison. The initial paper update was report-only; the
subsequent uniform-fee revision reran its CPU simulation using the existing
LLM measurements.

## Sources and interpretation

Both source directories are under
`/mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/`:

| Source | Paper use |
| --- | --- |
| `profiled-replay-varied-k` | Experiment 1; unit-weight Experiment 2 control; Experiment 3 |
| `runtime-weighted-selection` | Fitted-weight Experiment 2; nested prompt-held-out runtime predictions |

The 88 verified pairs cover 32 WikiText-2 prompts and 56 prompt/cap variants.
There are 304 separate inference-only profiling draws. Realised output K is
used for complexity; estimated K is only a selection-policy input. Almost
all pairs hit their caps, so the selection study concerns heterogeneous
declared output budgets, not unconstrained EOS prediction.

Experiments 1 and 3 retain the unit-weight schedule. Only Experiment 2 has a
completed fitted-weight comparison. The regression is nonnegative and uses
only reference counts derived from realised N,K. The final mining weights
use all 88 pairs; nested prompt-held-out predictions are a separate check.

| Experiment | Reported result | Important boundary |
| --- | --- | --- |
| 1 | 50 blocks; mean interval 295.26 s; 546 completed attempts replay recorded times | PoW reference is 50 exponential draws, not a Bitcoin implementation; one chain cannot establish a variance advantage |
| 2 | Held-out MAPE 10.12% → 3.95%; long-policy completed-ticket-rate excess 8.02% → 1.49% | Short-policy block yield remains 1.179× uniform, CI [1.055, 1.351]; only three chains per policy |
| 3 | Mean of 75 cell means: 12.18%, 18.04%, 22.59% at 300/600/900 s | Collision-only completed complexity; 16 exhausted trials excluded from affected cell means; canceled work excluded |

Current protocol text uses one hash with a complexity-adjusted threshold.
Archived Experiments 1–2 instead enumerate C virtual SHA-256 tickets. Their
winning laws agree under independent uniform hashing up to integer threshold
rounding; their hash inputs and sample paths differ. Experiment 3 samples the
aggregate probability directly. Hashing adds no simulated time. Replayed
proofs attest their original bank challenges, not each virtual attempt's new
challenge. These differences are stated in the evaluation.

## Reproduce the figures without repeating experiments

From the repository root:

```bash
.venv/bin/python experiments/plot_llm_paper_results.py \
  --baseline /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/profiled-replay-varied-k \
  --weighted /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/runtime-weighted-selection \
  --collisions /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/uniform-fee-scaling \
  --output /home/thomas/poml-sim/.scratch/PoML_paper_draft/plots
```

With `--collisions`, the driver validates all 360 full-grid cells and 36,000
uniform-fee races. The source is recorded separately in the figure statistics.
Without that option it reproduces the earlier 225-cell collision archive.
The driver validates completion markers, sample counts, independent selection
seeds and chain lengths, exposure-pooled yield, and grid/trial coverage for
the selected collision source. It saves SHA-256 hashes of numerical inputs and the
plotting source in `plots/llm_paper_figure_statistics.json`.

| Figure files, PDF and PNG | Content |
| --- | --- |
| `llm_replay_boxplots` | Block intervals and completed-pair service times; boxes and whiskers with mean/median lines, no individual points |
| `llm_runtime_fit` | All 88 nested held-out predictions, including the low unit-weight predictions |
| `llm_runtime_selection` | Unit/fitted completed-ticket throughput and relative block-yield confidence intervals |
| `llm_collision_{300,600,900}s` | Three red-to-green panels sharing a 0–100% color scale and mean ± sample SD labels; full grid with the new collision source |

Heatmap means and sample SDs are rounded to whole percentages for readability;
the exact values remain in `cells.csv`. The section places three panels side
by side at text width. In the new source, every completion after the first for
a query is a duplicate; the archived source uses winning-prefix priority.

## Build and verification

```bash
cd /home/thomas/poml-sim/.scratch/PoML_paper_draft
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/mnt/nas/thomas_work/poml-sim/.scratch/paper_build/llm_results main.tex
```

The existing `plots/PoML.drawio.png` had a text metadata chunk exceeding
pdfTeX's PNG reader limit. Its metadata was compressed without changing
decoded pixels or embedded draw.io content; both were checked for equality.
The original is backed up under the NAS `paper_build/before_experiment_results/`.

The paper directory contains `main.pdf` for the full compiled manuscript and
`llm_experiments_preview.pdf` for the current four-page evaluation excerpt
(manuscript pages 24–27, including adjacent text at its boundaries).

The report tests cover unequal-duration exposure pooling, duplicated seeds,
incomplete chains/grids, exhaustion and response credit. The current
uniform-fee/report checks pass 20 tests; Ruff lint/format pass, and the full
paper compiles. Query-pool references resolve. Existing missing liveness and
appendix references, the duplicated `eq:online-depth-bound` label, and
earlier-section overfull boxes remain outside this update. Build products
and rendered page inspections stay under NAS.
