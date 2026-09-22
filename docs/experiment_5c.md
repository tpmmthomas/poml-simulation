# Experiment 5c: Miner Count and Query-Pool Size

## Rationale

Experiment 5 showed that miners selecting from the same pool of uniform-fee
queries can complete duplicate inference-and-proof work before a block is
adopted. Experiment 5b suggested that this waste is governed mainly by the
ratio of available queries to miners, $Q/M$, but its randomly sampled settings
were too irregular for direct cell-by-cell comparison. Experiment 5c therefore
uses a regular grid to locate the transition between collision-heavy and
collision-light regimes and to test how that transition changes with the
target block time.

## Procedure

The experiment reuses the calibrated discrete-event race engine from
[Experiment 5](experiment_5.md). Attempt durations are sampled with replacement
from 30 measured EZKL inference-and-proof times (mean 49.34 s). Every query has
fee 1, and each miner processes an independently shuffled permutation of the
query pool. A race ends when the first successful lottery attempt is adopted
or when all miners exhaust their query lists.

Miner count $M$ ranges from 10 to 10,000 and query-pool size $Q$ from 20 to
100,000, both on 1-2-5 logarithmic grids. Only the 75 cells satisfying $Q>M$
are simulated. Each cell is run for 100 deterministic seeds at target block
times of 300, 600, and 900 s, giving 22,500 attempted races. For each adopted
race, wasted work is the percentage of completed query-proof pairs that
duplicate an already completed query:

$$
W=100\times\frac{N-U}{N},
$$

where $N$ is the number of completed pairs and $U$ is the number of distinct
queries among them. Each heatmap cell reports the mean of $W$ across adopted
races and its across-seed sample standard deviation. Gray cells are infeasible
settings with $Q\leq M$.

## Findings

The heatmaps show a strong diagonal pattern, indicating that $Q/M$ is the main
control on wasted work. At a 300 s target, configurations with $Q/M=2$ average
roughly 50--56% wasted work, whereas those with $Q/M=100$ cluster tightly
around 2.5--2.9%. The pattern is similar across scale: for $Q/M=10$, the 300 s
cell means range only from 18.7% to 24.1% while miner count varies from 10 to
10,000.

Longer block targets shift the transition toward larger query pools because
more work completes before adoption. For the representative setting
$M=1{,}000$, $Q=2{,}000$ ($Q/M=2$), mean waste increases from
$51.43\%\pm26.72\%$ at 300 s to $71.99\%\pm24.25\%$ at 600 s and
$74.85\%\pm23.14\%$ at 900 s. At $M=100$, $Q=10{,}000$ ($Q/M=100$), it rises
from $2.61\%\pm2.53\%$ to $4.95\%\pm4.58\%$ and
$8.96\%\pm6.77\%$, respectively.

Waste approaches zero only when the query pool is much larger than the miner
population. Across cells with $Q/M=1{,}000$, mean waste is approximately
0.24--0.38% at 300 s, 0.53--0.70% at 600 s, and 0.78--0.92% at 900 s. The
smallest observed cell mean is 0.039% for $M=10$, $Q=100{,}000$, and a 300 s
target. Run-to-run variation is largest in the collision-heavy region: the
$M=1{,}000$, $Q=2{,}000$ examples above have standard deviations of about
23--27 percentage points. Thus, a cell mean describes a regime rather than a
typical outcome of every individual race.

## Interpretation and Limitations

The results suggest that provisioning queries in proportion to miner count is
not sufficient to avoid duplicate work. Under this model, $Q/M\approx100$
keeps mean waste below 10% throughout the tested block-time range, while ratios
near 2--5 remain collision-heavy. This is a simulation result for independent,
uniform query selection and immediate block adoption; it does not model fee
heterogeneity, coordinated query assignment, propagation delay, or miners
sharing one physical host.

Of the 22,500 attempted races, 22,413 adopted a block and 87 exhausted the
query pool without a winner. All failures occurred in eight small-pool cells;
the most affected was $M=10$, $Q=20$, at 900 s, where only 62 of 100 races
adopted. Failed races are retained in `runs.csv` but excluded from cell means,
so results for these edge cells are conditional on adoption and should be
interpreted cautiously.

The complete design and provenance are recorded in
[campaign.json](../experiments/results/exp5c_m_q_heatmap_100rep/campaign.json),
run-level observations in
[runs.csv](../experiments/results/exp5c_m_q_heatmap_100rep/runs.csv), and
cell-level statistics in
[summary.csv](../experiments/results/exp5c_m_q_heatmap_100rep/summary.csv).
The experiment can be rerun with
[`exp5c_m_q_heatmap.py`](../experiments/exp5c_m_q_heatmap.py).