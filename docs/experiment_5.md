# Experiment 5: Uniform-Fee Query Collisions

This document specifies the design, implementation, calibration, results, and
reproduction procedure for Experiment 5. The experiment asks:

> When all eligible inference queries pay the same fee, how much completed
> inference-and-proof work duplicates a query already completed by another
> miner before the first block is adopted?

The experiment varies the number of miners, target block time, and query-pool
size. It is a one-block, accelerated discrete-event experiment. It is not a
wall-clock execution of 1,000 EZKL provers on one machine, and it does not use
the repository's full account, mempool, or network stack.

The canonical final artifacts are:

- [campaign.json](../experiments/results/exp5_final_2026_08_12/campaign.json):
  complete design and provenance;
- [runs.csv](../experiments/results/exp5_final_2026_08_12/runs.csv): all 27,000
  run-level records;
- [summary.csv](../experiments/results/exp5_final_2026_08_12/summary.csv): the
  27 configuration-level results;
- [timing_calibration.json](../experiments/results/exp5_run_all/timing_calibration.json):
  real EZKL timing measurements and model hashes;
- [difficulty_calibration.json](../experiments/results/exp5_run_all/difficulty_calibration.json):
  fixed lottery thresholds and their verification;
- [exp5_seed_manifest.json](../experiments/exp5_seed_manifest.json): the frozen
  experimental grid and seed roots; and
- [exp5_uniform_fee_collisions.py](../experiments/exp5_uniform_fee_collisions.py):
  the reference implementation.

## 1. Experimental Design

### 1.1 Independent variables

The final campaign is the full factorial product of the following values:

| Parameter | Symbol | Values |
|---|---:|---|
| Miner count | $M$ | 10, 100, 1,000 |
| Target mean block time | $\tau$ | 300, 600, 900 s |
| Query-pool size | $Q$ | 1,000, 5,000, 10,000 |
| Fee per query | $f$ | 1 for every query |
| Outputs per query |  | 1 |
| Replicates per configuration |  | 1,000 |
| Configurations |  | $3 \times 3 \times 3=27$ |
| Total final races |  | 27,000 |

The first five replicates of every configuration retain full attempt-level and
per-miner query-order details. All 1,000 replicates retain run-level results.

### 1.2 Query selection

Each race initializes query IDs $0,\ldots,Q-1$. The queries are deliberately
equivalent except for identity: each has fee 1, one output, and the same cost
distribution. Query identity cannot affect the sampled attempt duration.

Each miner receives an independently shuffled uniform permutation of all $Q$
query IDs and processes its permutation in order. Consequently:

- a miner never attempts the same query twice in one race;
- different miners can select and complete the same query; and
- a miner stops if it exhausts all $Q$ queries without winning.

The implementation generates only the consumed prefix of each permutation,
using a lazy Fisher-Yates shuffle. Cross-miner selections are independent.

### 1.3 Miner and event behavior

Every miner has at most one active combined inference-and-proof attempt. At
virtual time zero, every miner starts its first attempt. An attempt duration is
sampled independently and with replacement from the 30 measured EZKL durations
in Section 3.1.

Pending completions are processed in this exact order:

1. ascending virtual completion time;
2. ascending run-seeded 64-bit tie key; and
3. ascending insertion sequence.

When an attempt completes, the coordinator:

1. records the completed query and whether it is a collision;
2. draws a uniform 256-bit integer $L$ for that miner;
3. declares a lottery win if and only if $L<D$; and
4. either immediately adopts the winning block or starts that miner's next
   query at the current virtual time.

The first winning completion is adopted immediately. There is no simulated
post-win propagation delay, fork choice, or competing block after this point.
An event with the same completion time but ordered after the winning event is
cancelled and reported as `adoption_preceded_tied_event`; it is not a completed
or partial attempt. Other active attempts are discarded partial work at the
adoption time.

If every miner exhausts its permutation without a win, the race is recorded as
`no_winner`. Its frozen seed is never replaced, it is excluded from collision
means, and the final campaign exits unsuccessfully. No such failure occurred
in the reported campaign.

## 2. Outcomes and Estimands

Let $c_q$ be the number of fully completed attempts for query $q$, including
the winning attempt, up to and including the adopted event. The number of
collisions in one race is

$$
C=\sum_q \max(0,c_q-1).
$$

If $N=\sum_q c_q$ is the number of completed query-proof pairs, the run-level
collision ratio is

$$
R=\frac{C}{N}
 =\frac{N-\left|\{q:c_q>0\}\right|}{N}.
$$

The primary configuration-level outcome is the arithmetic mean of $R$ over
adopted replicates:

$$
\bar R=\frac{1}{n}\sum_{i=1}^{n}R_i.
$$

The output also reports the sample standard deviation of $R$ and a pooled
ratio,

$$
R_{\mathrm{pooled}}=\frac{\sum_i C_i}{\sum_i N_i}.
$$

These two summaries answer different questions. The mean gives every race
equal weight; the pooled value gives longer races, which complete more work,
more weight. The mean collision ratio is the value shown in the heatmaps and
is the primary result.

Additional outcomes are adoption time, completed pairs, unique completed
queries, unfinished active attempts at adoption, elapsed and remaining virtual
time for those partial attempts, and tied completions cancelled by event
ordering.

## 3. Calibration

### 3.1 Real inference-and-proof timings

Thirty isolated calls to `run_inference_and_prove` were measured. Each call
performed EZKL witness generation and proof generation for the same fixed
input: 64 zero noise values followed by 64 zero conditioning values, packed as
a tensor of shape `[1, 2, 8, 8]`. Verification time was not included. Every
attempt ran in a fresh child process. All 30 attempts succeeded, all successful
measurements were retained, and no outlier was removed.

The exact duration sample, in seconds, was:

```text
49.6930837626569, 49.75114785414189, 50.43197503499687,
51.299318932928145, 47.94556702906266, 47.2570734359324,
48.30184564879164, 50.3551438646391, 48.82792083499953,
48.41572102392092, 48.52749679423869, 48.173486162908375,
48.75345072709024, 47.73184000514448, 47.59006975684315,
49.44425121601671, 49.25286992825568, 49.01214967016131,
48.10076447809115, 47.95771145820618, 47.99082546308637,
48.87539714388549, 49.86716907797381, 52.50339687196538,
54.18951359624043, 54.25783117674291, 50.50991499284282,
46.52818863512948, 49.75470555014908, 48.79590343683958
```

| Statistic | Value (s) |
|---|---:|
| Count | 30 |
| Mean | 49.33652445212938 |
| Sample standard deviation | 1.8211898535681037 |
| Minimum | 46.52818863512948 |
| Median | 48.85165898944251 |
| Maximum | 54.25783117674291 |

The reference timing host and software were:

| Field | Recorded value |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 5955WX 16-Cores |
| Logical CPUs | 32, restricted to CPU IDs 0-15 |
| Memory | 134,883,356,672 bytes |
| OS | Linux, kernel 6.8.0-124-generic, x86-64, glibc 2.35 |
| Python | 3.11.5 |
| EZKL | 23.0.5 |
| NumPy | 2.4.4 |
| Matplotlib | 3.10.8 |
| Pydantic | 2.13.3 |
| Cryptography | 46.0.7 |
| Thread limits | `RAYON_NUM_THREADS=16`, `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16` |

This calibration represents independently provisioned, homogeneous miners
with the measured isolated-node performance. It does not represent 10, 100, or
1,000 provers contending for the reference host's 16 selected CPUs.

### 3.2 Model and proof artifacts

The measurements used the repository's tiny U-Net with input shape
`[1, 2, 8, 8]` and EZKL hashed public input/output visibility. The exact
artifacts were:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `network.onnx` | 50,663 | `838b5ec6b04d40c3eb45310f2a978139309c88c8294b50de845425fb2865d794` |
| `network.onnx.data` | 45,120 | `4110552825f3867ee100840771f181dc7de991a74bd59d5c0f05bd7633a848ae` |
| `network.ezkl` | 389,500 | `46c81bb3944b4aee2dc92a83140aac9644a139ac5c21420174c040c2dfca4af3` |
| `settings.json` | 8,879 | `60acfd48b76a35ccfd1c6465463122d399b17b08529471672ec9672e847e3e9d` |
| `pk.key` | 9,633,665,811 | `2dcc82ce1deae57a3e2dea95246c40ba49b27e38c5009d761720e059c818e987` |
| `vk.key` | 3,542,471 | `1990538f640f3435111d76f8c750ec23c0591452fa26596fdb3f50df533b5138` |
| `kzg.srs` | 67,109,124 | `d1a1655b4366a766d1578beb257849a92bf91cb1358c1a2c37ab180c5d3a204d` |

The recorded EZKL settings use `check_mode: UNSAFE`; therefore, this experiment
uses EZKL to measure the cost of the repository's configured proving path, not
to claim production-security parameters.

The current model export and EZKL setup scripts use unseeded random model
initialization and random calibration inputs. Running `scripts/setup_model.sh`
again can therefore produce different artifact hashes. Exact reproduction of
the reported timing calibration requires the archived artifacts above.
Reproducing the experimental method with newly generated artifacts instead
requires collecting a new timing calibration and reporting it as a new
campaign.

### 3.3 Lottery difficulty

For each $(M,\tau)$ pair, the per-attempt threshold was computed once from the
empirical mean attempt duration $E[T]=49.33652445212938$ s:

$$
D=\left\lfloor 2^{256}\frac{E[T]}{M\tau}\right\rfloor,
\qquad
P(\text{win per completed attempt})=\frac{D}{2^{256}}.
$$

The same threshold was used for all three query-pool sizes at a given
$(M,\tau)$. Each threshold was checked with 1,000 separate-seed races using
$Q=10{,}000$. The check measured adoption times only: collision outcomes were
not inspected, and thresholds were not retuned after verification.

| $M$ | $\tau$ (s) | Win probability | Difficulty threshold $D$ | Verification mean (s) | Relative error |
|---:|---:|---:|---|---:|---:|
| 10 | 300 | 0.016445508150709793 | `0x0435c5d7ac63f147cc8b30e25cb8ed919a01dc6f7e6204927d745b35d7bcffaa` | 317.596117 | 5.865% |
| 10 | 600 | 0.008222754075354896 | `0x021ae2ebd631f8a3e64598712e5c76c8cd00ee37bf3102493eba2d9aebde7fd5` | 626.421924 | 4.404% |
| 10 | 900 | 0.005481836050236597 | `0x016741f28ecbfb17eed9104b743da485de009ecfd4cb56db7f26c911f2945538` | 942.872470 | 4.764% |
| 100 | 300 | 0.0016445508150709793 | `0x006bc6fbf7a39820c7a784e36fac17c1c299c93e597033a83fbed5ebc8c61991` | 328.094849 | 9.365% |
| 100 | 600 | 0.0008222754075354896 | `0x0035e37dfbd1cc1063d3c271b7d60be0e14ce49f2cb819d41fdf6af5e4630cc8` | 644.630788 | 7.438% |
| 100 | 900 | 0.0005481836050236598 | `0x0023ecfea7e132b597e281a125395d4096334314c87abbe2bfea474e98420885` | 945.089077 | 5.010% |
| 1,000 | 300 | 0.00016445508150709793 | `0x000ac719325d28d013f726e38b2acf2cf9dc2db96f58052a6cc648979413cf5b` | 309.591482 | 3.197% |
| 1,000 | 600 | 0.00008222754075354896 | `0x0005638c992e946809fb9371c59567967cee16dcb7ac02953663244bca09e7ad` | 661.325563 | 10.221% |
| 1,000 | 900 | 0.00005481836050236598 | `0x000397b310c9b8455bfd0cf683b8efb9a89eb9e87a72ac6379976d87dc069a73` | 905.783207 | 0.643% |

All 9,000 verification races adopted a block.

## 4. Randomness and Reproducibility

The final seed design was frozen before final collision outcomes were
collected. The final root is:

```text
poml-exp5-final-frozen-2026-08-12-v1
```

For domain string `final-run`, miner count $M$, target $\tau$, pool size $Q$,
and zero-based replicate $r$, a 64-bit run seed is derived as follows:

```text
material = root + NUL + "final-run" + NUL + str(M) + NUL
           + str(tau) + NUL + str(Q) + NUL + str(r)
run_seed = unsigned_big_endian(SHA256(UTF8(material))[0:8])
```

The 27,000 resulting final seeds are unique. Given a run seed, separate seeds
are derived with the same function for each miner and each random domain:

| Domain | Ordered seed parts | Use |
|---|---|---|
| `query-order` | miner ID | lazy query permutation |
| `duration` | miner ID | empirical duration draws |
| `lottery` | miner ID | 256-bit lottery draws |
| `tie-order` | miner ID | 64-bit equal-time ordering keys |

The reference implementation uses Python 3.11.5's `random.Random` for all four
streams, including `randrange`, `choice`, and `getrandbits`. An independent
implementation must reproduce these PRNG operations and the event order above
to reproduce the exact rows rather than only the distribution.

Configuration and run IDs are SHA-256 hashes of canonical JSON: keys are
sorted, separators are `,` and `:`, and JSON is ASCII encoded. A run ID hashes
`{"config_id": config_id, "replicate": r, "seed": run_seed}`.

## 5. Final Results

The campaign ran at simulator revision
`79615522d04c4e2ffff8c51bbf5a734b797010a3` from a clean worktree. It completed
on 2026-08-12 with campaign ID
`ec39e359e0ebbcede5ec544dec7d9c2f5126ac7a109df7144e2a674bc01b5802`.
All 27,000 requested races adopted a block; there were zero failed races.

The complete results are below. Ratios and secondary quantities are rounded to
six decimal places for readability; [summary.csv](../experiments/results/exp5_final_2026_08_12/summary.csv)
is the canonical source for full-precision values.

| $M$ | $\tau$ (s) | $Q$ | Mean collision ratio | SD | Pooled ratio | Mean adoption (s) | Mean completed pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 300 | 1,000 | 0.027747 | 0.031961 | 0.052399 | 341.125239 | 64.562 |
| 10 | 300 | 5,000 | 0.005201 | 0.010662 | 0.010856 | 313.409041 | 58.955 |
| 10 | 300 | 10,000 | 0.002617 | 0.006363 | 0.005234 | 321.414009 | 60.567 |
| 10 | 600 | 1,000 | 0.050384 | 0.050830 | 0.096939 | 605.971706 | 118.291 |
| 10 | 600 | 5,000 | 0.011501 | 0.016096 | 0.021594 | 613.290042 | 119.756 |
| 10 | 600 | 10,000 | 0.005734 | 0.010073 | 0.011375 | 605.566980 | 118.240 |
| 10 | 900 | 1,000 | 0.073425 | 0.066752 | 0.134485 | 904.664096 | 178.845 |
| 10 | 900 | 5,000 | 0.015484 | 0.017661 | 0.030779 | 893.682142 | 176.584 |
| 10 | 900 | 10,000 | 0.008737 | 0.010708 | 0.017419 | 985.507362 | 195.076 |
| 100 | 300 | 1,000 | 0.223606 | 0.165917 | 0.377879 | 332.891784 | 624.488 |
| 100 | 300 | 5,000 | 0.056328 | 0.053284 | 0.108351 | 329.676778 | 619.055 |
| 100 | 300 | 10,000 | 0.029032 | 0.027971 | 0.055448 | 325.018332 | 610.051 |
| 100 | 600 | 1,000 | 0.344301 | 0.226259 | 0.551464 | 626.023477 | 1,219.147 |
| 100 | 600 | 5,000 | 0.108910 | 0.095685 | 0.203284 | 659.276530 | 1,285.846 |
| 100 | 600 | 10,000 | 0.055822 | 0.053391 | 0.110147 | 626.836771 | 1,219.883 |
| 100 | 900 | 1,000 | 0.417313 | 0.241161 | 0.626039 | 864.450417 | 1,702.209 |
| 100 | 900 | 5,000 | 0.142335 | 0.118637 | 0.257893 | 890.935054 | 1,755.815 |
| 100 | 900 | 10,000 | 0.080444 | 0.070346 | 0.146965 | 917.528912 | 1,809.907 |
| 1,000 | 300 | 1,000 | 0.653395 | 0.258630 | 0.846258 | 295.318553 | 5,468.395 |
| 1,000 | 300 | 5,000 | 0.359000 | 0.223904 | 0.560955 | 342.728260 | 6,453.253 |
| 1,000 | 300 | 10,000 | 0.206345 | 0.157632 | 0.358953 | 302.427297 | 5,616.648 |
| 1,000 | 600 | 1,000 | 0.794759 | 0.221545 | 0.927197 | 652.997223 | 12,730.453 |
| 1,000 | 600 | 5,000 | 0.498667 | 0.254412 | 0.705852 | 626.012664 | 12,183.256 |
| 1,000 | 600 | 10,000 | 0.343366 | 0.221313 | 0.545436 | 617.322725 | 12,012.622 |
| 1,000 | 900 | 1,000 | 0.845099 | 0.193991 | 0.948685 | 940.680821 | 18,560.559 |
| 1,000 | 900 | 5,000 | 0.590188 | 0.260067 | 0.789414 | 956.174420 | 18,873.818 |
| 1,000 | 900 | 10,000 | 0.440862 | 0.252063 | 0.659569 | 972.394606 | 19,214.471 |

### 5.1 Main observations

Within every fixed $(M,\tau)$ pair, increasing $Q$ reduces the mean collision
ratio. Within every fixed $(\tau,Q)$ pair, increasing $M$ increases it; and
within every fixed $(M,Q)$ pair, increasing $\tau$ generally increases it by
allowing more completions before adoption.

The smallest observed mean collision ratio was 0.002617 for
$(M,\tau,Q)=(10,300,10{,}000)$. The largest was 0.845099 for
$(1{,}000,900,1{,}000)$. Thus, under the simulated conditions, duplicate
completed work is negligible with few miners and a large pool, but dominates
when many miners draw independently from a small uniform-fee pool over a long
block interval.

Pooled ratios are consistently larger than equal-run means. High-collision
races tend to run longer and complete more attempts, so weighting every
completion equally gives those races more influence. This is why the estimand
must be stated when reporting a collision ratio.

The mean number of discarded unfinished attempts is close to $M-1$ in every
cell, as expected from one active attempt per non-winning miner at immediate
adoption. Same-time events ordered after adoption are reported separately in
the canonical summary.

## 6. Reproduction Procedure

### 6.1 Reproduce the archived campaign exactly

Exact reproduction requires the simulator revision, model/proof artifacts,
timing artifact, difficulty artifact, seed manifest, and Python PRNG behavior
specified above.

1. Check out commit `79615522d04c4e2ffff8c51bbf5a734b797010a3` in a clean
   worktree and use Python 3.11.5.
2. Install the package and test dependencies:

   ```bash
   python -m pip install -e ".[dev]"
   ```

3. Place the archived model/proof files in `model/` and verify the SHA-256
   digests in Section 3.2. Do not regenerate them for an exact reproduction.
4. Verify the supplied calibration and seed files against the artifact IDs and
   hashes recorded in `campaign.json`.
5. Run the experiment unit tests:

   ```bash
   python -m pytest tests/test_exp5_uniform_fee_collisions.py
   ```

6. Run the frozen final campaign into a new, empty output directory:

   ```bash
   python experiments/exp5_uniform_fee_collisions.py run-final \
       --timings experiments/results/exp5_run_all/timing_calibration.json \
       --calibration experiments/results/exp5_run_all/difficulty_calibration.json \
       --seed-manifest experiments/exp5_seed_manifest.json \
       --artifacts-dir model \
       --output-dir experiments/results/exp5_reproduction
   ```

The program intentionally refuses a non-empty output directory and rejects a
dirty worktree or a simulator/model/calibration mismatch. Compare the produced
`runs.csv`, `summary.csv`, and `campaign.json` with the archived files.

### 6.2 Reproduce the method with a new PoML implementation

If the archived EZKL artifacts are unavailable, or if a different PoML model
or prover is used, perform the following protocol and report the resulting
timing calibration as part of a new campaign:

1. Fix one valid query input whose identity cannot affect computational cost.
2. On a documented, resource-controlled reference miner, measure 30 successful
   complete inference-and-proof attempts. Retain every success and record all
   failures; do not remove successful outliers.
3. Use the arithmetic mean of those 30 durations to compute the nine
   thresholds with the formula in Section 3.3. Use an unsigned 256-bit uniform
   lottery and strict comparison $L<D$.
4. Verify each $(M,\tau)$ threshold with 1,000 disjoint-seed, $Q=10{,}000$
   races. Record adoption-time statistics, but do not inspect collision
   outcomes and do not retune thresholds.
5. Freeze the final grid, seed root, seed derivation, and failure policy before
   collecting final outcomes.
6. Run 1,000 replicates for each of the 27 cells using the event algorithm in
   Section 1.3. Sample measured durations independently with replacement and
   independently of query identity.
7. Retain at least every run's seed, status, adoption time, completed count,
   unique-query count, collision count, collision ratio, partial-attempt count,
   and winning miner/query. Retain attempt-level records for an announced,
   outcome-independent subset.
8. Report both the equal-run mean ratio and pooled ratio, together with failed
   run counts and enough provenance to identify the simulator, model, proof
   system, host, duration sample, and random-number implementation.

A reproduction with a new timing distribution should be expected to match the
qualitative dependence on $M$, $\tau$, and $Q$, not the exact archived CSV
rows.

### 6.3 Regenerate figures

The three heatmaps share a common color scale from zero to the largest mean
collision ratio across all 27 cells. Regenerate them with:

```bash
python experiments/exp5_uniform_fee_collisions.py plot \
    --summary experiments/results/exp5_final_2026_08_12/summary.csv \
    --output-dir experiments/results/exp5_final_2026_08_12/figures
```

This produces PNG and PDF heatmaps for targets 300, 600, and 900 seconds.

## 7. Output Interpretation and Checks

The following invariants should hold for every adopted run:

```text
collision_count = completed_pairs - unique_queries_completed
collision_ratio = collision_count / completed_pairs
0 <= collision_ratio < 1
winner is included among completed pairs
```

For each configuration, `runs_requested = runs_adopted + runs_failed`. The
reported final campaign additionally satisfies:

```text
configurations = 27
runs_requested per configuration = 1,000
total runs = 27,000
total failed runs = 0
detailed replicates per configuration = 5 (replicates 0 through 4)
```

The `details/<config_id>/runs.json` files contain the complete per-miner query
completion lists for retained replicates. The corresponding `attempts.csv`
files distinguish completed attempts, unfinished work cancelled at adoption,
and same-time completions ordered after adoption.

## 8. Scope and Limitations

The experiment isolates duplicate query selection under uniform fees. It does
not model heterogeneous miner speeds, correlated prover times, strategic query
selection, changing fees, multiple outputs, query arrivals, account balances,
fee reservation or debit, shared-hardware contention, proof verification cost,
network propagation, competing blocks, or multi-block dynamics.

The 30-value empirical duration distribution is small and host-specific, and
the final simulator treats its draws as independent across miners and attempts.
The results therefore quantify collision behavior under the stated homogeneous
one-block model. They should not be interpreted as direct measurements of a
geographically distributed production PoML network.