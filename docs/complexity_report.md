# GPT-2 online complexity experiment

Date: 2026-09-10.  The experiment used the checked-in DeepProve GPT-2
release binary, `openai-community/gpt2`, eight Rayon threads, and the default
quantisation/proving path.  The proving context was generated once (54,266
ms) and is deliberately absent from the cost below.  Verification and model
quantisation are also excluded.

DeepProve's supported `--sequence` mode uses a random prompt of `s-2` tokens
and proves a total context of `s`, so these runs have `K=2` and
`N=s-2`.  The online cost is `inference_time + prove_full`; `prove_full`
already includes witness generation and cryptographic proving.

| s | N | K | inference (ms) | prove_full (ms) | online (ms) |
|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 2 | 6,511 | 42,164 | 48,675 |
| 16 | 14 | 2 | 10,858 | 45,949 | 56,807 |
| 32 | 30 | 2 | 19,498 | 54,926 | 74,424 |
| 64 | 62 | 2 | 37,231 | 75,178 | 112,409 |

A least-squares quadratic gives the following practical reference schedule
(milliseconds):

\[
  C(N,K)=40{,}550.5+990.415\,s+2.07044\,s^2,\qquad s=N+K.
\]

Predicted costs at the four measured lengths are 48,606, 56,927, 74,364,
and 112,418 ms.  Training RMSE is 75.6 ms, MAE 64.4 ms, and the maximum
absolute relative error is 0.212%.  This is a satisfactory interpolation for
the measured regime, but it is not evidence for extrapolation beyond 64
tokens.  A leave-one-length-out check (refitting on three lengths and testing
the fourth) has maximum relative error 2.37%, a more realistic small-sample
estimate of interpolation error.

The attempted independent prompt-length sweep was rejected by DeepProve:
loading a proving context generated for one maximum context and invoking a
different `--max-context` panicked while looking up GeLU challenges.  It must
therefore use a separately generated context per maximum length (or a code
change to make contexts reusable).  The present data cannot identify separate
coefficients for N and K.  For this implementation, the defensible function
is consequently a function of total proved length `s=N+K`; a future sweep
with compatible contexts should test whether prompt and generated-token
lengths add any independent effect.

Raw measurements are in the gitignored
`.scratch/deep-prove/zkml/complexity_small.csv`.  The fitting command is:

```bash
python experiments/complexity_experiment.py \
  .scratch/deep-prove/zkml/complexity_small.csv
```

## Two-variable follow-up

The modified benchmark accepts `--pairs N:K,...`.  Two runs, each generating
one setup for maximum context 16, successfully proved 16 pairs with (N\ge2),
including different splits at the same total length.  The fitting utility can
combine the two pair CSVs:

```bash
python experiments/complexity_experiment.py \
  .scratch/deep-prove/zkml/complexity_pairs.csv \
  .scratch/deep-prove/zkml/complexity_pairs_more.csv
```

For these 16 observations, the recommended two-variable least-squares model
(milliseconds) is

\[
 C(N,K)=35128.6+1486.12N+2958.89K-60.1387NK-95.5248K^2.
\]

It has RMSE 1,401 ms, MAE 1,111 ms, maximum in-sample relative error 5.20%,
and leave-one-out maximum relative error 8.10%.  A total-length quadratic has
15.6% leave-one-out error on the same data, so retaining both variables is
justified at this context range.  Coefficients should not be extrapolated:
the sample is small, timings are noisy, and the negative interaction terms
are empirical corrections rather than asymptotic claims.

The benchmark now calls `compute_all_contexts`, whose setup is constructed for
the largest configured context and is valid for shorter pairs in the same
run.  A one-token prompt currently triggers an internal DeepProve slice panic,
so the experiment uses (N\ge2).  Reloading serialized setup in a fresh
process still fails because quantized model/setup compatibility is not yet
stable across processes; for now generate setup once per measurement run.

## CUDA smoke validation

The CUDA feature builds successfully when `/usr/local/cuda/bin` is added to
`PATH` (the toolkit was installed but not on the default shell path).  A
fixed-length GPT-2 trial on the RTX A6000 reported 46,690 ms inference and
59,055 ms proving for `N=4,K=4`; `nvidia-smi` observed about 46 GiB in use,
confirming GPU execution.  This tiny context was slower than the CPU trial
because kernel-launch and GPU PCS overhead dominate.  CUDA changes timing
coefficients and crossover points, but not the operation-count growth model.

A subsequent eight-trial CUDA run (contexts 7--16, fixed length) completed in
10.8 minutes including 1.4 minutes of setup.  Proving times were 53.8--65.1 s
per trial; the repeated `(N=3,K=4)` trials were 53.8--57.2 s.  Inference was
sub-millisecond to 27 ms after warm-up, while proving dominated total cost.
The resulting plot and fit are written under
`experiments/results/complexity/pairs_20260910_045013.*`.  The small sample is
for backend validation only; it is not a replacement for the full 115-trial
calibration.

## Full CUDA calibration

The full boundary-aware campaign used 100 distinct `(N,K)` pairs with
`N+K <= 64`, five repeated anchor cells, fixed EOS behavior, and a shared
setup generated for the largest context.  It completed 115 trials on the RTX
A6000 with `--cuda`; setup time, quantisation, and verification are excluded
from the online cost.  The raw results and 3-D plot are:

* `experiments/results/complexity_cuda/pairs_20260910_050530.csv`
* `experiments/results/complexity_cuda/pairs_20260910_050530.json`
* `experiments/results/complexity_cuda/pairs_20260910_050530.png`

Across all trials, inference averaged 2,352 ms (median 1,770 ms), while
`prove_full` averaged 98,090 ms (median 107,422 ms).  Thus the mean online
cost (`inference + prove_full`) was 100,442 ms, with a range of 52,412--119,267
ms; proving contributed 97.7% of that cost.  Verification averaged 1,754 ms
and is intentionally omitted from the complexity function.

The proof-only model

\[
  C_{\rm proof}(P)=c_0+c_1P+c_2P\log_2P,\qquad
  P=2^{\lceil\log_2(N+K)\rceil},
\]

has held-out RMSE 3.73 s, MAE 2.84 s, and maximum relative error 9.44%.
The combined five-term model (adding `a(N+K)+b(N+K)^2`) has 11.55% maximum
held-out relative error.  The separate inference fit is unstable on this
GPU run (negative quadratic coefficient and 255% maximum held-out error).
Inference timings are highly variable, including a 30,476 ms outlier; the
cause needs investigation before using this fit for calibration.  More
repetitions and an explicit warm-up/synchronization audit would help.  The
sharp cost increases at total lengths 17 and 33, immediately after 16 and
32, support including power-of-two proving buckets in the approximation.
