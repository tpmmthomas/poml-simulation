# Critical analysis of the PoML complexity function

## What the current experiment establishes

The benchmark measures wall-clock `inference_time + prove_full` for one GPT-2
implementation, one quantisation path, one CPU configuration, and one proof
system.  It is useful evidence about the honest cost curve on that machine.
It does not define a device-agnostic complexity measure.  The fitted
coefficients absorb thread scheduling, memory bandwidth, cache effects,
parallelism, Rust allocator behaviour, and the current DeepProve implementation.
The two-variable polynomial also has negative interaction coefficients and an
8.1% leave-one-out error, which is a warning against interpreting its
coefficients as portable physical costs.

The experiment has further confounders:

* DeepProve runs an autoregressive loop and can stop on EOS, so two nominally
  equal `(N,K)` requests can execute different numbers of iterations unless EOS
  is disabled or treated as an explicit fixed token.
* The setup bound and padding policy affect polynomial and lookup domains.  A
  short query proved under a larger maximum context may pay work determined by
  that bound rather than its realized length.  The bound must therefore be a
  recorded input to the function, or setup must be defined at the query's
  exact length.
* Prompt token values can alter control flow, cache behaviour, and (with
  quantisation calibration) numerical paths.  Random prompts are appropriate
  for averaging, but a public schedule needs a fixed token-generation rule and
  a stated aggregation such as median or an upper quantile.
* `prove_full` includes witness generation.  Adding a separately measured
  inference time is correct only when that inference is not already part of
  the witness path.  Otherwise the sum double-counts work.
* Batch size, number of chunks, parallel executor, precision, model revision,
  and proof parameters all change the curve.  These are protocol parameters,
  not noise to silently average away.

## A better device-agnostic definition

Define PoML complexity as a deterministic amount of reference work, not
seconds:

\[
  W_{\theta,\Pi}(N,K;B,S)=W_{\rm inf}(N,K;B)+W_{\rm prove}(N,K;S,B),
\]

where `B` is the declared execution mode (batching, precision and model
configuration) and `S` is the setup/padding bound.  The protocol assigns fees
and lottery weight from `W`; devices only determine how quickly they realize
those work units.  A reference schedule can still publish a calibration in
seconds for convenience, but that calibration is separate from the consensus
definition.

For a transformer with hidden width `d`, feed-forward width `f`, and `L`
layers, a useful inference work model counts scalar multiply-adds.  With KV
caching it has the form

\[
 W_{\rm inf}=L\left[c_0d^2(N+K)+c_1df(N+K)+c_2d\left(N^2+NK+K(K-1)/2\right)\right]
       +c_{\rm vocab}V(N+K).
\]

The constants count projections, MLPs, attention, and vocabulary work in the
fixed reference implementation.  If the proof path performs one full causal
pass rather than KV-cached decoding, replace the attention term with
`c2*d*(N+K)^2`; do not mix the two execution models.  This equation gives the
growth rate from architecture and token counts without depending on a GPU.

For proving, count proof-system primitives rather than milliseconds:

\[
 W_{\rm prove}=a_{\rm fld}F+a_{\rm lookup}Q+a_{\rm fft}R\log R
                 +a_{\rm commit}M,
\]

where `F` is field operations, `Q` lookup rows, `R` the relevant polynomial
domain size, and `M` commitment/opening work.  DeepProve already exposes
related counters (`prove_claims`, `prove_commitment_opening`, proof size and
layer timings); instrumenting exact field/lookup/FFT counts would make these
terms explicit.  The coefficients `a_*` are proof-system constants and can be
calibrated once per implementation, while the resulting growth in `N,K` is
portable across devices.  Setup generation remains outside online work, but
the setup bound `S` and padding rule must be part of the public function.

## Recommended validation strategy

1. Fix model revision, tokenizer, quantisation, precision, batch/chunk mode,
   proof parameters, and a maximum setup bound.  Make EOS an explicit token
   and force exactly `K` transitions.
2. Generate a space-filling design of `(N,K)` pairs, with repeated fixed
   token sequences per cell.  Include boundary lengths around powers of two
   to expose padding steps.  Record actual trace length and setup bound.
3. Collect primitive counts and wall-clock timings separately.  Use timings
   only to estimate a local calibration from work units to seconds; do not
   put device-specific coefficients in the PoML rule.
4. Fit nested nonnegative models: architecture FLOP counts, proof primitive
   counts, and a small residual term for padding buckets.  Validate on held-
   out pairs and on a second device.  Report relative error of growth ratios
   such as `W(2N,2K)/W(N,K)`, not only absolute milliseconds.
5. If proof counts show sharp domain transitions, publish a deterministic
   bucketed schedule indexed by padded lengths.  A bucket table is preferable
   to a polynomial that predicts below the actual cost or has negative
   coefficients.

The current timing campaign should therefore be retained as an engineering
benchmark, but the recommended PoML definition is a reference primitive-count
function with explicit setup/padding parameters.  The two-variable timing
polynomial can serve as a provisional interpolation for GPT-2 context lengths
up to 16, not as the consensus complexity function across devices.

CUDA does not alter this conclusion.  It changes the mapping from reference
work units to seconds and can change which terms dominate.  In a smoke trial,
GPU launch and proof-system overhead made an 8-token GPT-2 job slower than the
CPU run; larger contexts may cross over.  CPU and CUDA timings should therefore
be reported as separate calibrations of the same work schedule.
