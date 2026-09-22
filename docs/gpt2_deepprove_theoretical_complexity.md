# Theoretical GPT-2 + DeepProve complexity for PoML

**Superseded for the audited maximum-context-64 configuration by the
[reference-work report](features/deepprove_work_accounting.md).** The source and
instrumentation audit corrects the attention count to dense rectangular work,
accounts for the inclusive decoding loop and repeated K/V requantization,
removes the speculative FFT term, and separates unpadded proof reductions from
the five padded proof profiles. The formulas below remain the historical
proposal and should not be used as the finalized definition.

Date: 2026-09-10.  This report derives a public reference work function for
the GPT-2 path used by PoML.  It follows the protocol's online boundary:
VRF-derived randomness, model inference, proof generation, and encryption are
online; CRS/setup generation and verification are excluded.

## Protocol boundary and variables

The paper's LLM protocol defines a tokenized prompt of length `N`, exactly
`K` generated transitions (or EOS earlier), and one randomness value per
prompt token.  The public complexity should therefore be evaluated on the
declared `(N,K)` and a setup/padding bound `S`.  To make cost predictable, the
query should commit to `K_max` before mining, and EOS should count as an
explicit transition.  Otherwise two queries with the same nominal `K` can
execute different work.

For the DeepProve implementation, `run_elements` performs the prompt pass,
one-token cached passes while generating, and a final full-sequence pass to
produce the trace that is proved.  Thus its inference path is not equivalent
to a serving engine that only performs one prefill plus K cached decodes.
`prove_full` then builds and executes a local GKR/logup proof graph over that
trace; it includes witness/claim generation and commitment openings.

## GPT-2 architecture count

The checked-in `openai-community/gpt2/config.json` has

| symbol | value |
|---|---:|
| layers `L` | 12 |
| hidden width `d` | 768 |
| heads | 12 |
| head width | 64 |
| FFN width `f` | 3072 = 4d |
| vocabulary `V` | 50,257 |
| maximum context | 1,024 (reduced by benchmark bound) |

Each transformer block contains two layer norms, fused QKV projection,
causal attention, output projection, GeLU, and two FFN projections.  Ignoring
elementwise operations and biases, one token's dense multiply-add count is

\[
  A=L(3d^2+d^2+df+fd)=12L d^2=84{,}934{,}656.
\]

The tied vocabulary projection contributes

\[
  A_V=Vd=38{,}597{,}376
\]

per logits row.  Therefore the linear work per token is approximately
`123,532,032` multiply-adds.  Causal attention contributes `2Ld` operations
per query/key position pair, or `18,432` per pair for this model.

For an ordinary KV-cached implementation, a useful architecture formula is

\[
 W_{\rm serve}(N,K)=
 (A+A_V)(N+K)
 +2Ld\left[\frac{N(N+1)}2+NK+\frac{K(K+1)}2\right].
\]

The bracket is prompt self-attention plus all decode attention.  This is a
device-independent operation count; it does not claim that every kernel
performs exactly one hardware instruction per multiply-add.

For DeepProve's actual runtime, add the final full trace pass and the cached
loop.  A close count is

\[
 W_{\rm DP\text{-}inf}(N,K)=
 2(A+A_V)(N+K)
 +Ld\left[N(N+1)+2\sum_{t=1}^{K}(N+t)+(N+K)(N+K+1)\right].
\]

The first term counts the generation loop and final trace pass.  The second
counts prompt attention, cached one-token attention, and final causal
attention.  Layer norms, GeLU, reshapes, token sampling, and memory copies
are linear in `N+K` and can be represented by a fitted linear residual.

## DeepProve proof work

DeepProve constructs a graph node for each model operation and proves the
trace by backward claim propagation, lookup-table proofs, sumchecks, and a
polynomial commitment opening.  The source exposes separate timings for
`prove_claims`, `prove_commitment_opening`, and `prove_full`; `prove_full` is
the online proving quantity and already includes witness generation.

The exact cost depends on padded tensor shapes and the polynomial commitment
parameters.  Let `S` be the setup maximum and let `R(S)` be the largest
polynomial evaluation domain used by the padded trace.  A faithful reference
form is

\[
 W_{\rm proof}(N,K;S)=
 a_F F(N,K;S)+a_Q Q(N,K;S)+a_R R(S)\log R(S)+a_M M(N,K;S).
\]

`F` counts field operations in layer sumchecks, `Q` lookup rows (GeLU,
requantisation, softmax, argmax and masks), `R log R` covers FFT-like domain
work, and `M` covers commitment/opening operations.  The constants are fixed
by the selected DeepProve/PCS implementation.  Setup generation is excluded,
but `S` and its padding rule must be public because they affect these counts.

For a first approximation, all proof terms except attention-shaped witness
polynomials can be grouped into a linear/quadratic padded-length function:

\[
 W_{\rm proof}\approx b_0+b_1P+b_2P^2+b_3P\log_2P,
 \qquad P=\operatorname{nextpow2}(N+K)\le S.
\]

This is a schedule approximation, not a theorem about optimized provers.
When measurements reveal jumps at powers of two, replace it with a table
indexed by `P`.

## Combined PoML reference function

The recommended device-agnostic function is a weighted primitive count:

\[
 C_{\theta,\Pi}(N,K;S)=
 W_{\rm DP\text{-}inf}(N,K)+W_{\rm proof}(N,K;S)
 +T\,w_{\rm VRF}+w_{\rm enc},
\]

where `T=N` for the paper's one-noise-vector-per-prompt construction,
`w_VRF` is a fixed reference VRF evaluation/proof unit, and `w_enc` is one
fixed encryption unit.  These auxiliary terms are linear and are negligible
relative to GPT-2 inference/proving at the measured sizes, but retaining them
makes the protocol definition complete.  If witness generation recomputes
the model and the measured proof count already includes it, do not add that
inference a second time.

## Comparison with measurements

The mixed-pair data (16 observations, context at most 16) gives a linear
calibration of the architecture count to wall time with correlation 0.93 and
`R²=0.865` (RMSE 1.78 s).  This is consistent with the count predicting the
growth direction, while the remaining error reflects proof-domain steps,
memory effects, and timing noise.  The fitted wall-time polynomial is a
useful local interpolation, but its coefficients are not portable between
devices.

## What can be omitted

For GPT-2 at these sizes, VRFs, hashing, encryption, serialization, layer
norms, GeLU, and token sampling are lower-order linear terms.  They may be
omitted from a first public schedule if a fixed additive safety margin is
charged.  Setup and verifier time must remain excluded as required by the
protocol.  Padding/domain effects must not be omitted: they can dominate the
step-to-step changes in proving time.

## Recommended experiment to finalize constants

Use the new `--pairs` mode with a fixed maximum `S`, force exactly `K`
transitions, and collect per-pair `F`, `Q`, `R`, `M`, trace length, and the
three proof timing components.  Fit the primitive-count model on one device,
then validate growth ratios and residuals on another device.  Publish the
integer work-unit coefficients and a padded-length bucket table if the domain
steps are visible.  Publish wall-clock conversions only as informational
benchmarks.
