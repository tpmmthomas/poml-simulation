# Deterministic GPT-2 / DeepProve complexity

Date: 2026-09-10. Reference rule: `gpt2-deepprove-work-3`.

## Recommendation

Use a **weighted vector of specified reference operations**, with a formula for
inference and a five-entry structural table for the padded part of proving.
Keep the weights symbolic. Do not fit seconds, or interpret a MAC, a field
operation, and an elliptic-curve operation as having the same cost.

There is no remaining regression for \(a,b,c_0,c_1,c_2\). Architecture and
counting rules determine the inference coefficients; instrumented proof plans
determine the five integer vectors. Once published, these vectors are fixed
parts of the definition. A future implementation may justify a new version,
rather than changing the gas charged by the existing version.

For this bounded implementation, define

\[
\boxed{C_{\theta,\Pi,\Sigma,\mathbf w}(N,K)
=\mathbf w^\mathsf T\!\left[
  \mathbf I(N,K)+\mathbf e_{\rm axis}D(N+K)
  +\mathbf B_{2^{\lceil\log_2(N+K)\rceil}}\right].}
\]

Here \(\Sigma\) freezes the reference implementation, tokenizer, quantization, setup,
padding, termination rule, and operation-counting rules. \(\mathbf I\) and
\(D\) are given below. Each \(\mathbf B_P\) is an explicit integer vector;
it is not a timing coefficient. \(\mathbf e_{\rm axis}\) selects the
axis-reduction field-MAC component. Unsupported configurations must use a
different schedule version rather than extrapolating this table.
All vectors use the same 47-component basis: ten inference families, one axis
family, and the 36 proof families tabulated below; absent families have count
zero. Each component retains its own symbolic weight.

For the paper, identify \(N\) with the authenticated input-token count and
\(K\) with the authenticated number of output transitions. Select \(\Sigma\)
and the same weights for everyone through the public model/proof registration.
Positive integer weights for charged operation families make the complexity a
positive integer, as required in `4_protocols.tex`. Rational weights can instead
be scaled to integers using one public denominator. The implementation allows
zero weights to explicitly omit a family; it never supplies default weights.
A registered schedule must still give positive cost to every allowed query;
the all-zero choice is not a valid PoML complexity function.

This is an exact **definition of reference gas**. Its relationship to optimized
physical effort is an approximation. Agreement of the structural counters does
not establish a universal timing error or a lower bound on an adversary.

**Important correction to the earlier proposal:** even with the model, protocol
and setup fixed, literal proof-operation counts need not be exactly determined
by N and K. DeepProve's LayerNorm chooses a right shift from the activation
values, which can change lookup decomposition and the resulting proof plan.
The fixed five-profile schedule approximates this small variation; it is not
claimed to recover every value-dependent execution exactly.
For a concrete observed counterexample, two verified executions at
\((N,K)=(2,2)\) used 387 and 388 MSMs of size 4096. The shape-dependent
inference vector was identical. This is why the recommendation separates an
exact public gas definition from an approximate description of each execution.

## Fixed configuration and scope

| Item | Audited configuration |
| --- | --- |
| Model | GPT-2 small, `openai-community/gpt2`; 12 blocks, width 768, 12 attention heads, MLP width 3072, vocabulary 50257 |
| DeepProve | `9d1a53e2ef49ffa2c902b8689cd3c58057a4e662`, with the supplied patch |
| dp-crypto | `22e8e93cd0f94a7638616a1ae22190feb0a6b275`, with the supplied patch |
| Proof configuration | Local, nondistributed, default GPT-2 quantization with `ZKML_BIT_LEN=12`; BN254 and HyperKZG |
| Setup capacity | 64, shared by every trial in a campaign |
| Allowed lengths | Integers \(N\ge2\), \(K\ge1\), \(N+K\le64\) |
| Termination | Fixed length; EOS does not stop generation or truncate the verified links |
| Inference scope | Prompt prefill, autoregressive generation, and full trace reconstruction |
| Proof scope | Online proof generation, including claims and commitment opening |
| Excluded | Model loading/quantization/setup/precomputation, verification, ledger writing |

The benchmark's setup seed and a digest of its serialized verifier context are
recorded. The context digest identifies the concrete setup within this run;
serialization need not be canonical across independent processes. A production
registration should publish the actual quantized weights, scales, setup
artifacts and canonical hashes, not merely a seed or a mutable model repository
name.

Prompt token IDs are drawn uniformly from the vocabulary using recorded seeds;
the continuations are generated greedily with EOS stopping disabled. These are
controlled random-token checks, not a representative natural-language workload.

The patches preserve the earlier variable-length support, fixed-EOS verification,
and GPT-2 embedding/output-weight parsing correction. They also allow the valid
edge case \((N,K)=(63,1)\). The calculator deliberately does not support \(N=1\),
which has a special role in the existing inference driver.

## Inference derivation

Let

\[
S=N+K,\quad
U=N^2+NK+\frac{K(K+1)}2+S^2,\quad
R=NK+\frac{K(K-1)}2.
\]

The inspected driver executes one \(N\)-row prefill, **K** cached one-row
passes, and one full \(S\)-row trace pass. Its generation loop is inclusive:
it generates an extra final prediction, discards that prediction, and retains
K output tokens. Consequently the reference processes \(2S\) rows. This is a
property of this driver, not a necessary cost of all autoregressive inference.

The dense fixed-width projections require

\[
A=12Ld^2+dV=123{,}532{,}032
\]

MACs per row, where \(L=12,d=768,V=50257\). The attention matmuls require
\(2Ld=18{,}432\) MACs per query/key pair. Their matrices are rectangular and
dense: masking does not turn the executed matmul into triangular work. Thus

\[
I_{\rm MAC}(N,K)=247{,}064{,}064S+18{,}432U.
\]

The complete inference vector uses one separate symbolic weight per row below.
The element operations are composite reference charges for a specified layer
kind and quantization rule. Reductions are charged by their input-element
counts; one such charge is not claimed to equal one CPU instruction.
The quantized graph stores values as `Element = i64`. One MAC unit is one
logical scalar product accumulation, regardless of backend vectorization or
packing; it is a different family from a BN254 field MAC.

| Component | Count |
| --- | ---: |
| `integer_mac` | \(247064064S+18432U\) |
| `tensor_embeddings_elements` | \(1536S\) |
| `tensor_positional_elements` | \(1536S\) |
| `tensor_layer-norm_elements` | \(38400S\) |
| `tensor_add_elements` | \(36864S\) |
| `tensor_activation_elements` | \(73728S\) |
| `tensor_logits_elements` | \(100514S\) |
| `tensor_attention-mask_elements` | \(144U\) |
| `tensor_softmax_elements` | \(144U\) |
| `tensor_requant_elements` | \(150528S+18432R\) |

The extra \(R\) term is required because the graph concatenates cached K/V
prefixes before requantization. These prefixes are revisited during decoding.
Inference therefore depends on N and K separately even at fixed total length.

## Equivalent compact definition for the paper

The vector definition can be written as

\[
\boxed{C_{\mathbf w}(N,K)
=\alpha_{\mathbf w}S+\beta_{\mathbf w}U+\gamma_{\mathbf w}R
+w_{\rm axis}D(S)+G_{\mathbf w}(P).}
\]

Here \(G_{\mathbf w}(P)=\mathbf w^T\mathbf B_P\), using the five-entry
proof table below. The inference coefficients are the following known linear
combinations of the corresponding operation weights:

\[
\begin{aligned}
\alpha_{\mathbf w}={}&247064064w_{\rm MAC}
+1536(w_{\rm emb}+w_{\rm pos})+38400w_{\rm LN}\\
&+36864w_{\rm add}+73728w_{\rm act}+100514w_{\rm logits}
+150528w_{\rm requant},\\
\beta_{\mathbf w}={}&18432w_{\rm MAC}
+144(w_{\rm mask}+w_{\rm softmax}),\\
\gamma_{\mathbf w}={}&18432w_{\rm requant}.
\end{aligned}
\]

The model, proof implementation and setup subscripts are suppressed here only
for readability. These coefficients and table values are not independently
fitted free parameters: they follow from the same public operation weights.

## Proof derivation and gas rules

The unpadded Einstein-sum axis reductions have the exact structural formula

\[
D(S)=123{,}863{,}808+111{,}360S+144S^2.
\]

For each block, count the reductions of the Q/K/V projections, two attention
products, attention output, and two MLP projections, followed by the final
vocabulary projection. Reducing Q/K/V's two output axes successively contributes
the additional \(3Ld\cdot12\) term beyond \(A\). The test suite independently
enumerates those equations. The other metered families are collected in
\(\mathbf B_P\), where \(P\in\{4,8,16,32,64\}\).

The following rules specify the reference machine. Sumcheck preserves stored
polynomial prefixes and implicit zero padding. Other operations use the dense
reference reductions stated explicitly below. Degree-specific algebra and
backend shortcuts can still change literal instruction counts.

| Event | Charged components |
| --- | --- |
| Sumcheck with v variables; MLE j has logical dimension \(u_j\), stored-prefix size \(2^{h_j}\); monomial i has dimension \(q_i\), minimum factor-prefix size \(2^{r_i}\), degree \(d_i\) | `sumcheck_fold` = \(\sum_j(2^{h_j}-1+u_j-h_j)\); `sumcheck_term` = \(\sum_i H_i\); `sumcheck_factor` = \(\sum_i d_iH_i\); `sumcheck_round` = v, where \(H_i=(v-q_i)2^{r_i}+(d_i+1)(2^{r_i}-1+q_i-r_i)\) |
| Ordinary MLE reduction from M entries to \(M/2^r\) entries | `mle_fold` = \(M-M/2^r\); full evaluation uses M−1 |
| Polynomial random linear combination | `field_rlc_mac` = sum of input polynomial lengths |
| Einstein-sum outer-axis dot products | `field_axis_mac` = number of input entries reduced; total D(S) above |
| HyperKZG opening of M = \(2^v\) entries | `pcs_fold` = M−2; `pcs_horner` = 3(2M−2); `pcs_division` = 3(M−1) |
| MSM of r scalar/base pairs | One `msm_r`; its weight is an unspecified size-dependent function \(g_{\rm MSM}(r)\) |
| LogUp circuit of M rows and c input columns | `logup_column_cell` = Mc; `logup_fraction_merge` = max(0,M−2), because this tree stops at two rows |
| Lookup decomposition of M entries into q limbs | `lookup_limb` = Mq |
| Transcript | Number of appended field scalars and curve points, number of challenge calls, and number of challenge output bytes |

The three sumcheck bulk charges describe dense pair folds, monomial
accumulations, and factor evaluations/products. Each has its own weight.
Crucially, **each polynomial keeps its own stored prefix and logical domain**.
A small polynomial is not charged as though it filled the largest polynomial
in the batch. Before a monomial's own variables are reached, its stored prefix
is summed once per inactive round; this corresponds to the uncached branch in
`sumcheck_macro/src/lib.rs`.
During its active rounds, summing the pair counts gives \(2^{r_i}-1\). After
that prefix reduces to one coefficient, each of the \(q_i-r_i\) remaining
padding variables scales that coefficient. There are \(d_i+1\) reference
evaluation points per pair. A product can skip positions outside its smallest
factor prefix because at least one factor is zero there. This canonical
ordering avoids making gas depend on factor ordering.

Worker splits are undone when recording domains and stored lengths. Placeholder
zeros in empty workers are excluded. A Rust regression test checks reconstruction
across powers of two and worker counts, so thread count does not enter this rule.

Degree-specialized algebra may further reduce literal instructions. Scalar
interpolation and bookkeeping are not expanded into all
their primitive operations here. Sumcheck's internal MLE folds are accounted
for by its sumcheck charge, not again through `mle_fold`. The earlier
largest-domain counter was discarded because it overcharged small polynomials
in mixed-size batches. Merely recording each polynomial's declared domain was
also insufficient: `zero_pad_num_vars` changes that domain without allocating
the extra coefficients. The final counter records both quantities.

HyperKZG folds the multilinear polynomial, performs three univariate evaluation
chains, forms one batched polynomial, and performs three synthetic divisions.
Its RLC and actual commitment calls are charged separately, so the opening rule
does not add them a second time. This configuration has **no FFT term**:
[the pinned HyperKZG implementation](https://github.com/Lagrange-Labs/dp-crypto/blob/22e8e93cd0f94a7638616a1ae22190feb0a6b275/dp-crypto/src/arkyper/mod.rs)
works directly with the multilinear evaluation representation.

MSM sizes remain separate because one giant MSM and many small MSMs need not
have the same resource profile. Choosing \(g_{\rm MSM}(r)=r w_{\rm pair}\) is
one later simplification; the present definition does not assume it.

Transcript byte lengths are diagnostic only. An identity curve point has a
shorter serialization, so charging raw encoded bytes would introduce a
token-value dependency. Logical scalar/point charges remove that dependency.
There is no separate charge for transcript metadata labels or raw append bytes.

No forward matmul events occurred during the instrumented proof phase. The
full trace reconstruction is already included in inference; this implementation
does not justify charging another entire model forward pass during proving.

## Validation results

**87 complete inference/proof executions verified successfully**, covering
40 distinct (N,K) pairs and 24 total lengths. Five canonical profiles define
the table; the remaining 82 executions include shape checks, fresh prompts,
thread replays and a controlled counterexample. There are 35 distinct length
pairs outside the five canonical pairs. These executions are not all
statistically independent samples.

All inference components and D(S) matched their formulas exactly. The full
proof vector matched exactly in 67 of 87 executions. The maximum proof-component
error was **3.1088%**; 85 of 87 executions were within 1%. The two executions
above 1% were the original outlier and its controlled replay. The mean of the
per-execution maximum component errors was 0.1468%; this is not a timing error.

| Campaign | Verified trials | CPU threads | Prompt seed / source | Maximum proof-component error |
| --- | ---: | ---: | --- | ---: |
| Final profiles, edge case and fresh repeats (`audited`) | 14 | 8 | 20260910 | 3.1088% |
| Boundary neighbors and extreme splits (`boundary_check`) | 20 | 8 | 20260910 | 0.2674% |
| Independent short prompts (`independent`) | 7 | 4 | 20260911 | 0.7692% |
| Stratified random pairs (`random_check`) | 16 | 4 | 20260912 | 0.2674% |
| Identical canonical prompts (`thread_check`) | 5 | 4 | 20260910 | 0% |
| Fresh short-context stress (`short_stress`) | 21 | 8 | 20260913 | 0.7958% |
| Normal / outlier / normal replay and capacity anchor (`paired_replay`) | 4 | 8 | Explicit token IDs | 3.1088% |

The random-pair sample used seed 20260912, sampled separately within the five
padded-length regions, and added a total-64 pair to retain setup capacity.
The short stress run repeated `(2,1), (2,2), (2,3), (2,4), (3,2)` four times
with fresh prompts, followed by a total-64 anchor. Its pairs were fixed before
examining its results, and the original proof profiles were not retuned.

The CPU was an AMD Ryzen Threadripper PRO 5955WX. Changing from eight to four
threads reproduced all five canonical count vectors exactly on identical
prompts. This is a worker-count check on one physical device. CUDA compilation
passed, but the GPUs lacked sufficient free memory for the context-64 proving
check; no GPU or second-physical-device count validation is claimed.

The consolidated [validation summary](../../experiments/results/deepprove_work/validation_summary.json)
links the campaign statistics through one manifest identifier. Detailed
differences remain in each campaign's `validation.json`, with the original
verified ledgers retained. The earlier boundary run stopped at the prompt-length
guard described below; its 20 successful trials were preserved as checks, and
the unfinished cases were run after the fix.

- Schedule SHA-256: `6d62b1c6aae20268df847bcf0e1b7a213b97e5458b5a3bd751e696edc0e75c5d`.
- Canonical-run setup digest: `5274ccb62e46245c4181633e5584091b4cb737d17dd3c066e32faa5a98157ab5`.
- `weights` remains `null`; no relative gas prices were assigned.

The [3D component plots](../../experiments/results/deepprove_work/audited/work_counts.png)
and [PDF](../../experiments/results/deepprove_work/audited/work_counts.pdf) evaluate
the reference formula for all **1,953 allowed pairs**. Those are calculated
surfaces, not 1,953 proof runs. The associated `all_pairs.csv` exports every
symbolic gas coefficient.

The maximum-prompt regression exposed two off-by-one guards that reserved two
output positions. Both now allow a prompt that leaves one output position.
The corrected \((63,1)\) case generated a verified proof with exact inference
and proof counts. Replaying all five canonical prompts after the fix reproduced
the same profiles; the counting rules did not change. The default campaign
retains this edge case as an executable regression check.

| Implementation check | Result |
| --- | --- |
| Full Python suite, including EZKL integration | 159 passed |
| Rust `bench-llm` unit tests | 6 passed, including exact-token replay validation |
| Stored-prefix reconstruction: actual Rust helper and test module compiled in isolation | 1 passed across the enumerated domains and worker splits |
| Python lint/format and modified Rust formatting | Passed |
| Patches applied twice to fresh pinned checkouts | Passed; temporary checkouts removed |
| CUDA `cargo check --release --features cuda --bin bench-llm` | Passed; no GPU proving result claimed |

Inference and D(S) must agree exactly. For proof components, validation uses
the explicit criterion

\[
\epsilon_{\rm trial}=\max_j
\frac{|\widehat W_j-W_j|}{W_j}\le 5\%,
\]

where W is the counted reference-operation vector for the actual execution and
the hat denotes the fixed length-only schedule. Both-zero components have zero
error; a predicted positive component with an observed zero has unbounded
relative error and fails validation. Every discrepancy is saved, rather than
silently corrected or averaged. A failed validation is evidence to revise the
schedule or narrow its claimed scope, not a failed cryptographic proof.

The initial 1% criterion was falsified. At the same \((N,K)=(2,3)\), two
verified executions under one identical setup used 374 and 386 MSMs of size
8192. Repeating the normal prompt returned to 374 and reproduced its entire
count vector exactly. Twelve lookup circuits
changed from one input column to two, although their row counts and the number
of decomposed limbs stayed the same. This is consistent with the
activation-dependent normalization shifts selecting different lookup layouts.
For any single length-only estimate z of this component,

\[
\min_z\max\left(\frac{|z-374|}{374},\frac{|z-386|}{386}\right)
=\frac{386-374}{386+374}\approx1.58\%.
\]

Thus no length-only rule can meet 1% for both executions under arbitrary
nonnegative weights. Adding more N,K polynomial terms cannot fix that
counterexample. The profiles were left unchanged; a 5% acceptance criterion
was selected for a fresh short-context stress campaign. This threshold is an
experimental adequacy criterion, not a guaranteed worst-case bound over prompts.

For **any nonnegative relative gas weights** and positive observed weighted
cost, the triangle inequality gives

\[
\frac{|\mathbf w^T\widehat{\mathbf W}-\mathbf w^T\mathbf W|}
{\mathbf w^T\mathbf W}\le\epsilon_{\rm trial}.
\]

Thus the observed componentwise bound also bounds the relative error in the
combined weighted count, independently of the weights. This bound compares two
reference-work counts on tested executions. It does not bound elapsed-time error
or guarantee the same residual for every possible prompt.

The full table of proof coefficients is reproduced below. Each entry is a count
under the reference rules above, before multiplication by its symbolic weight.
`field_axis_mac` is excluded from this table because its formula uses S, not P.

| Component | P=4 | P=8 | P=16 | P=32 | P=64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `field_rlc_mac` | 292,345,098 | 296,618,518 | 305,248,302 | 322,872,414 | 359,349,438 |
| `logup_column_cell` | 4,807,944 | 9,081,360 | 17,711,136 | 35,335,232 | 71,812,224 |
| `logup_fraction_merge` | 3,706,916 | 7,122,732 | 14,009,660 | 28,004,700 | 56,879,516 |
| `lookup_limb` | 2,610,176 | 5,238,784 | 10,551,296 | 21,397,504 | 43,974,656 |
| `mle_fold` | 68,494,694 | 68,696,079 | 69,098,848 | 69,904,385 | 71,515,458 |
| `pcs_division` | 3,145,725 | 3,145,725 | 3,145,725 | 3,145,725 | 3,145,725 |
| `pcs_fold` | 1,048,574 | 1,048,574 | 1,048,574 | 1,048,574 | 1,048,574 |
| `pcs_horner` | 6,291,450 | 6,291,450 | 6,291,450 | 6,291,450 | 6,291,450 |
| `sumcheck_factor` | 2,140,756,588 | 2,379,606,084 | 2,862,209,244 | 3,847,289,182 | 5,870,966,822 |
| `sumcheck_fold` | 391,453,456 | 416,688,149 | 467,594,346 | 568,059,239 | 775,184,279 |
| `sumcheck_round` | 10,377 | 12,188 | 14,183 | 16,362 | 18,725 |
| `sumcheck_term` | 1,125,549,212 | 1,224,924,189 | 1,425,612,286 | 1,830,276,325 | 2,662,963,299 |
| `transcript_challenge` | 19,869 | 22,160 | 24,635 | 27,298 | 30,146 |
| `transcript_output_byte` | 317,904 | 354,560 | 394,160 | 436,768 | 482,336 |
| `transcript_points` | 4,837 | 4,837 | 4,837 | 4,841 | 4,846 |
| `transcript_scalars` | 66,826 | 74,367 | 82,644 | 91,692 | 101,415 |
| `msm_2` | 1 | 1 | 1 | 1 | 1 |
| `msm_4` | 196 | 1 | 1 | 1 | 1 |
| `msm_8` | 1 | 196 | 1 | 1 | 1 |
| `msm_16` | 865 | 1 | 196 | 1 | 1 |
| `msm_32` | 1 | 1 | 1 | 196 | 1 |
| `msm_64` | 1 | 865 | 1 | 1 | 196 |
| `msm_128` | 1 | 1 | 1 | 1 | 1 |
| `msm_256` | 2,881 | 1 | 865 | 1 | 1 |
| `msm_512` | 1 | 2,881 | 1 | 1 | 1 |
| `msm_1024` | 2 | 2 | 2,882 | 866 | 2 |
| `msm_2048` | 1 | 1 | 1 | 2,881 | 1 |
| `msm_4096` | 387 | 14 | 14 | 14 | 3,758 |
| `msm_8192` | 1 | 374 | 1 | 1 | 1 |
| `msm_16384` | 73 | 1 | 374 | 1 | 1 |
| `msm_32768` | 1 | 73 | 1 | 375 | 1 |
| `msm_65536` | 5 | 5 | 77 | 5 | 378 |
| `msm_131072` | 1 | 1 | 1 | 73 | 1 |
| `msm_262144` | 4 | 1 | 1 | 1 | 73 |
| `msm_524288` | 1 | 4 | 1 | 1 | 1 |
| `msm_1048576` | 3 | 3 | 6 | 9 | 15 |


The first trial at each boundary defines its canonical profile. Later trials,
including repeated boundaries, are validation data. The source-derived padding
rule is checked on those trials, with the explicitly reported adaptive-shift
residual. There is no statistical regression or formal proof that every
implementation branch has been covered.
The published table defines the normative schedule exactly. The validation
provides evidence that it follows the counted implementation structure.

## What the growth looks like

Inference contains linear row work and quadratic attention/cache terms.
Proof components mostly follow the padded length P, with the separate smooth
axis-reduction term D(S). This produces steps after S=4, 8, 16 and 32; the
table makes no claim about lengths above 64.

For example, changing \((N,K)=(16,16)\) to \((16,17)\) increases dense
inference MACs from 7,936,868,352 to 8,185,738,752. The reference sumcheck factor
count jumps from 3,847,289,182 to 5,870,966,822 because P changes from 32 to 64.
These are two different resource changes, not one weighted percentage.

At fixed S=64, \((2,62)\) and \((63,1)\) share the same proof profile and
axis-reduction count. Their dense MAC counts differ by only about 0.23%, but
their requantized-element counts are 46,774,272 and 10,795,008 respectively.
Keeping N and K separate is therefore justified. Which difference matters
most to gas depends on the still-unspecified weights.

## Relation to PoML and limitations

1. The paper's query-complexity interface can use this public integer function.
   A verifier computes N and K from the authenticated query and completion,
   checks the registered termination rule, and evaluates the schedule without
   repeating inference or proving. A miner's self-reported counter is never used
   for its fee or lottery weight.
2. The reference includes prompt prefill and trace reconstruction. If PoML's
   registered M1/M2 split excludes prefill, or the fresh challenge modifies the
   generation graph, that is a different reference computation. Do not subtract
   an empirical constant and keep the same schedule identifier.
3. This benchmark proves DeepProve's quantized GPT-2 greedy generation. It does
   not implement the paper's complete VRF-conditioned sampler, encryption and
   freshness relation. Those constraints and their proof costs must be included
   when that full relation is implemented. Their cost is currently omitted,
   not empirically shown negligible.
4. Memory traffic, allocation, conversions, tensor-layout operations, some
   polynomial preparation, bias additions internal to matmuls, and scalar
   bookkeeping are not separately metered. Bulk nonlinear layers and lookup
   work have composite charges. There is no measured bound on the residual
   physical cost, and there is no claim of low percentage timing error.
   For literal proof counts, activation-dependent LayerNorm shifts would be
   additional inputs. `evaluate_quantised_internal` derives these from the
   normalization factors, and `LayerNormLookupVerifier::chunking_info` uses
   them in lookup decomposition. Treating one observed difference as a special
   case for total length 3 would therefore be unsound. The approximate public
   profile avoids exposing or trusting those private execution details.
5. Device independence means that every device receives the same public gas
   value. It cannot mean identical elapsed-time ratios across CPUs and GPUs:
   memory bandwidth, arithmetic, and MSMs accelerate differently. The full
   vector preserves the information needed for later weight selection.
6. The paper's depth and computational-independence assumptions are separate.
   Work is circuit size/reference operations; depth is the longest dependency
   chain. Autoregressive K changes sequential dependence even where total work
   is similar. These measurements do not prove the aggregate calibration or
   no-reuse lower bounds used to justify lottery fairness.
7. A more efficient driver could remove the discarded extra prediction or avoid
   full trace recomputation. This version intentionally describes the audited
   driver. Such improvements require rederiving and registering its successor.

If exact agreement with every execution's structural plan becomes necessary,
there are two alternatives: make the lookup layouts independent of activation
values, or extend the public cost input with a canonical, proof-bound descriptor
of the prescribed execution. Merely authenticating a miner-chosen padded layout
would not justify charging extra gas. Both alternatives require further
protocol/implementation work. The fixed
length-only table is the simpler choice for the present approximation goal.

## Reproduce and use

Implementation locations:

- `src/poml_sim/gpt2_work.py`: formula, operation rules, strict manifest compiler,
  digest verification, symbolic count calculator, explicit-weight evaluator.
- `experiments/run_deepprove_work.py`: progress bars, one shared setup, complete
  verified ledgers, profile compilation, and export of all allowed pairs.
- `scripts/prepare_deepprove_work.py` and the two `*_work.patch` files: reproducible
  pinned source changes. Crypto is copied to `.scratch/dp-crypto-work`; Cargo's
  cached checkout is not modified.
- `tests/test_gpt2_work.py` and `tests/test_deepprove_work_campaign.py`: formula,
  boundary, malformed-input, phase-exclusion, and mismatch rejection tests.

From this repository, rerun the 29-trial boundary/split campaign in a new output
directory:

```bash
.venv/bin/python experiments/run_deepprove_work.py --prepare --output-dir experiments/results/deepprove_work/recheck
```

The default is CPU with eight threads. Add `--cuda --device 0` to build and use
CUDA, with sufficient free device memory and the existing CUDA prerequisites.
No second-device validation result is implied by this option. A reference gas
calculation itself uses no GPU and runs no model:

```bash
.venv/bin/python experiments/run_deepprove_work.py --calculate 16:17
```

For exact token replay, `--work-prompts path.json` accepts one array of integer
token IDs per pair. Each array must have exactly N entries within the model's
vocabulary. Requested and observed tokens are recorded and compared. For example,
the controlled counterexample can be reproduced with:

```bash
.venv/bin/python experiments/run_deepprove_work.py --pairs 2:3,2:3,2:3,32:32 --work-prompts experiments/results/deepprove_work/replay_inputs.json --compare-schedule experiments/results/deepprove_work/audited/schedule.json --output-dir experiments/results/deepprove_work/replay_check
```

The input file contains the normal prompt `[15969,42773]`, the outlier
`[7905,34786]`, the normal prompt again, and a 32-token capacity-anchor prompt.
The complete arrays are also saved in the replay's `run.json`. The last pair
keeps setup capacity at 64 while the first three isolate input values and query
history at one fixed N,K pair.

The result contains separate inference and proof count vectors plus their sum.
For an independent backend check, add
`--compare-schedule experiments/results/deepprove_work/audited/schedule.json`
to a new campaign. Inference and axis formulas must match, and each proof
component must stay within the stated tolerance of the existing schedule;
validating a newly compiled backend-specific table alone is insufficient.
Apply `weighted_cost(counts, weights)` only after explicitly specifying every
needed weight. Generated `schedule.json` intentionally contains `weights: null`.

The runner refuses to overwrite an existing ledger. To analyze a completed run
again without generating new proofs:

```bash
.venv/bin/python experiments/run_deepprove_work.py --analyze experiments/results/deepprove_work/audited/ledger.jsonl
```

The output directory contains `run.json` (source/binary/seeds provenance),
`ledger.jsonl` (only successfully verified trials), `schedule.json` (five proof
profiles and digest), `validation.json`, `all_pairs.csv`, diagnostic timings and
the process log. `work_counts.png` and `work_counts.pdf` show three separate
resource families across N and K; MSM scalar/base entries in that plot are a
size diagnostic, not a choice of MSM gas weights. Setup and verification execute
during validation but are outside the counter scope. Instrumentation overhead makes these timings
unsuitable as an uninstrumented performance benchmark. Generated results stay
gitignored; this report and the source patches preserve the reviewable definition.

No Python dependencies were added. The Rust meter uses `serde_json` as a runtime
dependency; it was already in DeepProve's dependency graph and remains pinned
by its `Cargo.lock` to 1.0.149. The preparation command preserves local changes
on a patch conflict and never edits Cargo's cached dependency checkout.

## Source audit trail

- Paper: `.scratch/PoML_paper_draft/4_protocols.tex`, query complexity;
  `3_problem.tex` and `appendix.tex`, depth/reuse/aggregate calibration assumptions.
- DeepProve: `zkml/src/model/llm.rs`, inclusive generation loop, trace rebuild,
  context reuse and fixed-length verifier; `parser/llm/models/gpt2/mod.rs`, graph;
  `layers/einsum/evaluate.rs` and `axis.rs`, matmuls and unpadded reductions.
- DeepProve: `lookup/logup_gkr/circuit.rs` and `lookup/operation/decomposer.rs`,
  lookup construction and decomposition; `bin/bench/llm.rs`, online boundaries.
- dp-crypto: `sumcheck/prover.rs`, sumcheck shape; `poly/dense.rs`, MLE and RLC;
  `arkyper/{mod,msm,hyperkzg_gpu}.rs`, opening and commitment calls;
  `arkyper/transcript`, logical transcript elements and byte diagnostics.

This report supersedes the earlier timing-based candidates for the fixed
configuration above. Historical timing results remain useful for estimating
benchmark duration; they do not define consensus gas.
