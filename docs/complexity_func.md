# A complexity function for LLM inference and proof generation

**Current recommendation:** use the [deterministic reference-work definition](features/deepprove_work_accounting.md).
For the audited GPT-2/DeepProve setup with maximum context 64, it is
\(C_{\mathbf w}(N,K)=\mathbf w^T[\mathbf I(N,K)+\mathbf e_{\rm axis}D(N+K)+\mathbf B_{2^{\lceil\log_2(N+K)\rceil}}]\).
The inference and axis terms are formulas; the five proof profiles contain
integer structural counts. All relative gas weights remain variables. Setup
and verification are excluded. The candidates and timing experiment below are
historical motivation, not the current gas schedule; in particular this
HyperKZG configuration does not need an FFT term.

## Background

PoML needs a public complexity function (C_{\theta,\Pi}(N,K)) for a fixed model (M_\theta) and proof system (\Pi). It determines fees, work units, and lottery weight. Here (N) is the tokenized prompt length and (K) is the number of generated output transitions (with the EOS convention stated explicitly). The function should describe a fixed, reproducible reference computation; it is not automatically a cryptographic lower bound on an optimized adversary.

## Candidate formulas

For KV-cached inference, the discussion draft proposes

\[
C_{\rm inf}(N,K)=C_{\rm prefill}(N)+C_{\rm decode}(N,K),
\]
\[
C_{\rm prefill}(N)\approx a_0+a_1N+a_2N^2,
\]
\[
C_{\rm decode}(N,K)\approx b_0K+b_1\left(NK+\frac{K(K-1)}2\right).
\]

The constant and linear terms cover fixed-width projections, MLPs, normalization, vocabulary work, and kernel overhead. (N^2) represents prompt self-attention. During decoding, each new token attends to the (N+t-1) cached positions, producing the (NK) and (K^2/2) terms. If prompt prefill is supplied by an (M_1) service and excluded from mining, omit its cost from the lottery function, but retain (N)-dependent decode attention.

For DeepProve-style whole-sequence proving, let (s=N+K):

\[
C_{\rm DP}(N,K)=F(s)+G(K),
\qquad F(s)\approx d_0+d_1s+d_2s^2+d_3s\log s.
\]

Here (F) is the proof cost for one causally masked transformer pass and (G) covers output-token checks, VRF/sampler constraints, and related proof overhead. The (s^2) and (s\log s) terms are candidate empirical components, not guaranteed asymptotics; padding, lookup domains, FFTs, commitments, and implementation choices may make the curve stepwise.

The combined reference cost is

\[
C_{\theta,\Pi}(N,K)=C_{\rm inf}(N,K)+C_{\rm DP}(N,K)+C_{\rm aux}(N,K),
\]

where (C_{\rm aux}) includes VRFs, randomness expansion, encryption, serialization, and any other disjoint work. Shared forward or witness work must be counted once. If the prover benchmark already includes inference or witness generation, use its measured end-to-end cost instead of summing overlapping terms.

## Experiment plan: GPT-2 and DeepProve

Use GPT-2 (`openai-community/gpt2`) and the existing WikiText-2 and LAMBADA prompt sets from `gpt2_poml_collision_experimental_plan.md`; optionally add HellaSwag prompts from the embedding-perturbation plan. Tokenize prompts and retain a broad range of (N), for example 16, 32, 64, 128, 256, and 512, subject to the model context limit. For each prompt, evaluate output lengths (K\in\{1,4,8,16,32,64,128\}), including a fixed-length mode so EOS does not confound comparisons.

For each ((N,K)), freeze model weights, tokenizer, quantization, batch size, hardware, software version, DeepProve commit, proof mode, and padding policy. Run warm-up trials, then collect repeated measurements for:

1. prompt prefill time and operation counts;
2. KV-cached decode time and operation counts by token;
3. DeepProve proving time, memory, field operations, lookups, FFTs/MSMs, and proof size;
4. VRF/sampler, encryption, serialization, and verification overhead;
5. end-to-end challenge-to-proof time.

Run DeepProve’s GPT-2 sequence benchmark over the same total lengths (s=N+K). Record whether the implementation proves a padded length or the realized length. Keep separate measurements for inference, proving, and the end-to-end pipeline to detect double counting.

Fit nested models: (i) linear baselines, (ii) the proposed inference expression, (iii) the proposed proof expression, and (iv) the combined expression. Use nonlinear least squares or robust regression on training length combinations, with held-out prompts and held-out ((N,K)) combinations for validation. Report coefficients with confidence intervals, RMSE, MAE, relative error, (R^2), residual plots, and worst-case relative error. Compare against a lookup table indexed by allowed length buckets.

The formula is credible only if coefficients are stable across prompt datasets, residuals show no systematic dependence on (N), (K), or prompt content, and held-out error is small enough for fee/work-unit calibration. If proof cost depends mainly on (s), simplify (F); if padding or hardware effects dominate, publish a deterministic bucket table rather than extrapolating a polynomial.

Finally, treat the fitted function as a public reference schedule. Separately test the security assumption by measuring how much inference and proving work an attacker can reuse after a fresh challenge. A low fitting error establishes predictability of honest cost, not computational independence.
