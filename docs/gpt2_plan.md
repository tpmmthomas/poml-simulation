# GPT-2 PoML collision and cache-reuse experimental plan

## Scope and primary recommendation

The main empirical question is:

> For a fixed prompt and two independently sampled, VRF-driven GPT-2 traces, how quickly does the probability of an identical generated-token prefix decrease with prefix length?

The primary experiment should estimate this curve directly. The other experiments support its interpretation: temperature characterizes token concentration, a fixed maximum output length \(K\) makes the measurement protocol-compatible, and cache experiments translate prefix matches into an actual computational advantage.

Use the publicly released gpt2 checkpoint (approximately 124M parameters), its native byte-level BPE tokenizer, deterministic evaluation mode, and a fixed software/model hash. Do not fine-tune the model. Before collecting collision data, run a sanity check against the standard next-token perplexity evaluation on the selected corpus.

## Benchmark choice

### Primary benchmark: WikiText-2 test split

Use the raw WikiText-2 test split as the main prompt source. WikiText was introduced as a standard long-term-dependency language-modeling dataset, preserves case, punctuation, and numbers, and is less aggressively normalized than Penn Treebank. GPT-2 was explicitly evaluated on WikiText-2. Sources: [Merity et al., *Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843), [the WikiText dataset description](https://www.salesforce.com/blog/the-wikitext-long-term-dependency-language-modeling-dataset), and [OpenAI's GPT-2 evaluation report](https://openai.com/index/better-language-models/).

Use the test split only. Remove empty and formatting-only entries, preserve ordinary prose and punctuation, and deduplicate exact token prompts. Sample approximately 1,000 prompts, stratified across articles. For each prompt, create a 32-token GPT-2 BPE window. Use 16 and 64 prompt tokens as sensitivity conditions. The main result should not depend on one article or prompt template.

### Secondary benchmark: LAMBADA test subset

Use a smaller LAMBADA test subset as a robustness check. LAMBADA consists of narrative passages designed to require broader discourse context for word prediction, and it was also used in GPT-2 evaluation. Source: [Paperno et al., *The LAMBADA dataset*](https://arxiv.org/abs/1606.06031).

Sample 500--1,000 passages and truncate each context to a fixed GPT-2-token budget, such as 32 or 64 tokens. We use LAMBADA as a natural-language prompt source, not as a claim that we reproduce its original word-prediction score.

### What not to use as the primary benchmark

Avoid MMLU, HellaSwag, TruthfulQA, long-document tasks, and instruction-tuning datasets for the main collision estimate. They add task formatting, long contexts, or capabilities not needed for this question. Penn Treebank can be a small preprocessing sanity check, but its normalized text is less representative of ordinary prompts.

## Common protocol

Let \(x\) be a tokenized prompt of length \(p\). At decoding step \(t\), GPT-2 produces

\[
P_\theta(v\mid x,y_{<t};\tau),
\]

where \(\tau\) is temperature. A VRF-derived uniform value \(u_t\) is mapped to a token by inverse-CDF sampling. For a fixed prompt, model, chain binding, miner key, temperature, and VRF transcript, the trace is deterministic.

Use full categorical sampling initially: top-\(p=1\), no top-\(k\), and no repetition penalty. This isolates temperature and randomness. Record the exact VRF input and output used at every step. A cryptographic hash/VRF-derived counter stream is preferable to an implementation-dependent pseudorandom generator; a standard library sampler can be retained as a cross-check.

Use temperatures

\[
\tau\in\{0.7,1.0,1.3,1.5,2.0\},
\]

with \(\tau=1.0\) as baseline. If resources are limited, begin with \(0.7,1.0,1.5\).

Set the primary maximum output length to \(K=32\), with \(K=16\) and \(K=64\) sensitivity runs. The primary protocol-faithful mode stops at EOS or \(K\), whichever comes first. Do not invent tokens after EOS. At prefix length \(\ell\), count a pair only when both traces have generated at least \(\ell\) tokens and agree on all \(\ell\) tokens. EOS rates are supporting statistics, not the main result.

For each condition, fix the prompt list and use independent VRF seeds for the two traces in every pair. A practical starting point is 1,000 WikiText-2 prompts, 256 independent pairs per prompt, and the same number of pairs for the LAMBADA subset. Increase the number of pairs if the tail remains unresolved.

## Experiment 1: prefix-collision probability and curve

### Hypothesis

The probability of an identical generated prefix decreases rapidly with \(\ell\), but the rate is prompt-dependent. A small number of concentrated prompts may have much larger collision probabilities than the aggregate average.

### Measurement

For two traces \(Y^{(a)}\) and \(Y^{(b)}\) generated from the same prompt and independent seeds, define

\[
I_\ell=\mathbf{1}\{Y^{(a)}_{1:\ell}=Y^{(b)}_{1:\ell}\},
\qquad
C_\ell=\Pr[I_\ell=1].
\]

Here \(C_0=1\). Because GPT-2 is deterministic conditional on tokens, equality of the first \(\ell\) generated tokens is equivalent to equality of the decoder input at the next step, assuming canonical numerical execution. Record the first divergence position

\[
L=\min\{t:Y^{(a)}_t\ne Y^{(b)}_t\}.
\]

Estimate \(C_\ell\) at every \(\ell\leq K\) using independent pair indicators. Compute curves separately for WikiText-2 and LAMBADA, and separately for each temperature.

### Statistical treatment

- Use prompt-level bootstrap confidence intervals, not only token-level intervals, because prompts are the unit of generalization.
- Report pooled estimates, the median prompt-level curve, and the 10th/50th/90th percentile prompt curves.
- If no collisions are observed at a given \(\ell\), report a binomial upper confidence bound; do not report probability zero.
- Report the empirical distribution or survival curve of \(L\).
- Check monotonicity \(C_{\ell+1}\leq C_\ell\); violations indicate a measurement or EOS-handling bug.

### Main outputs

1. A log-scale plot of \(C_\ell\) versus \(\ell\), with 95% prompt-bootstrap bands.
2. Separate curves for WikiText-2 and LAMBADA.
3. A table containing \(C_1,C_2,C_4,C_8,C_{16},C_{32}\), with confidence intervals.
4. Prompt-level quantiles and the fraction of prompts with at least one collision at each \(\ell\).
5. A first-divergence CDF or survival plot.

The paper claim should be calibrated to the measured curve, for example: “On the evaluated GPT-2 prompt distribution, \(C_\ell\) falls below [measured bound] by \(\ell=[measured length]\).” This is distributional evidence, not a universal CIA theorem.

## Experiment 2: temperature and token concentration

### Hypothesis

Increasing temperature generally reduces top-token probability and should reduce prefix collisions, but the effect may be weak for intrinsically concentrated prompts.

### Measurement

Before sampling each token, record:

- top-token probability \(q_t=\max_v P_\theta(v\mid x,y_{<t};\tau)\);
- the identity of the top token;
- the sampled token and its probability;
- EOS probability;
- optionally, entropy and top-10 probability mass.

For the paper, focus on \(q_t\) and its temperature dependence. Aggregate over prompts and active decoding steps, while also reporting positions \(t=1,2,4,8,16\). Since the distribution is conditional on the generated prefix, report medians and quantiles rather than only one global mean.

### Main outputs

1. Temperature versus median top-token probability, with 10th/50th/90th percentiles.
2. Temperature versus the prefix-collision curves from Experiment 1.
3. A scatter plot of prompt-level average top-token probability against \(C_8\) or \(C_{16}\).

This experiment characterizes the randomness/collision trade-off; it does not establish which temperature gives the best language quality. A perplexity sanity check at \(\tau=1\) is sufficient to detect basic implementation errors.

## Experiment 3: fixed maximum length \(K\)

### Recommendation

Use a fixed maximum \(K\) for the main collision experiment. The purpose is to estimate prefix reuse, not to model the full response-length distribution. Run \(K=32\) as the primary condition and \(K=16,64\) as sensitivity checks.

### Measurement

For each \(K\), report:

- the collision curve up to \(\ell=K\);
- the fraction of traces ending in EOS before \(K\);
- the effective number of defined pair comparisons at each \(\ell\);
- the change in \(C_\ell\) for values of \(\ell\) shared by all \(K\) conditions.

If a purely fixed-step diagnostic is useful, run a separate no-early-stop mode in which EOS is treated as an ordinary sampled token for exactly \(K\) steps. Label this as a diagnostic; it should not replace the EOS-aware result because post-EOS tokens are not part of the deployed protocol.

### Main outputs

Use the \(K=32\) curve as the main figure. Add an inset or appendix plot showing \(K\)-sensitivity and EOS rates. If curves agree before the cap, state that the collision estimate is not an artifact of the chosen maximum length.

## Experiment 4: cache and precomputation advantage

### Objective

Measure how much computation a miner can save when it has valid state from an earlier query, and connect that saving to the prefix-collision probabilities measured in Experiment 1.

### Workload families

Construct controlled query streams from the benchmark prompts:

1. **Independent prompts:** no exact token prefix in common.
2. **Exact repeats:** repeat the entire prompt with a new VRF seed.
3. **Shared-prefix prompts:** preserve an exact prefix of 8, 16, or 32 tokens and change the suffix.
4. **Near matches:** alter one token near the beginning or end of the prompt.
5. **Output-prefix reuse:** use different VRF seeds and retain/reuse decoder KV state only when the generated output prefix is exactly identical.

Use repeat probabilities \(\rho\in\{0,0.1,0.5,1.0\}\) to model different workloads. Do not assume benchmark prompts are equally likely in deployment.

### Cache configurations

Measure at least:

- fresh inference with no cache;
- prompt embedding cache only;
- exact prompt-prefill KV cache (\(M_1\) reuse);
- exact decoder KV-prefix cache for \(M_2\);
- a bounded LRU cache with several memory capacities.

For every configuration, measure wall-clock time, model FLOPs if available, peak memory, cache size, and fraction of work saved. Separate prompt-prefill savings from decoding savings. If the proof system is included, measure proving time separately: a model-state cache does not automatically imply reusable ZK-prover work.

Define the measured saving for condition \(j\) as

\[
a_j=1-\frac{T_{\mathrm{cached},j}}{T_{\mathrm{fresh},j}}.
\]

For decoder reuse, combine the speedup at prefix length \(\ell\) with \(C_\ell\) to estimate expected advantage under the benchmark distribution. Report this separately for exact prompt reuse, shared prompt-prefix reuse, and cross-query decoder-prefix reuse.

### Main outputs

1. A table of fresh versus cached latency, FLOPs, memory, and saved fraction.
2. Cache hit-rate and memory curves for bounded caches.
3. Speedup as a function of shared prompt-prefix length.
4. Decoder-cache speedup as a function of output-prefix length.
5. An empirical upper bound on \(\varepsilon_{\mathrm{total}}\) under each workload.

The protocol conclusion should not be that KV caching is impossible. Exact prompt-prefix caching can be useful in real serving workloads. The experiment should distinguish canonical \(M_1\) savings—which can be given to all \(M_2\) miners—from residual \(M_2\) reuse, whose expected advantage is governed by the prefix-collision curve.

## Reproducibility and controls

Record the model checkpoint hash, tokenizer version, Transformers/PyTorch versions, precision, hardware, batch size, sampling implementation, VRF/hash construction, prompt-selection seed, and all temperature/\(K\)/prompt-length settings. Release prompt IDs and tokenized inputs, not only random seeds.

Run these controls:

- deterministic greedy decoding, where collision should be essentially one until EOS;
- a uniform-token toy distribution, where the expected collision curve is known analytically;
- repeated execution with the same VRF transcript, which must reproduce the exact trace;
- cross-device execution, which must reproduce the same token sequence and commitments.

## Minimum publishable result

The minimum result should contain:

1. the WikiText-2 \(K=32\) prefix-collision curve with prompt-level confidence intervals;
2. the same curve for a small LAMBADA subset;
3. temperature versus top-token probability and collision curves;
4. \(K\)-sensitivity and EOS rates;
5. a cache-speedup table for fresh, M1-cache, and output-prefix-cache cases.

The strongest defensible conclusion is distributional: the experiments can show that honest GPT-2 traces rarely share long prefixes under a specified prompt distribution and sampler. They cannot, by themselves, establish computational independence for arbitrary adversarially chosen prompts.

