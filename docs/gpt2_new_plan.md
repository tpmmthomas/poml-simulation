# GPT-2 embedding-perturbation experiments for PoML

## 1. Purpose and scope

This document specifies two small experiments for the proposed one-time, challenge-conditioned perturbation of GPT-2 prompt embeddings. They support two separate claims:

1. **Utility:** the perturbation can be applied without making the frozen model useless on a range of standard language and commonsense benchmarks.
2. **Non-reusability at the proving boundary:** independently challenged prompts produce different values after the exact DeepProve normalization and quantization pipeline, so an old quantized trace, cache, witness, or proof cannot simply be replayed for a new challenge except with the measured collision probability.

These experiments do **not** prove that every prover operation must be recomputed. A changed tensor rules out exact replay of that committed object, but does not rule out model-weight commitments, proving-key preprocessing, lookup tables, static FFT work, or a cheaper algebraic update. The latter is the quantity that must remain bounded in the Section 5 reduction. Experiment 2 therefore includes a small direct measurement of reusable inference and proving work.

The intended claim is model- and implementation-specific: for the committed GPT-2 checkpoint, quantization configuration, perturbation law, and evaluated adversarial prompt classes, we observe the stated utility and separation properties. A finite experiment is not a universal cryptographic bound for every prompt or every circuit implementation.

## 2. Shared protocol and implementation

### 2.1 Model and software

Use **gpt2** (124M parameters) as the primary model, because GPT-2 is supported by the [public DeepProve code](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md). If resources permit, repeat the final operating point on **gpt2-medium** as a robustness check; do not make the larger model a prerequisite. Use the native GPT-2 byte-level BPE tokenizer, deterministic evaluation mode, and a fixed model/checkpoint hash. Record exact PyTorch, Transformers, CUDA, and DeepProve commits, arithmetic precision, batch size, hardware, and quantization configuration.

Use the EleutherAI Language Model Evaluation Harness for the utility baseline. Its official task list includes WikiText, LAMBADA, HellaSwag, PIQA, ARC, and other standard tasks, and its GPT-2 quickstart reports the metrics used below ([task list](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md), [GPT-2 quickstart](https://lm-evaluation-harness.readthedocs.io/getting_started/quickstart/)). Set **num_fewshot=0** so that each example has one unambiguous prompt prefix and no demonstrations are perturbed.

### 2.2 Perturbation law

For a tokenized prompt $x=(x_1,\ldots,x_p)$, let $E(x_i)$ denote the input representation at the insertion point. For GPT-2 this is the token-plus-position representation before the first transformer block. Apply noise once, to the prompt prefix:

$$
\widetilde e_{r,i}=E(x_i)+\eta_{r,i},\qquad
\eta_{r,i}=\sigma_{\rm abs}z_{r,i},\qquad
z_{r,i}\sim{\cal N}(0,I_d).
$$

The challenge-conditioned vectors must be generated deterministically in the protocol, for example by a domain-separated PRG/finite Gaussian expansion from the verified VRF value, query digest, and position $i$. The embedding-noise stream and per-token sampling stream must have different domain separators. The production experiment must use the same discretized/truncated Gaussian and finite-field encoding that will be placed inside the circuit; floating-point Gaussian noise is only a preliminary sanity check.

Let $s_E$ be the entrywise standard deviation of the committed GPT-2 embedding table. Report both the absolute scale and the dimensionless scale

$$
\alpha=\sigma_{\rm abs}/s_E.
$$

Use the candidate grid

$$
\sigma_{\rm abs}\in\{0,0.005,0.01,0.02,0.04\}.
$$

The $0.01$--$0.04$ range is a literature-anchored starting range from Gaussian embedding-perturbation work, not a value that can be assumed to transfer numerically to GPT-2 ([RESTA](https://arxiv.org/html/2501.16497), [Noiser](https://arxiv.org/html/2504.02911)). Select one operating point only after applying both the utility and separation criteria.

Generated continuation-token embeddings are not independently perturbed. The claim being tested is that the one-time perturbed prefix remains in the causal computation and therefore continues to affect later hidden states, even when sampled output tokens happen to be identical.

### 2.3 Exact comparison object

The relevant threshold is the quantized value used by the proving circuit, not the Euclidean norm of the raw noise. If $Q_\ell$ is the exact DeepProve quantizer at layer or cache boundary $\ell$, with one least-significant unit corresponding to real scale $\Delta_\ell$, compare

$$
Q_\ell(h_{\ell,r})\quad\text{and}\quad Q_\ell(h_{\ell,r'}).
$$

For a complete prefix tensor, equality means equality of every quantized coordinate, not merely equality of one row or one token. A nonzero integer difference is at least one LSB; the changed-coordinate fraction indicates whether the difference is negligible or spread through the tensor. Use DeepProve's actual clipping, outlier handling, lookup tables, and activation scales, because recalibration or clipping can otherwise erase challenge dependence.

## 3. Experiment 1 - utility after one-time perturbation

### 3.1 Datasets and metrics

Use five established tasks with different failure modes:

| Dataset/task | What it tests | Primary metric | Source |
|---|---|---|---|
| WikiText-2 test | ordinary language modelling and long-range prose | word perplexity (lower is better) | [WikiText task configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml) |
| LAMBADA test | discourse-context word prediction | exact last-word accuracy (or harness accuracy) | [Paperno et al.](https://arxiv.org/abs/1606.06031) |
| HellaSwag validation | commonsense sentence completion | length-normalized accuracy (acc_norm) | [Zellers et al.](https://arxiv.org/abs/1905.07830) |
| PIQA validation | physical commonsense reasoning | length-normalized accuracy (acc_norm) | [Bisk et al.](https://arxiv.org/abs/1911.11641) |
| ARC-Easy test | grade-school science knowledge | accuracy (acc) | [Clark et al.](https://arxiv.org/abs/1803.05457) |

ARC-Easy is preferable to ARC-Challenge for the primary GPT-2 experiment because ARC-Challenge can be close to chance for a small, non-instruction-tuned model. Add ARC-Challenge as a one-line robustness result only if it is not prohibitively noisy. The harness's WikiText configuration uses rolling log-likelihood and word perplexity, so arbitrary free-form generations are not the main quality measure.

For multiple-choice tasks, perturb the shared question/context prefix and score each answer choice as an unperturbed candidate continuation. For WikiText-2 and LAMBADA, perturb only the context and score the original target continuation. This tests the proposed one-time prefix operation rather than silently adding fresh noise to every target token.

### 3.2 Sampling and evaluation protocol

1. Run the clean, unperturbed baseline with the frozen checkpoint and standard zero-shot task templates.
2. For every example and every nonzero $\sigma_{\rm abs}$, evaluate 3--5 independent perturbation seeds. Use the same examples, templates, and candidate continuations in clean and perturbed conditions.
3. For a pilot, use 1,000 WikiText-2 examples and the full LAMBADA/HellaSwag/PIQA/ARC-Easy validation or test split when feasible. If the full split is too expensive, use a fixed, stratified 1,000-example subset per task and record the subset seed. Use the same subset at every noise level.
4. Keep sequences below the DeepProve-supported limit (for example, cap prompts at 512 GPT-2 tokens) and record truncation.
5. For a free-running sanity check only, generate a short VRF-sampled continuation for 100 fixed prompts. Do not make subjective generation quality the primary result.

### 3.3 Minimal outputs and decision rule

Report one table:

| Task | Clean metric | Perturbed metric at selected $\sigma$ | Absolute change | 95% paired CI |
|---|---:|---:|---:|---:|

Add one plot with noise scale on the x-axis and normalized utility on the y-axis (accuracy divided by clean accuracy, or clean perplexity divided by perturbed perplexity), with one line per task.

Use paired bootstrap confidence intervals over examples. Predeclare an engineering utility tolerance, for example: accuracy loss no more than 2 percentage points and WikiText perplexity increase no more than 5% relative to clean. These are protocol-design tolerances, not universal literature constants. Select the smallest candidate that passes utility and Experiment 2 separation criteria. If no candidate passes both, raw one-time perturbation is not adequate for the selected PoML configuration.

An optional next-token agreement diagnostic can explain failures. It is not the security claim: high-margin prompts may preserve exactly the same token while still having different hidden states and logits.

## 4. Experiment 2 - challenge-conditioned non-reusability

### 4.1 Prompt strata

Compare ordinary benchmark prompts with structurally targeted copy/repetition prompts:

1. **Standard prompts:** 100 examples from each Experiment 1 task (500 total), using the exact evaluation templates.
2. **BIG-bench repeat_copy_logic:** all 32 free-text examples. The task requires exact-string repetition, repetition counts, nested operations, and simple conditionals ([task description](https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/repeat_copy_logic/README.md), [task data](https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/repeat_copy_logic/task.json)). This is a small but established structural copy task, inspired by the repeat-copy task in [Neural Turing Machines](https://arxiv.org/abs/1410.5401). The related theoretical analysis [Repeat After Me](https://arxiv.org/abs/2402.01032) supports varying copy length, but is not itself a ready-made natural-language prompt dataset.
3. **Inverse-Scaling Prize Resisting Correction:** use the updated data-release version (7,344 examples in the published analysis; the release removes 21 duplicate-token cases, leaving 7,323 effective examples). Each prompt gives few-shot examples of verbatim repetition and then asks the model to repeat a sentence containing a typo or other grammatical anomaly; the answer classes are typically single-token alternatives such as the erroneous versus corrected word ([published analysis and task description](https://par.nsf.gov/servlets/purl/10544334), [data release](https://github.com/inverse-scaling/prize/tree/main/data-release)). This is the strongest existing large-scale source for a prior toward a particular next token while following an exact-copy instruction. The release notes that the updated task is named resisting-correction and supersedes the earlier quote-repetition version.
4. **Target-token concentration suite:** the structural construction in Section 4.3 below. It creates prompts for explicitly chosen one-token targets and retains only prompts whose clean GPT-2 distribution is demonstrably low-entropy and target-concentrated. This supplies scale and control without treating a generic adversarial benchmark as a repeat benchmark.

The Inverse-Scaling Prize also contains memo-trap, pattern-matching-suppression, and repetitive-algebra. They are useful optional controls for memorized continuations, breaking repetitive patterns, and few-shot label-frequency bias, respectively, but are not required for the minimal copy/repetition result. JailbreakBench and AdvBench should remain outside the core table because they target harmful behaviour rather than the concentrated copy regime needed here.

For Resisting Correction, retain each final prompt through its 'Output:' prefix and score the two released one-token class continuations directly from GPT-2's next-token logits. This avoids conflating the stress test with a long free-running generation. For repeat_copy_logic, retain the full target sequence and use exact-string agreement plus the first-output-token probability as the simple diagnostics.

The published Resisting Correction effect was established across a model-scaling study, whereas GPT-2 is a small base (not instruction-tuned) model. Therefore first report the clean GPT-2 class probability and margin. If GPT-2 does not show a concentrated class distribution on a subset, keep those examples as a public copy-control but do not relabel them as targeted-token cases; the controlled suite in Section 4.3 supplies that stratum.

### 4.2 Literature-informed construction decision

There is no single standard benchmark whose primary purpose is to make a decoder-only LM assign an arbitrarily selected token probability close to one. The closest established evidence comes from four complementary lines of work:

| Evidence | What it contributes | How it is used here |
|---|---|---|
| [Universal Adversarial Triggers](https://arxiv.org/abs/1908.07125) | A gradient-guided search finds short input-agnostic token sequences that induce a chosen prediction, including targeted GPT-2 continuations. | Optional white-box upper-bound stress test; do not present optimized triggers as natural prompts. |
| [AutoPrompt](https://aclanthology.org/2020.emnlp-main.346/) and [LinkPrompt](https://aclanthology.org/2024.naacl-long.360/) | Prompt search can elicit a desired label or prediction; LinkPrompt adds a natural-token constraint, while AutoPrompt is primarily demonstrated for masked LMs. | Methodological support for a held-out trigger construction and a warning that transfer from another model is not guaranteed. |
| [Distribution Prompting](https://aclanthology.org/2025.emnlp-main.1057.pdf) | GPT-2 experiments show that low-entropy target next-token distributions are easier to elicit than arbitrary high-entropy distributions. | Justifies filtering a fixed target-token suite by clean entropy and target probability, rather than claiming every chosen token is equally controllable. |
| [Repetition Neurons](https://aclanthology.org/2025.naacl-short.41/) | Activation analysis across pretrained LMs identifies units that become increasingly active during copying/repetition. | Mechanistic motivation for including a copy/repetition stratum; it is not itself a ready-made benchmark. |

Thus the core result should use public, inspectable tasks (BIG-bench and Resisting Correction) and a controlled target-token suite. Trigger optimization is a separate upper-bound experiment. This separation avoids overstating what the public datasets establish and keeps the main result reproducible without a costly search.

### 4.3 Structural target-token concentration suite

The goal is to construct a reproducible adversarial stratum in which the target output token is chosen in advance, rather than selecting prompts after seeing whichever token GPT-2 happens to prefer. This is grounded in work showing that low-entropy target distributions are easier to elicit from GPT-2 and other LMs ([Distribution Prompting](https://aclanthology.org/2025.emnlp-main.1057.pdf)), and in hard-prompt/trigger methods that optimize a short textual prefix for a desired prediction ([Universal Adversarial Triggers](https://arxiv.org/abs/1908.07125)). Keep two explicitly labelled subfamilies:

* **Visible-copy prompts (primary):** the target appears in a natural instruction such as "Repeat exactly one token: <target>. Output:". These model the user's "repeat after me" case and test whether copying remains highly skewed after challenge noise.
* **Target-absent triggers (optional upper bound):** the final carrier does not contain the target. A short prefix is found on construction carriers with the targeted-trigger objective below and evaluated on held-out carriers. These prompts test the strongest model-specific ability to force a token and should not be described as ordinary user traffic.

Use the following construction:

1. Choose 16--32 ordinary GPT-2 vocabulary entries that each tokenize to exactly one token (including the leading space where GPT-2 requires it). Exclude special tokens, whitespace-only tokens, and extremely rare byte fragments. Fix this target set before looking at test results.
2. Create a small library of neutral visible-copy templates, for example: "Answer with exactly one token: <target>. Next token:", "Repeat the requested symbol once. Requested symbol: <target>. Output:", and "The next token is <target>. Continue:".
3. Fill templates with 20 carrier strings that do not contain the target token (short neutral sentences, digits, and randomly generated alphanumeric strings). The target is allowed in the instruction field but must not occur accidentally in a carrier or delimiter. The default factorial design is 16 targets x 3 templates x 20 carriers = 960 candidates. Keep carriers disjoint between construction and evaluation, and retain at most 10 examples per target/template after filtering.
4. On an 80% construction split, compute the clean GPT-2 next-token distribution and retain examples satisfying a predeclared target condition, such as target probability \(p_y\geq0.90\), entropy in the lowest quartile, and logit margin \(\log p_y-\log p_{(2)}\geq\gamma\). If too few examples pass, relax the threshold once and record the change.
5. Freeze the resulting prompts and evaluate the remaining 20% carriers and independent perturbation seeds. Never use perturbed results to select prompts or targets.

For the target-absent subfamily, optimize a short hard trigger prefix \(\pi_y\) for each target \(y\) using the universal-trigger objective

$$
\min_{\pi_y}\;-\frac{1}{|{\cal C}_{\rm train}|}
\sum_{c\in{\cal C}_{\rm train}}
\log p_\theta(y\mid \pi_y\Vert c),
$$

then evaluate the trigger on held-out carriers. Report this as a white-box, model-specific stress test, not as a natural prompt distribution. The simple visible-copy template-and-filter version is the primary experiment because it is easier to audit and less likely to overfit. If the optimized prefix is itself reused across challenges, that does not weaken the experiment: it is the public prompt, while only the embedding perturbation is challenge-dependent.

For each retained prompt, store only the target token ID, tokenization length, clean target probability, clean entropy, and clean top-1 margin. These fields define concentration bins (for example \(p_y\in[0.90,0.99)\) and \(p_y\geq0.99\)) for the collision analysis; they are not a large auxiliary dataset.

### 4.4 Pairing and conditions

For every prompt $x$, draw 50 independent challenge pairs $(r,r')$ at each candidate noise scale. For the final selected scale, use 100 pairs for the BIG-bench, Resisting Correction, and target-token suites if the initial run has no collisions. Keep the prompt fixed within a pair. Run three conditions:

* **No embedding perturbation:** independent per-token VRF sampling only (baseline).
* **Embedding perturbation, common token stream:** independent embedding challenges but the same token-sampling stream, isolating the prefix perturbation.
* **Full protocol:** independent embedding and per-token sampling streams, with actual domain separation and statement binding.

The common-token-stream condition is a diagnostic. It answers whether a changed prompt trace prevents reuse even when the final sampled sequence is held as constant as possible.

### 4.5 Values to record

Run the exact quantized DeepProve graph and record only:

1. quantized perturbed embedding;
2. post-first-normalization representation;
3. first-layer $Q$, $K$, and $V$;
4. output of every transformer block (or every DeepProve re-quantization boundary);
5. final quantized logit vector;
6. selected token and, for short stress prompts, the generated token prefix.

Do not dump full floating-point tensors to the paper. Store them for audit and report only the compact statistics below.

### 4.6 Primary metrics

For each boundary $\ell$, define the **whole-prefix quantized collision rate**

$$
c_\ell=\Pr\left[Q_\ell(h_{\ell,r})=Q_\ell(h_{\ell,r'})\right].
$$

Also define the changed-coordinate fraction

$$
d_\ell=
\frac{\left\|Q_\ell(h_{\ell,r})-Q_\ell(h_{\ell,r'})\right\|_0}
{\#\text{coordinates}}.
$$

For pre-quantized tensors, optionally report the fraction of coordinates satisfying $|h_{\ell,r}-h_{\ell,r'}|\geq\Delta_\ell$. The security-facing statistics are $c_\ell$ and $d_\ell$ on the exact integer tensors used in the proof.

At the logits, report exact quantized-logit collision and token agreement separately. Same-token outputs are expected on high-margin repeat prompts and are not evidence that the computation was reusable. A valid proof must constrain challenge-conditioned logits and the sampler, not merely the final token string.

### 4.7 Prefix-collision measurement

For each stratum, compare the probability that two runs share the first $\ell$ generated tokens, for $\ell\in\{1,2,4,8,16,32\}$, under the three conditions:

| Stratum | Condition | $C_1$ | $C_4$ | $C_8$ | $C_{16}$ | $C_{32}$ |
|---|---|---:|---:|---:|---:|---:|

This makes the expected distinction explicit: perturbation may leave $C_\ell$ high for a skewed copy prompt while driving $c_\ell$ for the internal quantized trace to zero. Output-token diversity is not required for non-replayability.

If no collisions are observed, report a one-sided binomial upper confidence bound; the rule-of-three approximation $3/M$ for $M$ independent pairs is a convenient summary, not a cryptographic negligible bound.

### 4.8 Direct proof-replay test

On a small subset of pairs whose generated token sequence is identical, produce a DeepProve proof for challenge $r$. Verify it first under the original statement and then under the otherwise identical statement containing $r'$. Expected results:

* old proof under old challenge: accepted;
* old proof under new challenge: rejected;
* freshly generated proof under new challenge: accepted.

This test is meaningful only if the embedding-noise expansion, query digest, challenge, and token-sampling values are bound inside the proved relation. If full DeepProve integration is not yet implemented, perform the same test at the circuit/witness-commitment level and label it a pre-integration result rather than a ZKP result.

### 4.9 Minimal reuse-work measurement

To connect the experiment to Section 5, implement the strongest obvious precomputation attacker:

* before the challenge, precompute model-weight commitments, proving-key/lookup/FFT tables, token-index-independent preprocessing, and all clean prompt work allowed by the implementation;
* after $r$ is revealed, perform only the exact challenge-dependent update and complete the proof;
* measure fresh versus precomputed inference and proving time.

Report one compact table with fresh time, online update time, and saved fraction

$$
\rho=1-\frac{T_{\rm online\ update}}{T_{\rm fresh}}.
$$

Report separate $\rho_{\rm inf}$ and $\rho_{\rm pf}$. A changed cache or witness does not by itself imply a small $\rho$; this timing/update experiment supplies the percentage needed by Section 5. If the measured value violates

$$
\max(\rho_{\rm inf},\rho_{\rm pf})<\frac{1}{q_{\mathsf{ML}}+1},
$$

the protocol parameters or construction must change. Do not replace this bound with a statement that "the logits are different."

## 5. Main reported results

Keep the paper-facing output to four items:

1. **Utility table:** five tasks, clean versus selected perturbation, absolute changes, and paired 95% confidence intervals.
2. **Utility plot:** normalized performance versus $\sigma_{\rm abs}$.
3. **Trace-separation table/plot:** collision rates $c_\ell$, changed-coordinate fractions $d_\ell$, and final-logit/token agreement for standard, BIG-bench repeat-copy, Resisting Correction, and target-token prompts.
4. **Replay/reuse table:** proof replay pass/fail counts and measured $\rho_{\rm inf}$ and $\rho_{\rm pf}$.

Everything else - full tensors, seed lists, per-layer raw distributions, and qualitative examples - belongs in a reproducibility artifact or appendix.

## 6. Interpretation and limitations

The desired result is compatible with a highly skewed prompt:

* the token sequence can remain identical because perturbed logits preserve the same large-margin choices;
* prompt embeddings, normalized states, $Q/K/V$, block outputs, and/or quantized logits can nevertheless differ;
* a proof tied to $r$, the noise expansion, the quantized DeepProve graph, and the VRF sampler must then be recomputed or updated, and an old proof must fail under $r'$.

The experiment must not claim more than it measures. A nonzero difference in an intermediate activation does not establish that all prover work is fresh, and a finite zero-collision result is not a $2^{-128}$ guarantee. The correct paper statement is an empirical/model-specific cache or witness-collision bound plus a separately measured or assumed bound on reusable inference and proving work.

If one-time raw embedding noise causes unacceptable utility loss, that result rules out this simple construction for the selected GPT-2/DeepProve configuration; it does not justify silently introducing per-token noise. Fresh noise after every generated token would be a different protocol and requires a separate utility and proof-cost study.

## 7. Reproducibility checklist

- checkpoint, tokenizer, model, and circuit hashes;
- exact noise scale, Gaussian discretization/truncation, PRG/VRF domain separators, and seed derivation;
- DeepProve quantization scales, clipping, lookup tables, and finite-field encoding;
- task templates, prompt-selection seeds, truncation, and EOS rules;
- number of perturbation seeds and challenge pairs;
- hardware/software versions and timing methodology;
- code to reproduce the compact tables and confidence intervals.
