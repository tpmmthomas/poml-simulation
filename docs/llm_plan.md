# Autoregressive LLM Collision Probabilities for PoML

## Scope

The current question is deliberately narrower than the full PoML protocol analysis:

> Given the same prompt and two independent, honest autoregressive LLM samplings, what is the probability that the two runs produce the same transformer input at each generation step?

The transformer input at a generation step consists of the initial prompt followed by the tokens generated so far. This is distinct from the later adversarial question of whether a miner can deliberately precompute a high-probability response, and from the question of how much ZK-proof computation can actually be reused.

## Prefix-collision formulation

Fix a tokenized prompt (x), a model checkpoint, tokenizer, decoding policy, and an exactly (k)-token generation length. Let

\[
Y=(Y_1,\ldots,Y_k), \qquad Y'=(Y'_1,\ldots,Y'_k)
\]

be two independent samples from the model's autoregressive distribution.

Define

\[
C_\ell(x)=\Pr[Y_{1:\ell}=Y'_{1:\ell}\mid x].
\]

The interpretation is:

- (C_0(x)=1): both runs have the same initial prompt.
- (C_1(x)): the prompt plus the first generated token is the same.
- (C_2(x)): the prompt plus the first two generated tokens is the same.
- In general, (C_\ell(x)) is the probability that the transformer receives the same complete context through step \(\ell\).

At the call predicting token (t), the context contains (t-1) generated tokens, so the relevant collision probability is (C_{t-1}(x)).

These events are nested: once the two prefixes differ, they cannot later become equal as complete prefixes. Therefore, the useful object is the whole survival curve

\[
C_0(x),C_1(x),\ldots,C_k(x),
\]

not the probability that there is “some collision at some point” (which is trivially one because the initial prompt collides).

The first-mismatch position and common-prefix length are equivalent descriptions. In particular,

\[
\mathbb E[\operatorname{LCP}(Y,Y')\mid x]
=\sum_{\ell=1}^{k}C_\ell(x).
\]

This prefix quantity is likely more relevant to computational reuse than only full-response equality.

## Exact full-response collision probability

Let the post-processing sampling distribution at position (t) be

\[
p_t(v\mid x,y_{<t}),
\]

including temperature, top-(p), top-(k), repetition penalties, and any other decoding transformation. The sequence probability is

\[
P_x(y_{1:k})=
\prod_{t=1}^{k}p_t(y_t\mid x,y_{<t}).
\]

The two complete responses match exactly with probability

\[
C_k(x)=\Pr[Y=Y'\mid x]
=\sum_{y\in\mathcal V^k}P_x(y)^2.
\]

This is the order-2 Rényi, or collision, probability:

\[
C_k(x)=\exp\left(-H_2(P_x^{(k)})\right).
\]

The tokens within one response are not independent; the formula already accounts for their autoregressive dependence.

For a prefix (s), define its local next-token collision probability by

\[
q_t(s)=\sum_v p_t(v\mid x,s)^2.
\]

Then

\[
C_t(x)=
\sum_{s\in\mathcal V^{t-1}}P_x(s)^2q_t(s).
\]

The averaging is weighted toward high-probability prefixes. Consequently, one generally cannot estimate (C_k) by multiplying ordinary average per-token collision probabilities along one sampled or greedy trajectory.

If (q_t(s)\leq \bar q_t) for every reachable prefix, then

\[
C_k(x)\leq\prod_{t=1}^{k}\bar q_t.
\]

In particular, a positive collision-entropy rate is needed for a length-based claim:

\[
-\frac{1}{k}\log C_k(x)\geq h>0
\quad\Longrightarrow\quad
C_k(x)\leq e^{-hk}.
\]

Length alone is insufficient. A model could sample one random token and then emit a deterministic suffix, leaving (C_k(x)) bounded away from zero for all (k).

## Simple numerical illustrations

For a binary next-token distribution ((0.99,0.01)), two independent runs agree at that position with probability

\[
0.99^2+0.01^2=0.9802,
\]

not (0.99). If every position had this same distribution independently, then (C_k=0.9802^k), giving approximately (0.368) at (k=50) and (0.135) at (k=100).

For ((0.9,0.1)), the per-position collision probability is (0.82), so (C_{100}\approx2.4\times10^{-9}). These are toy calculations; real LLM distributions vary strongly by prompt, position, and prefix.

Shannon entropy or perplexity alone is not enough to determine (C_k). In general (H_2\leq H_1), so (e^{-H_1}) is a lower bound on collision probability rather than a useful upper bound. Collision entropy or full token distributions are the appropriate quantities.

At a fixed prefix, increasing temperature flattens the distribution and lowers the local (q_t(s)). However, the complete (C_k(x)) need not be theoretically monotone in temperature because changing early sampled tokens changes the later contexts.

## Prompt distributions and worst cases

If prompts are sampled from a specified distribution (X\sim D), the population collision probability is

\[
\bar C_k=\mathbb E_{X\sim D}[C_k(X)].
\]

This is meaningful only after defining (D). A mean over natural benchmark prompts can hide nearly deterministic prompts. For PoML, an adversary may deliberately choose such prompts, so prompt-level quantiles and worst cases matter more than only the average.

There is also a distinction between an honest random precomputation and a deliberately selected precomputation. If the precomputed response is itself an honest sample, the matching probability is (C_k(x)). If the adversary deliberately chooses a candidate sequence (y), its success probability is (P_x(y)), and the best fixed candidate succeeds with

\[
P_{\max}(x)=\max_yP_x(y),
\]

which can be much larger than (C_k(x)). Thus (C_k) describes natural collisions, not by itself a worst-case computational-independence theorem.

For fixed (k) and a fixed model, these probabilities are concrete constants. Even if (C_k) is practically tiny, it is not a negligible function of the cryptographic security parameter unless (k), model support, or entropy is made to scale with that parameter.

## Proposed experiment

Use a standard benchmark framework as a source of prompts, but add a custom repeated-sampling analysis. The EleutherAI `lm-evaluation-harness` supports standard academic tasks, custom prompts, and generation parameters, so it is suitable infrastructure for this purpose: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

The harness's ordinary accuracy score is not the target metric. For each prompt:

1. Fix the model, tokenizer, sampling policy, EOS convention, and maximum length (k).
2. Generate (n) independent responses.
3. For every prefix length (ell), estimate

   \[
   \widehat C_\ell(x)=
   \frac{2}{n(n-1)}
   \sum_{i<j}
   \mathbf 1\left[Y^{(i)}_{1:\ell}=Y^{(j)}_{1:\ell}\right].
   \]

4. Report the prefix-survival curve, full-response collision (\widehat C_k), first-mismatch distribution, and expected common-prefix length.
5. Aggregate across prompt categories using means, medians, high quantiles, and worst prompts.

Useful categories include factual/structured prompts, reasoning, code generation, summarization, and open-ended generation. Multiple-choice tasks are generally too short and constrained to reveal long-prefix behavior.

If selected-token log-probabilities are available, an alternative unbiased estimator is

\[
\widehat C_k^{\mathrm{lp}}
=\frac1n\sum_i
\exp\left(\sum_{t=1}^{k}
\log p_t(Y^{(i)}_t\mid x,Y^{(i)}_{<t})\right).
\]

This avoids waiting for exact duplicate full responses, which may be impractical when (C_k) is very small. For variable-length outputs, either force exactly (k) tokens or treat EOS as an explicit token.

## Existing empirical evidence

The literature does not appear to provide a universal top-token or full-response collision curve. Existing studies mainly show strong prompt/task dependence:

- Ouyang et al. report substantial non-determinism across 829 code-generation tasks and note that temperature zero does not guarantee deterministic outputs: [An Empirical Study of the Non-determinism of ChatGPT in Code Generation](https://arxiv.org/abs/2308.02828).
- Carandang et al. directly measure string-equivalent responses across repeated same-prompt clinical-note generations and find exact-string consistency ranging widely across models: [Are LLMs reliable?](https://aclanthology.org/2025.acl-industry.99/).
- Shur-Ofry et al. find concentrated single-model output distributions and increased diversity under higher temperature, while still observing less diversity than in human responses: [Growing a Tail](https://arxiv.org/abs/2411.02989).
- Wang et al. find highly heterogeneous token entropy in reasoning traces, with a minority of high-entropy “fork” tokens and many low-entropy tokens: [Beyond the 80/20 Rule](https://arxiv.org/abs/2506.01939).
- Lovering et al. provide controlled next-token probability measurements showing mode concentration in synthetic numeric contexts, but these are not natural long-form responses: [Language Model Probabilities are Not Calibrated in Numeric Contexts](https://arxiv.org/abs/2410.16007).

## Current conclusion

The proposed benchmark experiment is the right next step for the honest-collision question, provided the result is phrased conditionally:

> For a specified model, decoding policy, prompt distribution, and response length, measure the probability that two independent samples share the same transformer context through each generation step.

The strongest output is likely to be a prompt-stratified (C_\ell) curve rather than a single claim that “long LLM responses almost never collide.”
 
## Follow-up: the consensus boundary and KV-cache reuse

### Proposed M1/M2 boundary

For the honest-miner reuse question, it is reasonable to define the deterministic prompt-prefill computation as \(M_1\), with consensus-relevant work beginning when the first generated token is sampled. If \(x\) is the tokenized prompt, then \(C_0(x)=x\) is common to both executions, and the first potentially different transformer context is \(x\mathbin\Vert Y_1\). The subsequent contexts are

\[
C_0(x),\quad C_1(x)=x\mathbin\Vert Y_1,\quad \ldots,\quad C_k(x)=x\mathbin\Vert Y_{1:k}.
\]

The boundary needs one implementation detail stated explicitly: prompt prefill normally computes the KV states for the prompt and the logits used to sample \(Y_1\). Thus \(M_2\) should include the random sampling of \(Y_1\) and every subsequent decode step, while the deterministic construction of the prompt-side state is \(M_1\). Excluding \(M_1\) from consensus does not make it free; it only says that it is supplied or accounted for outside the consensus lottery. A delegated prefill/KV result would need to be bound to the exact tokenized prompt, model/configuration, adapter, and positional-encoding settings.

A fixed response length \(k\) is the cleanest first formalization. A query-declared expected length \(n\) can be supported, but \(n\) must be committed before mining and the proof must verify the actual number of decode steps (including the EOS convention). Fees, work weight, and the probability of winning then need to scale with verified work; otherwise short or prematurely terminated queries can be used to manipulate the lottery. Discrete length/work classes are a possible compromise.

### What KV caching actually permits

There are two different forms of caching:

1. **In-request caching.** During ordinary autoregressive decoding, the KV states for previously generated tokens are retained and reused. This is part of the honest baseline and should not be treated as an adversarial shortcut.
2. **Cross-request prefix caching.** Serving engines can also reuse KV blocks for an exact compatible token prefix from an earlier request. This is supported for repeated prompts, shared system or few-shot prefixes, and multi-turn conversations; systems such as TensorRT-LLM and SGLang document reuse of common prefixes across requests ([TensorRT-LLM KV-cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html), [SGLang RadixAttention](https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention)).

Therefore it would be too strong to claim that KV caches are always per-query and then discarded, or that identical prompts across queries are necessarily rare. For unrelated, high-diversity prompts, exact long-prefix hits may indeed be uncommon; templated workloads can have the opposite behavior because a long system/instruction prefix is shared by many requests. If two stochastic executions diverge at token \(j\), an exact cache cannot normally be used for later positions, so the honest prefix-collision curve \(C_\ell(x)\) is directly relevant to how much of a generated continuation could be reused.

Retention is resource- and policy-dependent rather than impossible. KV caches consume memory proportional to the number of layers, KV heads, head dimension, token count, and precision:

\[
\text{bytes/token}\approx 2L\,n_{\mathrm{KV}}\,d_{\mathrm{head}}\,b,
\]

where the factor 2 is for keys and values and \(b\) is bytes per element. For illustration, \(L=32\), \(n_{\mathrm{KV}}=8\), \(d_{\mathrm{head}}=128\), and FP16 require about 128 KiB per token, or about 512 MiB for a 4,096-token sequence. Larger models can require around a gigabyte or more for one long sequence. Consequently, engines use paging, finite pools, and eviction; TensorRT-LLM, for example, describes LRU-style eviction and memory-fraction limits ([KV-cache system](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)). Host-memory offload or persistent caches can extend retention, so “infeasible to store universally” is not a safe universal statement.

### Safer PoML claim and experiment

The defensible protocol assumption is:

> KV caching is memory-bounded and policy-dependent. In-request caching is part of the honest decoding baseline. Cross-request reuse is possible only for exact compatible prefixes that remain resident; its rate is workload-dependent, low for diverse independent prompts, and potentially substantial for repeated or templated prompts. The protocol should either measure/bound this reusable work or explicitly prevent it, rather than assume it is zero.

A low repeat probability under an organic workload is not a worst-case security bound: an adversary that chooses its own queries can deliberately request repeated or templated prompts. Query fees may make that strategy uneconomic, but that is an incentive calculation rather than a cryptographic impossibility.

The experiment should therefore measure more than full-response collisions. In addition to the \(C_\ell(x)\) curves, construct query streams with (i) independent prompts, (ii) repeated exact prompts, and (iii) shared system/few-shot prefixes with different suffixes. Run with prefix caching disabled and enabled under a finite memory budget, and report longest-compatible-prefix lengths, cache-hit rates, eviction rates, and actual decode-compute savings. This separates natural stochastic prefix collisions from deterministic cross-query cache reuse and gives the paper a workload-conditional claim instead of an unsupported “super-low” probability assertion.
 
## Follow-up: what can be placed in M1 and how to bound cache advantage

The prompt-side KV state can indeed be computed before any output token is sampled. Define

\[
K_x,V_x=\operatorname{Prefill}(x),
\]

where \(x\) is the exact tokenized prompt and all model/configuration parameters are fixed. This is standard prefix-cache prefill: the resulting state can be copied and used to generate several different continuations ([Hugging Face prefix-caching example](https://huggingface.co/docs/transformers/main/kv_cache#prefill-a-cache-prefix-caching)). It is therefore feasible to put the prompt-side computation in \(M_1\), or to delegate it to a non-mining worker.

This does **not** put the whole KV cache in \(M_1\). KV states for generated tokens depend on the sampled sequence:

\[
(K,V)(x\mathbin\Vert Y_{1:\ell})
\]

cannot be known before \(Y_{1:\ell}\) is sampled. These continuation states are part of \(M_2\). Consequently, a miner with a cached \((K_x,V_x)\) has no consensus advantage if all miners can obtain the same \(M_1\) result and prompt prefill is excluded from the PoML work. A cached continuation can still provide an advantage when its token prefix happens to match the VRF-selected path.

### A reusable-work bound

Let \(P_x(u)\) be the probability that sampling from prompt \(x\) begins with prefix \(u\), and let \(S_\ell\) be the set of length-\(\ell\) continuation prefixes for which a miner has precomputed and retained an exact KV state. If \(w_\ell\) is the decode work saved when such a state is usable, the expected saved model work is bounded by

\[
\mathbb E[W_{\mathrm{saved}}]
\;\leq\;
\sum_{\ell=1}^{k} w_\ell\,\Pr[Y_{1:\ell}\in S_\ell]
\;=\;
\sum_{\ell=1}^{k} w_\ell\sum_{u\in S_\ell}P_x(u).
\]

For one independently sampled precomputation, the expectation of the inner probability is exactly the honest prefix-collision probability \(C_\ell(x)\). For a deliberately chosen single trace, it becomes \(P_x(u)\), and is upper-bounded by the most likely prefix probability \(P_{\max,\ell}(x)\). If the miner stores at most \(B_\ell\) candidate prefixes of length \(\ell\), a simple bound is

\[
\Pr[Y_{1:\ell}\in S_\ell]
\leq \min\{1,\,B_\ell P_{\max,\ell}(x)\}.
\]

The resulting saved-work fraction \(\alpha=\mathbb E[W_{\mathrm{saved}}]/W_{\mathrm{decode}}\) gives a throughput advantage of roughly \(1/(1-\alpha)\) if mining success is proportional to the rate of completed valid inferences. The exact block-winning advantage still depends on PoML's reward and scheduling rule. There is no prompt-independent zero bound: for a nearly deterministic prompt, \(P_{\max,\ell}(x)\) can be close to one; for high-entropy long continuations, the bound can become small.

Thus the strongest claim is not that stored KV state gives no advantage. It is that deterministic prompt-prefill state is outside consensus and gives no *relative* advantage when made equally available, while any remaining advantage from cached stochastic continuations is captured by the prefix-probability bound above. Eliminating that latter advantage requires a fresh unpredictable input that invalidates old continuation states (which generally also sacrifices prompt-prefix reuse), or an explicit cache/work assumption in the security analysis. Fresh randomness in the ZK proof may further dilute the model-compute advantage, but that requires a separate proof or measurement.
 
## Clarification: when the KV-reuse probability equals C_l(x)

The equality with \(C_\ell(x)\) holds only for one specific experiment. Let \(Y\) be the current continuation and \(U\) be one previously cached continuation such that:

- both start from the same exact prompt and model configuration;
- \(U\) and \(Y\) are independent;
- both were honestly sampled from the same decoding distribution; and
- the KV state for \(U\) remains available.

If \(L\) is the longest shared generated-token prefix, then

\[
\Pr[L\geq \ell]
=\Pr[Y_{1:\ell}=U_{1:\ell}]
=C_\ell(x).
\]

Thus \(C_\ell(x)\) is the probability of reusing that one cached continuation through at least \(\ell\) generated tokens. It is not the probability of “any cache reuse” in every setting. In particular:

- reuse of the prompt-side \(M_1\) cache is deterministic for the same prompt, so its probability is one and it is excluded from consensus;
- the probability of reusing at least one generated-token state from one honest prior trace is \(C_1(x)\);
- the probability of reusing exactly \(\ell\) generated tokens and then diverging is \(C_\ell(x)-C_{\ell+1}(x)\);
- the expected reusable generated-prefix length from one honest prior trace is \(\sum_{\ell=1}^k C_\ell(x)\), or \(\sum_\ell w_\ell C_\ell(x)\) when decode steps have different costs.

For \(m\) independently and honestly sampled cached traces, let \(P_\ell(u)=\Pr[Y_{1:\ell}=u]\). The probability that at least one cached trace matches through \(\ell\) is

\[
R_{m,\ell}(x)
=\sum_u P_\ell(u)\left[1-(1-P_\ell(u))^m\right]
\leq \min\{1,mC_\ell(x)\}.
\]

For adversarially selected cached prefixes \(S_\ell\), the relevant probability is instead

\[
\Pr[Y_{1:\ell}\in S_\ell]
=\sum_{u\in S_\ell}P_\ell(u),
\]

which can be much larger than \(C_\ell(x)\). Therefore \(C_\ell(x)\) is the correct honest single-cache baseline, while multiple caches, adversarial selection, different prompts, cache eviction, and other configurations require separate terms.
 
## Adversarial case: chosen-prompt attacks and fee deterrence

### Threat model

A chosen-prompt adversary acts both as query issuer and miner. It selects a prompt \(x\) whose continuation distribution is highly concentrated, precomputes one or more likely continuation states, and later uses the VRF-derived sampling randomness honestly. Temperature does not give a worst-case solution because prompts can be constructed whose distribution remains concentrated even after temperature scaling.

Two probabilities must be kept separate:

- \(r_x\): the probability that the VRF-sampled continuation falls inside the adversary's precomputed cache, such as \(P_x(S_k)\) for a set of full traces or a weighted prefix-reuse probability for partial caches;
- \(p=D/2^\kappa\): the probability that a completed inference--proof pair wins the PoML hash lottery.

Cache reuse changes the cost or rate at which the adversary produces completed tickets. It does not change \(p\) for any completed ticket.

### One-shot expected-value model

Let:

- \(b\) be the nonrefundable component of a self-issued query fee;
- \(C_{\mathsf{pre}}\) be the precomputation cost;
- \(C_{\mathsf{online}}\) be the inference/proof cost paid when the cache hits;
- \(V\) be the net external value of causing the attack ticket to win a block, excluding fee transfers from the adversary's user identity back to its miner identity.

If the block contains only the adversary's own query and there is no subsidy, transaction-fee revenue, MEV, or other block-control value, then \(V=0\): the solve/include payment is only a transfer within the attacker's coalition. The attack is then unprofitable whenever it has any positive burn or computation cost. The interesting case is when the extra chosen-prompt ticket increases the chance of capturing fees from other users' queries.

If the adversary only produces a ticket when its full precomputation matches, a simplified expected utility is

\[
U_{\mathsf{one}}(x)
=r_x\bigl(pV-C_{\mathsf{online}}\bigr)
-C_{\mathsf{pre}}-b.
\]

A sufficient deterrence condition is therefore

\[
b+C_{\mathsf{pre}}+r_xC_{\mathsf{online}}
\geq r_xpV.
\]

For a prompt-independent conservative rule, set \(r_x=1\) and do not rely on adversarial compute cost, giving

\[
b\geq pV_{\max}.
\]

The solve and inclusion components of the attacker's own query fee are transfers within the attacker coalition if it wins; they do not deter the attack. Only a true burn, or a payment that the coalition cannot recover, counts as \(b\). If another miner solves the query, the attacker's loss is larger, so treating \(b\) as its only fee loss is conservative.

### Extra-ticket model

Suppose the adversary can complete \(m\) ordinary tickets and uses \(B\) chosen-prompt queries to obtain \(B\) additional cheap tickets. The incremental probability of winning the round is

\[
\Delta P_{m,B}
=(1-p)^m\bigl[1-(1-p)^B\bigr].
\]

If every chosen query is paid for before the mining attempt, an upper bound on incremental utility is

\[
\Delta U(B)
\leq \Delta P_{m,B}V-Bb-C_{\mathsf{attack}}(B).
\]

Because \([1-(1-p)^B]/B\leq p\), the rule \(b\geq pV_{\max}\) is sufficient for all \(B\), assuming one paid query creates at most one extra ticket opportunity. A sharper rule can use \(r_x\), the actual cached-prefix savings, and the adversary's precomputation budget, but a worst-case protocol rule should not assume these are small.

### Protocol conditions needed for this fee argument

The current draft permits same-block registration and mining, using \(Q.h_{\mathsf{reg}}\leq h\), and applies the fee atomically only if the candidate block becomes canonical. A losing candidate therefore does not pay the proposed burn for its privately added self-query. Under this rule, the expected-value calculation above does not apply: the fee is charged mainly on success rather than per attack opportunity. With negligible attack computation, deterrence would require approximately \(Bb\geq V\) conditional on winning, which can be far more expensive than \(b\geq pV\).

A workable fee-based design therefore needs:

1. **Canonical pre-registration.** Require \(h_{\mathsf{reg}}<h\), so the query fee is irreversibly charged before the query can produce a ticket.
2. **Bounded eligibility.** A registration must buy a bounded number \(N_q\) of ticket opportunities, ideally one height/parent. If the query remains eligible for several opportunities, use \(b\geq N_qpV_{\max}\), or charge a renewal burn for every eligibility window.
3. **A bounded prize.** Cap the net fee value that one additional ticket can capture. Without a finite \(V_{\max}\), no finite fixed burn can guarantee negative expected value. A maximum inference-prefix length and uniform fee cap give such a bound.
4. **Unbiasable query selection.** Derive query order from the canonical pool and public randomness rather than leaving uniform selection as an unenforced honest policy. An adversary can then increase the chance of selecting its prompts only by registering more of them and paying more burns.
5. **Work-proportional pricing.** For LLM queries with declared length/work weight \(w\), scale both lottery weight and burn with the verified work class, rather than using one fee for all output lengths.

The response-settlement mechanism can reduce \(V\). If a losing miner receives the solve fee with probability \(s\), the block-winning premium per genuine query is approximately

\[
\phi_{\mathsf{include}}+(1-s)\phi_{\mathsf{solve}},
\]

rather than the full query fee. With at most \(K_{\max}\) external queries whose rewards are at stake, one possible conservative prize cap is

\[
V_{\max}
=K_{\max}\left(\phi_{\mathsf{include}}+(1-s)\phi_{\mathsf{solve}}\right),
\]

under the stated assumption that there is no block subsidy, transaction-fee prize, MEV, double-spend value, or other external benefit from controlling the block.

### Scope of the conclusion

This mechanism can establish that a chosen-prompt attack is unprofitable for a rational, risk-neutral adversary under the fee-only utility model. It cannot replace the computational-independence assumption in a Byzantine security proof: an adversary seeking chain control may rationally burn money, and any unmodelled block-control value belongs in \(V\). The paper should therefore present the fee result as an economic deterrence theorem with explicit assumptions, not as an unconditional repair of the Backbone reduction.

## Clarification: no subsidy is not the same as no reward

Under a narrow monetary model in which there is no block subsidy, no fees from other users, no transaction fees or MEV, and the adversary controls both the query identity and the mining identity, a chosen-prompt attack has no positive monetary upside. Its own solve/include payment is only a transfer within the coalition; the attacker's utility is then the negative burn and computation cost (or zero if both are literally zero). This conclusion is model-independent: it applies to an LLM, a diffusion model, or any other PoML model.

However, removing a flat subsidy does not remove the reward in the current protocol. Fees paid by genuine users are external value. If an extra cheap chosen-prompt ticket increases the probability of winning a block containing those queries, then the marginal utility is approximately

\[
\Delta U
=\Delta P\,F_{\mathrm{external}}
-\text{burn}
-\text{incremental computation},
\]

where \(\Delta P\) is the extra block-winning probability created by the attack. Response settlement may reduce \(F_{\mathrm{external}}\), but generally does not make it identically zero because inclusion fees and delayed/non-guaranteed settlement remain valuable. Thus fee-only rewards can still create an incentive; the fee structure must make this expression non-positive.

There is also a distinction between economics and consensus security. A chosen prompt can violate the intended fresh-computation or \(q_{\mathsf{ML}}\)-bound even when it is financially unprofitable. An adversary may accept a monetary loss for censorship, chain control, griefing, or another external objective. Therefore “no rational monetary incentive when there is no external reward” is a valid economic observation, but it is not by itself a computational-independence proof.

## Fee design with external value and consecutive adversarial blocks

The fee condition should be stated in terms of the *marginal external value* of an additional chosen-prompt ticket. Let \(p=D/2^\kappa\) be the success probability of one completed ticket, let \(m\) be the number of ordinary tickets already in the candidate, and let \(V_{\max}\) bound the external value that can be captured if this ticket is decisive for the block. If a chosen query costs a non-refundable burn \(b\), then the marginal expected utility of the \(j\)-th chosen query is at most

\[
\Delta U_j
\leq p(1-p)^{m+j-1}V_{\max}-b-c_j,
\]

where \(c_j\) is any additional inference, proof, or precomputation cost. The factor \((1-p)^{m+j-1}\) accounts for the fact that the new ticket matters only if all earlier tickets fail. Hence the conservative rule

\[
b\;\geq\;pV_{\max}
\]

does make every additional chosen query weakly unprofitable, provided that (i) the burn is paid before the query creates its ticket, (ii) one paid query creates at most one ticket opportunity, and (iii) \(V_{\max}\) is a genuine finite upper bound. If \(b<pV_{\max}\), the total expected utility is generally concave rather than monotonically decreasing: adding queries initially may be profitable, but the marginal benefit falls geometrically while the burn grows linearly.

The bound must be expanded when a registration can be used repeatedly. If one burn buys \(N_q\) ticket opportunities over several parents or heights, a sufficient rule is

\[
b\;\geq\;N_qpV_{\max},
\]

or, preferably, charge a fresh burn for each eligibility window. If the adversary may extract value over \(H\) consecutive blocks, \(V_{\max}\) must bound the *cumulative* external value over that horizon. There is no finite fee-only bound when the relevant chain-control value is unbounded or when the attack can continue indefinitely.

The current same-block registration rule is incompatible with the simple \(b>pV_{\max}\) argument. With \(h_{\mathsf{reg}}\leq h\), a losing candidate's private registration is not canonical and its burn is never charged. The protocol should require canonical pre-registration (for example, \(h_{\mathsf{reg}}<h\)) and a bounded one-shot eligibility window before using this fee proof.

No worst-case argument should assume that the miner cannot recover an \(M_1\) payment. If the adversary controls both a mining identity and an \(M_1\)-worker identity, that payment is an internal transfer and has zero coalition-level value. The same coalition assumption should be used for solve/include fees on self-issued queries. Only burns or payments that provably leave the coalition are real attack costs.

With reliable response settlement, the relevant \(V_{\max}\) may be much smaller than the gross block fee pool: the extra value of winning is approximately the inclusion fees plus any solve fees that would otherwise be lost. A block query cap and a fee cap are therefore needed to make \(V_{\max}\) finite. The resulting statement is an economic deterrence result for a rational adversary; it does not protect against an adversary willing to spend money for chain-control or other non-monetary objectives.
 
## Audit of the proposed four-step LLM argument

The four steps form a plausible *conditional engineering and economic argument*, but they do not yet establish the unconditional \(q_{\mathsf{ML}}\)-bound or the Backbone security theorem. The key distinction is between honest, distributional evidence and a worst-case adversarial security property.

### Step-by-step assessment

1. **Low collision probability for genuine queries.** This can support a statement about a specified workload distribution \(D_{\mathrm{honest}}\), model, tokenizer, decoding policy, and length. It does not imply low collision for every valid prompt. The experiment should report the weighted prefix-reuse cost (and prompt-level quantiles), not only the probability of a full-response duplicate. A fixed model and fixed length also give a concrete constant, not a negligible function of the cryptographic security parameter.

2. **Move deterministic repeatable work to \(M_1\).** This is conceptually sound if \(M_1\) is defined exactly and its output is canonical, available, and bound to the model/configuration. It does not remove the cost from the system; it removes it from consensus. Communication, storage, delegation, and unequal access to the \(M_1\) state still need to be specified.

3. **Infer \(\varepsilon\)-computational independence from low honest collisions.** This is the main logical gap. Honest collision probability measures two independent samples, whereas computational independence quantifies an adversary's best reusable precomputation, possibly with an adaptively chosen prompt, many cached branches, approximate states, batching, and proof-level reuse. The implication becomes valid only after defining a distributional CIA and proving/estimating a cost-saving bound such as

   \[
   \Pr\!\left[W_{\mathrm{saved}}/W_{\mathrm{decode}}>\varepsilon\right]
   \leq \delta
   \]

   for the specified honest workload, or after separately bounding adversarial prefix mass \(P_x(S_\ell)\). A low full-trace collision can coexist with substantial savings from the first few generated tokens, so \(C_k\) alone is insufficient.

4. **Chosen prompts are economically unprofitable.** This is a separate rational-adversary result. With canonical pre-registration, one ticket opportunity per paid query, and a finite bound \(V_{\max}\) on marginal external block value, \(b>pV_{\max}\) is a conservative sufficient condition. It must be adjusted for repeated eligibility, consecutive-block horizons, multiple candidate branches, and any other block-control value. Under the current same-block registration rule, failed attack attempts do not pay the burn, so the condition does not apply without a protocol change.

### Further gaps to close

- **Worst-case prompts versus benchmark prompts.** If arbitrary prompts are accepted, an adversary can select a high-concentration prompt. Benchmark results cannot rule this out; they only characterize the chosen workload distribution.
- **Proof cost.** The model-side prefix bound does not establish that SNARK proving, VRF evaluation, encryption, serialization, or verification has the same independence. Each reusable component needs its own cost bound.
- **Seed and query timing.** The security game must specify when the prompt, query identifier, transaction fields, and parent binding become fixed relative to the VRF seed. Same-block adaptive registration or transaction grinding can give the adversary extra choices.
- **Many cached states.** \(C_\ell(x)\) is the one-honest-cache baseline. An adversary can precompute a tree of prefixes; the relevant quantity is the probability mass covered by that tree, constrained by its computation and storage budget.
- **Variable output length.** Early EOS or a declared short length can create cheap tickets unless the number of verified decode steps and fee/lottery weight are bound before mining.
- **Economic versus Byzantine security.** Negative expected monetary value does not stop an adversary pursuing censorship, chain control, griefing, or another external objective. Such value must either be bounded and included in \(V_{\max}\), or the claim must explicitly be limited to rational fee-maximizing miners.

### Narrowest defensible conclusion

The paper can safely claim something of the following form:

> For a specified LLM, decoding policy, prompt distribution, response-length class, cache policy, proof system, and eligibility/fee rule, experiments estimate low honest prefix reuse after deterministic prefill is excluded. We model the residual honest reusable work by a parameter \(\varepsilon\). For self-issued chosen-prompt queries, assuming canonical pre-registration, bounded ticket opportunities, a finite marginal external value \(V_{\max}\), and rational fee-only miners, a burn satisfying \(b>pV_{\max}\) (with the appropriate horizon multiplier) makes the attack non-profitable.

This is a useful conditional extension, but it should not be presented as proving that arbitrary-prompt autoregressive LLM inference satisfies the same worst-case computational-independence assumption currently used for the diffusion-model Backbone reduction. To retain that theorem, the protocol needs an additional worst-case mechanism—such as a fresh unpredictable model input that invalidates prior continuation states—or a proof that the adversary's cached branches cannot reduce the verified inference--proof cost beyond the required bound.

## Expanded experiment plan for the LLM candidate

The experiment should estimate both natural prefix collisions and actual reusable computation. A recommended protocol is:

1. **Fix the deployed computation.** Specify the exact checkpoint, tokenizer/chat template, quantization and fixed-point representation used by the ZK circuit, decoding rule (temperature, top-p/top-k, repetition penalties), VRF-to-sampler mapping, EOS convention, and output length class \(k\). Any nondeterminism from kernels or hardware must be disabled or included in the measurement.
2. **Build a stratified prompt corpus.** Use long-form prompts from factual/structured, reasoning, code, summarization, and open-ended tasks. Keep multiple-choice tasks separate because their short, constrained outputs are not representative. Record the tokenized prompt length and category.
3. **Generate independent traces.** For each prompt, generate \(n\) samples with independent seeds. For every pair or every sampled trace, estimate \(C_\ell(x)\) for all \(\ell\leq k\), the first-mismatch distribution, the expected longest common prefix, and \(C_k(x)\). Report confidence intervals and prompt-level quantiles; if no full duplicates are observed, report an upper confidence bound rather than zero.
4. **Measure weighted savings.** Associate each decode step with its measured inference/proof cost \(w_\ell\) and report \(\sum_\ell w_\ell C_\ell(x)\) or the corresponding saved-work fraction. This is the quantity needed for an \(\varepsilon\)-CIA claim; a small full-response collision alone is not enough.
5. **Evaluate cache regimes.** Replay streams containing (i) independent prompts, (ii) repeated exact prompts, (iii) shared system/few-shot prefixes with different suffixes, and (iv) repeated prompts with independent generated continuations. Run with prefix caching disabled and enabled, finite GPU memory, different batch sizes, arrival orders, and eviction policies. Report longest compatible prefix, hit rate, eviction rate, wall-clock/FLOP savings, and memory occupancy.
6. **Stress chosen prompts separately.** Construct high-concentration prompts and let an attacker precompute one trace, a set of traces, or a prefix tree subject to an explicit compute/storage budget. Measure covered probability mass \(P_x(S_\ell)\), not just honest \(C_\ell(x)\). This is a worst-case stress test, not part of the honest-workload average.
7. **Benchmark the proof layer.** Measure prover time and memory with a fresh statement/witness versus reused prompt-side state, reused generated-prefix state, and repeated proof attempts. The model collision experiment cannot establish ZK-prover independence.
8. **Use held-out prompts and report uncertainty.** Select prompt categories and attack templates before evaluation, keep a held-out set, and report mean, median, high quantiles, and worst observed values. The headline should be workload-conditional: “under this distribution and configuration, the measured reusable-work fraction is at most …,” not “LLM traces almost never collide.”

## Options for handling chosen prompts

There are three defensible design choices.

### Option A: weaken the CIA scope

Define \(\varepsilon\)-computational independence only for \(x\sim D_{\mathrm{honest}}\), and handle self-issued chosen prompts with the fee theorem. This is the least invasive option and is a reasonable *rational-adversary* extension, but it cannot support the current worst-case Byzantine Backbone theorem because that theorem permits the adversary to choose its own inputs. The paper must state the changed threat model explicitly.

### Option B: restrict the prompt distribution

Allow only prompts from an externally generated, authenticated, or otherwise constrained workload for which the CIA experiment applies. This preserves a distributional security claim but sacrifices arbitrary user prompts and introduces an admission/trust problem. It is clean theoretically but less attractive for a general-purpose LLM service.

### Option C: modify sampling to impose a worst-case entropy floor

Temperature alone is not a worst-case solution: for any finite temperature, a prompt can produce an arbitrarily large logit gap and leave the top-token probability arbitrarily close to one. A no-retraining alternative is to mix the model distribution with a broad proposal distribution at every token,

\[
\widetilde p_t=(1-\lambda)p_t+\lambda q_0.
\]

If \(q_0\) is uniform over a vocabulary of size \(|\mathcal V|\), then every conditional token probability is at most

\[
\alpha=1-\lambda+\lambda/|\mathcal V|,
\]

so exact prefix collisions satisfy the prompt-independent bound \(C_\ell(x)\leq \alpha^\ell\). A cached set of \(B_\ell\) prefixes covers at most \(\min\{1,B_\ell\alpha^\ell\}\). This gives a genuine worst-case entropy argument without retraining, but the required \(\lambda\) may harm output quality, especially with uniform random tokens. A less destructive proposal distribution can be tested, but its maximum probability must be included in the bound. Hidden-state noise or inference-time dropout without retraining has no comparable guarantee and may substantially damage quality.

The best practical presentation is likely Option A plus an explicit discussion of Option C as a possible stronger variant. Do not claim that high temperature or informal noise injection restores worst-case CIA unless the entropy bound and quality impact are demonstrated.

## Two-stage protocol: accepted M1 followed by M2 mining

The proposed two-stage workflow is sensible and fixes the fee-timing problem:

1. **M1 acceptance.** A query is registered together with the exact prompt/configuration. A worker computes the deterministic prompt-prefill state and posts a commitment (and, if miners are not expected to recompute it, a data-availability object and proof of correct prefill). The registration fee is charged and the burn occurs at this stage. The query becomes mineable only after canonical acceptance, ideally after an activation delay.
2. **M2 mining.** Miners use the accepted M1 commitment/state, derive the VRF seed from the parent/query/miner binding, sample the output, compute generated-token states and the ZK proof, and obtain at most one ticket for that query in the eligibility window.

The commitment must bind the tokenized prompt, model hash, tokenizer/chat template, adapters, positional-encoding settings, precision, output-length class, and M1 serialization. A hash alone is not enough if miners need the KV tensor; the state must also be available or miners must recompute it. If M1 correctness matters, it needs a verifiable prefill proof or an agreed replication rule even though its cost is excluded from the lottery.

This two-stage design removes deterministic prompt-prefill work from consensus and makes the burn payable before an adversarial M2 attempt. It does not by itself solve chosen-prompt concentration: the adversary can still precompute likely M2 branches. That residual problem is handled either by the distributional/economic assumptions of Option A, by prompt restrictions, or by a sampling entropy floor such as Option C.

## Zero-knowledge proof feasibility for LLM-based PoML (literature review; 2026-08-30)

### Main conclusion

ZK proving of LLM inference is now feasible for GPT-2-scale models, but the phrase “GPT-2 proof in a few seconds” hides an important distinction. Most systems benchmark one fixed transformer forward computation (often producing one next-token prediction), or a component of a forward pass. They do **not** benchmark a \(k\)-token autoregressive rollout with a fresh VRF sample at every token. The strongest fit for PoML is therefore a DeepProve-style *single proof of the whole claimed sequence*, adapted so that the proved relation also contains PoML's VRF and sampling rule. JOLT Atlas is a useful small-model prototype, while zkGPT/zkComposer are useful comparison points rather than complete PoML solutions.

### What the current systems actually measure

| System | Model/workload in the reported benchmark | Prover / verifier result | Hardware and important qualification |
|---|---|---|---|
| **DeepProve (Lagrange, 2026)** | End-to-end GPT-2 proof for a full sequence of 64, 128, 256, or 512 tokens; the sequence is fed once and causal masking certifies each next-token prediction. | GPT-2: 2.35, 3.02, 4.60, and 7.64 min respectively; verification 1.14--1.33 s; proof 7.95--10.71 MiB; 0.45--1.12 tokens/s. | AMD EPYC 9254, 24 cores, 504 GB RAM. Setup/context-generation is separable and reusable. The benchmark is a full-sequence proof, but the current implementation uses deterministic argmax; PoML's per-token VRF sampler is an additional relation. See the [paper](https://eprint.iacr.org/2026/1112) and [benchmark README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md). |
| **JOLT Atlas (2026)** | Open-source ONNX proof. GPT-2 is reported as a fixed-shape forward graph with trace shape \([1,16,65536]\), not as 16 sequentially sampled responses. A 0.25M-parameter nanoGPT example is also provided. | GPT-2 on the stated machine: setup 1.003 s, proving 14.889 s, verification 1.038 s, end-to-end 16.930 s. nanoGPT: proving 2.288 s, verification 0.127 s, end-to-end 2.678 s. | MacBook Pro M3, 16 GB RAM. The repository does not report autoregressive decoding, VRF sampling, or a generation loop; the 16 positions are parallel logits in one fixed circuit. See the [paper](https://arxiv.org/abs/2602.17452) and [repository](https://github.com/ICME-Lab/jolt-atlas). |
| **zkGPT (USENIX Security 2025)** | GPT-2 transformer proof, characterized by later work as a partial/single-token proof. The paper's table does not clearly specify a multi-token generated response; a later comparison labels the baseline as a 64-token context. | Best reported setting: 21.8 s proving, 0.35 s verification, 101 KB proof (32 threads). | Intel Xeon 6126 at 2.60 GHz, 16 cores, 200 GB RAM, 32 threads, 16-bit quantization. This is a strong transformer-proof baseline, but not evidence of a \(k\)-step sampled rollout. See the [paper](https://www.cse.ust.hk/~kaichen/papers/zkgpt-security25.pdf). |
| **zkComposer (2026)** | Partitions the zkGPT computation over layers and, optionally, input sequence positions. GPT-2 tables use a 64-position fixed context for the baseline and partitioned proofs; sequence lengths 128 and 256 are also evaluated in the sequence-partitioning experiment. | At 64 positions and 192 threads: baseline 147.7 s; 12-way layer partition 30.6 s (4.83x speedup); verification about 0.27--0.37 s; partitioned proof 268.6 KB. | Dual-socket AMD EPYC 9654, 192 cores/threads, 756 GB RAM. Prover time includes witness commitments, GKR proof, and openings; response time additionally models 100 MB/s transmission. It accelerates a forward-proof architecture; it does not itself add VRF sampling or establish that 64 positions are 64 generated tokens. See the [paper](https://arxiv.org/abs/2607.08095). |
| **zkLLM (CCS 2024)** | Component-level CUDA protocols for Llama2 7B/13B, typically with sequence length 2048. The benchmark estimates/times individual attention, lookup, and opening operations rather than running an automated full autoregressive generation. | Paper headline: 13B proof under 15 min. The benchmark repository estimates about 565 s (or 280 s with a different lookup accounting) for a 13B 2048-context operation set. | NVIDIA A6000/A40 in the public benchmark; paper comparisons also use an A100. The code is interactive, GPU-specific, not audited, and the benchmark excludes compilation/loading. Useful historically, but not a clean PoML baseline. See the [paper/repository](https://github.com/jvhs0706/zkllm-ccs2024) and [benchmark notes](https://github.com/jvhs0706/zkllm-benchmark). |
| **ZKML (EuroSys 2024)** | A distilled 81.3M-parameter GPT-2 circuit; one fixed end-to-end inference circuit, not a disclosed long autoregressive response. | KZG: 3,651.67 s proving, 18.70 s verification, 28,128 bytes; IPA: 3,949.60 s, 11.98 s, 16,512 bytes. | AWS r6i.32xlarge, 128 vCPU and 1 TB RAM. The exact generated-token count is not disclosed, so the result should not be converted into a per-token rollout cost. See the [paper](https://ddkang.github.io/papers/2024/zkml-eurosys.pdf). |
| **ZKTorch (2025)** | BERT 1 token, GPT-J 6B 2 tokens, and Llama2-7B 1 token in the end-to-end table. These are short-context forward proofs, not many-token generation. | BERT 880.42 s/26.67 s/4.88 MB; GPT-J 1,397.52 s/62.64 s/6.54 MB; Llama2-7B 2,645.50 s/100.14 s/22.85 MB (prove/verify/proof). | Intel Xeon Platinum 8358, 64 threads, 4 TB RAM; 10-bit quantization (12-bit for GPT-J). GPT-2 comparison is against prior systems rather than a direct GPT-2 experiment. See the [paper](https://arxiv.org/html/2507.07031). |
| **NanoZK (2026)** | Layer/block proofs for transformer dimensions \(d\leq256\), with a full block measured at \(d=128\); GPT-2-scale projection is extrapolated rather than fully measured. | At \(d=128\): attention 92.9 s, full block 94.2 s, verification 1.747 s, 5.7 KB proof. The paper projects roughly 14 min for 12 sequential GPT-2 blocks, or about 2 min with 12 GPU workers. | Main measurements use an Intel Xeon CPU without a GPU; an RTX 3090 prototype is discussed. Real-time conversational inference is explicitly out of scope. This is informative for layerwise parallelism, not a full PoML rollout benchmark. See the [paper](https://arxiv.org/abs/2603.18046). |

The table separates **prover time**, **verification time**, and **proof size**. It also separates setup/key generation where the source reports it. Setup is normally amortizable per model and circuit shape, whereas witness commitment, the actual proof, and the VRF/sampling relation are query-specific. “Inference time” in these papers is not interchangeable: zkLLM estimates cryptographic suboperations, JOLT measures a fixed ONNX graph, and DeepProve measures a complete sequence proof.

There is a reproducibility caveat for DeepProve: the ePrint abstract advertises approximately 174 GPT-2 tokens/minute, while the current public README's 24-core reference table reports 27--67 tokens/minute for sequence lengths 64--512. These are evidently different revisions/configurations or measurement conventions. The paper should quote the exact repository commit, quantization setting, sequence length, and hardware rather than mixing the two headline numbers.

The README's command describes \`--sequence 64\` as inference on a 64-token random prompt and records \`max_context\`; it does not separately expose prompt length and generated-response length. Thus, even for DeepProve, a PoML evaluation must report \(p\) and \(k\) separately and must not silently relabel a context-length benchmark as “64 sampled output tokens.”

### Why DeepProve is the closest architectural fit

Naively proving \(k\) autoregressive tokens means proving \(k\) growing-context forward passes, with a nominal \(O(k^2|C|)\) cost and awkward in-circuit KV-cache handling. DeepProve instead concatenates the prompt and claimed response, runs the transformer once, and uses the causal mask to certify the next-token condition at every response position. Its published sweep over 64--512 tokens is therefore the first directly relevant evidence that a full response can be proved as one object. The design also supports chunking across workers, which maps naturally to PoML's distributed miners.

For PoML, the proof statement must be extended beyond DeepProve's deterministic argmax relation. It should bind: (i) the committed quantized model and tokenizer/configuration, (ii) the accepted \(M_1\) prompt-prefill state or its commitment, (iii) the full \(M_2\) output sequence, (iv) each VRF evaluation or a recursively committed VRF seed stream, and (v) the exact fixed-point implementation of temperature/top-\(p\)/top-\(k\)/EOS sampling. The proof must show that the sampled token, not merely the maximum-logit token, follows from the proved logits and the VRF output. DeepProve's mention of deterministic derandomization by a single random seed is encouraging, but it is not automatically the same as PoML's per-token VRF requirement.

### Recommended PoML implementation path

1. **Primary path: DeepProve-style full-sequence proof.** Use a very small GPT-2-like or nanoGPT checkpoint first, keep prompt prefill in \(M_1\), and prove the accepted prompt plus the claimed response in one \(M_2\) circuit. Add a compact VRF/sampler gadget and measure its incremental cost. This avoids claiming that a one-token proof scales linearly when it may actually require redoing the growing context at every step.
2. **Prototype path: JOLT Atlas.** Its open-source Rust implementation and 2.288 s nanoGPT result on a 16 GB laptop make it attractive for an initial PoML demonstration. However, the first experiment must replace the fixed 16-position forward benchmark with (k\in\{1,4,8,16,32,64\}) autoregressive/VRF samples; otherwise it is only a proof of a batched logits graph.
3. **Comparison/acceleration path: zkGPT plus zkComposer.** These provide useful transformer-specific baselines and demonstrate that layer/sequence partitioning can trade memory and wall-clock time. They should not be described as already solving PoML's full sampled generation problem.
4. **Do not use zkLLM, ZKML, or NanoZK as the main feasibility claim.** They are valuable historical or component-level references, but their workloads are interactive/component estimates, a single fixed inference, or layer projections rather than a complete VRF-driven response.

### Minimum benchmark that PoML should report

For one fixed quantized checkpoint, report prompt length \(p\) and response lengths \(k\in\{1,4,8,16,32,64\}\). For each \(k\), measure native inference, prover time, peak memory, verifier time, proof bytes, setup/key-generation time (separately), VRF evaluation and sampler time, and communication. Run both greedy decoding (sanity check) and the actual VRF-driven sampler. State whether the circuit proves only \(M_2\) given an \(M_1\) commitment or reproves the prompt. Report quantization bits and perplexity/accuracy on a held-out corpus, because nearly all practical systems use fixed-point/quantized arithmetic. Finally, publish the exact hardware; a MacBook M3 result cannot be compared directly with a 24-core/504 GB server, a 192-core/756 GB server, or an A100.

The conservative paper claim is therefore: **LLM proving is feasible for a small PoML model, but current evidence supports a full-sequence proof architecture, not a claim that arbitrary autoregressive decoding can be handled by multiplying a single-token benchmark by the number of tokens.**

## DeepProve, the \(M_1/M_2\) split, and comparison with the diffusion experiment

### DeepProve's mechanism in PoML notation

Let the tokenized prompt be \(x=(x_1,\ldots,x_p)\), and let the claimed response be \(y=(y_1,\ldots,y_k)\). An autoregressive model would ordinarily compute

\[
\ell_t = F_\theta(x\Vert y_{<t}),\qquad
y_t = \mathsf{Sample}(\ell_t,r_t).
\]

The naive proof repeats a growing-context forward pass for every \(t\). DeepProve avoids this by forming one teacher-forced sequence \(z=x\Vert y\) and evaluating the transformer once on all positions, with the causal mask. The logits at position \(p+t-1\) see \(x\Vert y_{<t}\), but cannot see \(y_t\) or later tokens. Therefore the single forward pass contains exactly the logits that the autoregressive decoder would have produced at every step. The proof checks those logits and the output rule at all response positions. This is a proof of the *whole sequence*, not \(k\) independent proofs, and it avoids the quadratic repeated-pass/KV-cache problem.

The current DeepProve relation checks a deterministic next-token argmax. For PoML, replace that check by
\[
y_t=\mathsf{Sample}(\ell_t,\rho_t),\qquad
(\rho_t,\pi^{\mathsf{VRF}}_t)=\mathsf{VRF.Eval}(\mathsf{sk}_m,\mathsf{seed}_t),
\]

and prove the VRF binding and fixed-point sampler inside the relation (or in a separately verified compositional proof). No current DeepProve benchmark measures this VRF/sampling extension, so its incremental cost is an open measurement rather than a zero-cost assumption.

### Is DeepProve directly compatible with \(M_1/M_2\)?

Conceptually yes, but not out of the box. DeepProve's published model graph proves embeddings, prompt processing, attention, normalization, logits, and argmax end-to-end. In the proposed PoML split, \(M_1\) would provide a canonical prompt-side state, normally the prompt KV tensors and the logits needed for the first sampled token. \(M_2\) would then take that committed state, derive the VRF randomness, and prove the generated suffix.

The proof interface must bind the \(M_1\) state to the exact tokenized prompt, model hash, tokenizer/template, positional offset, precision, and quantization parameters. A hash of the KV state is not by itself a proof that the state is correct. The protocol therefore needs one of:

- an \(M_1\) correctness proof;
- canonical delegated \(M_1\) computation with an acceptance/replication rule; or
- recomputation of the prompt boundary inside the \(M_2\) proof.

The last option is simplest cryptographically but means that prompt prefill is still performed by the prover, even if it is excluded from the consensus accounting. The first option is cleaner for a genuine two-stage protocol, but adds another proof or verification mechanism. Thus DeepProve supplies a promising *full-sequence proof engine*, not a ready-made \(M_1/M_2\) protocol.

### Is it efficient relative to the current diffusion example?

The comparison must be qualified. The current PoML implementation uses an 11,401-parameter U-Net, one DDPM denoising step, EZKL, and a 16-core Intel Core i9/128 GB host; its measured inference--proof pair is approximately 117 s on average. DeepProve's public GPT-2 reference is approximately 141 s for a 64-position proof, rising to 458 s for 512 positions, on a much larger 24-core AMD EPYC 9254/504 GB host. These are not like-for-like: the models, proof systems, hardware, quantization, and sequence semantics differ by orders of magnitude.

The fairest conclusion is that **a small GPT-2 proof is in the same broad minutes-per-query regime as the current toy diffusion demonstration, but it is not yet evidence that LLM PoML is faster or equally scalable**. DeepProve's distributed mode and its reported paper-level throughput may reduce wall-clock proving time, while the current diffusion number is a single-host EZKL measurement. A meaningful comparison requires identical hardware, a fixed response length, the same quantized model size, and inclusion of VRF/sampler and \(M_1\)-boundary costs in both systems.

### Why zkGPT reports only 21.8 seconds

The 21.8 s zkGPT number is a real GPT-2 transformer-proving result, but it should not be interpreted as “21.8 s for a full \(k\)-token sampled response.” The zkGPT paper's comparison table reports a GPT-2 proof under 32 threads, with 0.35 s verification and a 101 KB proof, but does not clearly specify a multi-token generated-response length. DeepProve explicitly characterizes zkGPT as a partial implementation for a single token. A later zkComposer comparison uses a 64-position fixed context for its zkGPT baseline, but a context length is not automatically a 64-token autoregressive output.

The likely workload is therefore: prove one transformer forward computation over a fixed input embedding/context and certify one next-token output (with the sequence/context length treated as a circuit dimension). It is much cheaper than naively proving every generated token because it does not include a complete VRF-driven decoding loop. To use zkGPT in PoML, one would need either \(k\) separate proofs (with growing contexts and substantial repeated cost) or a DeepProve-style batched causal proof. The 21.8 s figure alone cannot distinguish those choices.

## Clarification: GPT-2 sampling and the VRF requirement

GPT-2 itself is a causal language model that outputs logits; argmax is a decoding policy, not an architectural requirement. The original OpenAI GPT-2 sampler divides logits by a temperature, optionally applies top-\(k\) and top-\(p\) filtering, and then calls a multinomial sampler. Setting top-\(k=1\) or using greedy decoding gives argmax; otherwise the same GPT-2 checkpoint can sample from its induced softmax distribution without retraining. The current Hugging Face generation API makes the same distinction: do_sample=False is greedy decoding, while do_sample=True enables temperature/top-\(k\)/top-\(p\) sampling. See the [original GPT-2 sampler](https://github.com/openai/gpt-2/blob/master/src/sample.py), [GPT-2 model documentation](https://huggingface.co/docs/transformers/main/model_doc/gpt2), and [generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation).

This means the PoML requirement is not incompatible with GPT-2. We should describe the deployed model as

\[
\text{GPT-2 logits}+\text{a fixed, protocol-defined randomized decoding policy}.
\]

The policy must be deterministic conditional on the prescribed VRF randomness. For example, let one VRF output be a seed \(\rho\), derive \(u_t=\mathsf{PRG}(\rho,t)\), and use \(u_t\) to perform inverse-CDF sampling from the temperature/top-\(p\)-filtered distribution at step \(t\). The proof checks

\[
F_\theta(x\Vert y_{<t})=\ell_t,\qquad
C_t(y_t-1)\leq u_t<C_t(y_t),
\]

where \(C_t\) is the fixed-point cumulative distribution derived from \(\ell_t\). The VRF proof binds \(\rho\) to the block/query/miner input. This uses one VRF proof per query and a deterministic seed expansion; if the protocol insists on an independent VRF evaluation for every token, the same final-token checks can use \(\rho_t\), but the cost of \(k\) VRF proofs must then be measured separately.

DeepProve's August 2026 engineering description explicitly says that sampling-based decoding can be handled by derandomizing with a single random seed and proving it with the same machinery. This is strong evidence that the required modification is structurally straightforward. It is not evidence that the public benchmark already measures PoML's exact VRF sampler: the public README currently advertises logits/argmax support and does not expose a sampling benchmark. We should therefore claim **compatibility in principle, supported by the DeepProve construction, pending an implementation and cost measurement for the VRF/sampler gadget**. See [Inside DeepProve](https://lagrange.dev/engineering-updates/inside-deepprove-proving-an-llm-end-to-end).

For the proposed protocol, using the full concatenated \(x\Vert y\) as the \(M_2\) proof input is the cleanest first implementation. \(M_1\) can independently prove the prompt-only computation for acceptance or data availability, but if \(M_2\) re-runs \(x\Vert y\), the \(M_1\) proof is not being used to reduce the \(M_2\) prover's transformer work. That is acceptable if \(M_1\) is deliberately outside the consensus cost; it should simply be stated as an accounting/acceptance split, not as computational composition. A true cache-reusing split would require a proof interface exposing and binding the prompt KV boundary.

## Is \(M_1\) still necessary when \(M_2\) proves the full \(x\Vert y\) trace?

### Short answer

If the chosen \(M_2\) relation proves the complete tokenized input and response, then \(M_1\) is no longer necessary as the mechanism that prevents honest miners from reusing prompt-side computation. The cleanest construction is either:

1. remove \(M_1\) from the core consensus computation and include all reusable work in a distributional CIA parameter \(\varepsilon\); or
2. retain \(M_1\) only as an optional acceptance, availability, or latency phase, while making clear that it does not reduce the \(M_2\) prover's work.

Keeping \(M_1\) as a *security boundary* while \(M_2\) independently proves \(x\Vert y\) gives little benefit and risks an ambiguous accounting argument.

### A full-sequence proof does not imply fresh physical computation

The statement proved by a ZK system is generally of the form

\[
\exists w:\; R(x\Vert y,w)=1,
\]

not “the prover executed every operation in \(R\) from an empty cache.” A fresh commitment and Fiat--Shamir challenge prevent an old proof transcript from being copied, but they do not prevent a prover from reusing deterministic witness material or arithmetic subcomputations that it computed for an earlier query. Thus, even when the proof contains the full \(x\Vert y\) trace:

- an identical prompt prefix may allow reuse of token embeddings, prompt-side key/value tensors, or other deterministic intermediate values;
- a similar, but not identical, prompt may allow implementation-dependent partial reuse, although this is harder to quantify and should not be assumed away;
- the generated suffix and the final proof transcript normally remain query- and randomness-dependent, so a cached prompt does not make the whole proof free.

DeepProve's one-pass causal construction removes the *quadratic repeated-generation* problem and avoids requiring a separate in-circuit KV-cache update for each token, but this is an algorithmic implementation fact, not a cryptographic guarantee against cross-query caching. Its published description should therefore not be read as proving that all work is necessarily recomputed for every query. See [Inside DeepProve](https://lagrange.dev/engineering-updates/inside-deepprove-proving-an-llm-end-to-end) and the [DeepProve benchmark README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md).

A useful accounting decomposition is

\[
W_{\mathrm{fresh}} =
W_{\mathrm{prompt}}+W_{\mathrm{suffix}}+W_{\mathrm{proof}}+W_{\mathrm{VRF}},
\]

where \(W_{\mathrm{prompt}}\) is the deterministic prompt-side work. If a cached prompt saves \(W_{\mathrm{prompt}}\) with probability \(r\), then the expected saved fraction is roughly

\[
r\,\frac{W_{\mathrm{prompt}}}{W_{\mathrm{fresh}}},
\]

not zero. The fraction is workload-dependent: it can be small for unique long prompts and large when many queries share a long system or few-shot prefix.

### When folding \(M_1\) into distributional CIA is valid

We can fold the advantage of prompt prefill, KV reuse, and any other honest-query cache into a *distributional* CIA assumption, but the definition must measure saved work rather than only exact output-prefix collisions. Let \(D\) be the stated honest-query distribution and let \(S_{\mathcal A}(x)\) be the maximum consensus work that an admissible miner \(\mathcal A\) can save using state obtained before the query is accepted. A suitable assumption is, for example,

\[
\Pr_{x\leftarrow D}\!\left[
\frac{S_{\mathcal A}(x)}{W_{\mathrm{fresh}}(x)}>\varepsilon
\right]\leq\delta,
\]

or, if an expected-work theorem is preferable,

\[
\mathbb E_{x\leftarrow D}\!\left[
\frac{S_{\mathcal A}(x)}{W_{\mathrm{fresh}}(x)}
\right]\leq\varepsilon.
\]

The bound should include prompt-prefix/KV reuse, reusable generated-prefix work, and any proof-generation reuse that remains possible. Exact trace collisions are one sufficient empirical proxy, but they are not equivalent to a work-savings bound. In particular, a collision experiment can report probability zero while a common system prefix still gives a substantial deterministic prefill saving.

If \(M_1\) is computed once, canonically committed, and supplied to every \(M_2\) miner, then its cost is outside the \(M_2\) consensus work by definition. In that case it should not also be counted as an adversarial cache advantage. Conversely, if a miner may privately compute \(M_1\) before acceptance and use it to accelerate its own \(M_2\) proof, that saved work must be included in \(\varepsilon\), even if the public protocol labels the computation “\(M_1\).”

The current paper's CIA definition is worst-case over adversarially chosen inputs. Replacing it with the condition above is therefore a theorem-level change: the security claim becomes workload/distribution-dependent (and normally probabilistic, with an explicit \(\delta\)), rather than a worst-case guarantee for every prompt. The reduction must use the resulting total \(\varepsilon_{\mathrm{total}}\), including proof and cache savings, in the same place where it currently uses \(\varepsilon_i\).

### Three design choices

**A. Remove \(M_1\) from the core construction.** Prove the full \(x\Vert y\) sequence in \(M_2\), and define one distributional \(\varepsilon_{\mathrm{total}}\) for all reusable work. This is the simplest statement and avoids claiming a compositional boundary that the proof system does not expose.

**B. Keep \(M_1\) as an optional protocol layer.** Use it for query acceptance, prompt-state availability, or service latency, but let \(M_2\) prove the full \(x\Vert y\) relation. State explicitly that \(M_1\) is outside consensus accounting and is not, by itself, a defense against private precomputation.

**C. Make \(M_1\) a genuine compositional boundary.** Have \(M_1\) produce a verifiably correct commitment to the prompt KV state, and make the \(M_2\) circuit start from that committed state. This can reduce \(M_2\) work, but it requires a new proof interface and a correctness proof for the boundary; a bare hash of KV tensors is insufficient.

### Recommendation for the paper

Use option **A** for the main security theorem, or option **B** if the system needs the two-stage fee/acceptance workflow. In either case, say directly:

> Full \(x\Vert y\) proving prevents transcript reuse but does not cryptographically force recomputation of deterministic prompt-side subcomputations. Any residual advantage from prompt/KV caching is included in the distributional \(\varepsilon_{\mathrm{total}}\) bound.

Do not present \(M_1\) as simultaneously (i) excluded from consensus, (ii) independently recomputed inside \(M_2\), and (iii) a guaranteed defense against reuse. Those three descriptions refer to different accounting models. Also note that if the paper wants to retain the existing worst-case CIA theorem, distributional CIA alone is insufficient; the theorem must either be weakened explicitly or the protocol must add a mechanism that invalidates prior prompt-side state.

## Clarification: what a full \(x\Vert y\) DeepProve proof guarantees

The understanding is essentially correct, with one qualification about the word "trace."

### What the proof certifies

Assuming the DeepProve relation is extended with the PoML sampling rule, the proof can certify that:

1. \(y\) is the response produced by the fixed model on \(x\), under the specified tokenizer, model parameters, precision, and causal-mask semantics; and
2. at every response position \(t\), the claimed token \(y_t\) is the result of applying the fixed sampling algorithm to the model logits and the VRF-derived randomness \(\rho_t\) (or to a seed \(\rho\) from which all \(\rho_t\)'s are deterministically derived).

The proof therefore establishes *functional correctness* of the complete response, not merely that the output is plausible. The VRF statement must itself be bound to the query, miner, and protocol-defined seed, and the sampler must be specified in a circuit-friendly fixed-point form.

### What the proof does not certify

The proof does not normally certify the resource-history statement

\[
\text{"the prover recomputed every intermediate value from scratch after query acceptance."}
\]

A ZK proof establishes that there exists a witness satisfying the circuit and, with fresh commitments/challenges, that the current proof is a fresh proof of that witness. It does not generally establish that the witness was generated without cached data. Thus:

- cached prompt embeddings or prompt-side KV tensors may still be reused;
- a cached deterministic prefix of the witness may still be inserted into a new proof;
- fresh masking, hashing, commitment, and proof-transcript work may still be required, even if the expensive model arithmetic for the cached portion is not repeated.

Consequently, proving the full \(x\Vert y\) sequence rules out copying an old full proof, but it does not automatically rule out a speedup from cached deterministic subcomputations.

### Full precomputation and a VRF match

Suppose a miner precomputes a candidate response using arbitrary randomness before it is eligible to mine. If the eventual VRF-derived randomness produces exactly the same token sequence, the miner may reuse the precomputed response-side witness and obtain a speedup. The proof can still be valid: it proves that the sequence is correct for the actual VRF values, not that the miner waited for those values before doing any work.

However, this attack succeeds only when the precomputed sequence (or enough of its witness) matches the VRF-conditioned sequence. If the VRF result differs, the precomputation is not directly usable for the response-dependent portion and may be wasted. This is the probabilistic event that belongs in the chosen-prompt/fee analysis. Prompt-side deterministic work is different: it can often be reused regardless of the eventual sampled suffix and should be accounted for in the cache-saving component of \(\varepsilon_{\mathrm{total}}\).

Therefore, the accurate statement is:

> A full \(x\Vert y\) proof certifies the model computation and VRF-conditioned sampling, and prevents reuse of an old proof transcript. It does not, by itself, enforce fresh recomputation of every intermediate computation. Any remaining speedup from cached deterministic state, or from a lucky precomputed VRF-matching response, must be bounded separately.

## Is \(M_1\) the right defense against asymmetric prompt caching?

### Verdict

Yes, a canonical \(M_1\) stage is the cleanest way to remove the *prompt-prefill* advantage between miners, provided that \(M_1\) includes the complete reusable prompt state and is available to every \(M_2\) miner before \(M_2\) eligibility begins. In a transformer, this means more than token embeddings: it should include whatever layer-by-layer prompt activations or KV tensors the \(M_2\) circuit actually consumes.

Under those conditions, a miner that happened to see the prompt earlier does not have a special advantage over a miner that did not. Both receive the same accepted \(M_1\) result. The prompt-side computation is then common protocol work and can be excluded from the differential \(M_2\)-CIA accounting.

There are two necessary qualifications:

1. \(M_2\) must actually use the canonical \(M_1\) state. If the \(M_2\) circuit still starts from raw \(x\) and recomputes the prompt internally, \(M_1\) does not reduce the \(M_2\) prover's work; it is only an administrative or fee-acceptance phase.
2. The \(M_1\) state must be verifiably bound to \(x\), the model hash, tokenizer/template, precision, positional offset, and quantization. A bare commitment or hash does not establish that the state was computed correctly. This requires an \(M_1\) proof, a trusted/canonical provider, or recomputation of the boundary inside a larger proof.

### What the prefix experiment can establish

The prefix-match experiment is useful for the residual \(M_2\) problem, but it should not be presented as measuring all cache reuse. It primarily estimates the probability that two independently sampled executions share the same generated prefix:

\[
\Pr[y_{1:\ell}=y'_{1:\ell}\mid x].
\]

That is relevant to reusing response-dependent work. It does not measure:

- how often future queries repeat an exact tokenized prompt prefix;
- how much prover work is saved when such a prefix is cached;
- common system or few-shot prefixes that appear in nearly every query;
- model-wide implementation optimizations or reusable lookup/constant data.

The experiment should therefore report two separate quantities:

\[
r_{\mathrm{prompt}}=\Pr[\text{a previously cached exact token prefix occurs}],
\]

and

\[
\alpha_{\mathrm{suffix}}(\ell)
 =\frac{\text{work saved by reusing a generated prefix of length }\ell}
        {\text{fresh }M_2\text{ work}}.
\]

The residual expected advantage is then related to \(r_{\mathrm{prompt}}\alpha_{\mathrm{prompt}}\) and to the generated-prefix collision terms \(\Pr[y_{1:\ell}=y'_{1:\ell}]\alpha_{\mathrm{suffix}}(\ell)\). Exact token-prefix equality matters here; semantic similarity normally does not produce identical KV states.

Thus the strongest defensible claim is:

> Canonical \(M_1\) removes asymmetric reuse of the deterministic prompt-prefill state. Remaining private reuse inside \(M_2\), including generated-prefix reuse and lucky VRF-matching precomputation, is not caught by full-sequence ZK soundness and must be bounded by a residual distributional \(\varepsilon_{M_2}\) assumption.

This is stronger and more precise than claiming that a low output-prefix collision probability proves that all reuse is negligible. It also keeps the chosen-prompt/fee argument separate from the honest-query cache argument.

## Verifying \(M_1\) as an on-chain transaction

### First correction about DeepProve

DeepProve does not merely verify the final output while ignoring all internal model operations. Its end-to-end relation constrains the transformer operators, including embeddings, attention, normalization, nonlinearities, quantization, and logits. The important limitation is different: the current construction does not expose or commit to every intermediate activation as a reusable public object. It proves the activations as private witness values and uses a one-pass full-sequence formulation in which the ordinary autoregressive KV-cache data structure disappears from the proof. See [Inside DeepProve](https://lagrange.dev/engineering-updates/inside-deepprove-proving-an-llm-end-to-end), especially its discussion of proving every operator and not committing to intermediate activations.

Therefore, a DeepProve proof can establish prompt-prefill correctness, but the current interface is not automatically an \(M_1\) interface. To let \(M_2\) start from a reusable prompt state, the relation must expose a commitment to the selected prompt boundary (for example, the layer-by-layer KV tensors) and bind the suffix proof to that commitment.

### Three verification choices

#### A. Deterministic recomputation plus a state commitment

For a small GPT-2-like model, the simplest construction is:

1. A prefill worker computes \(s=M_{1,\theta}(x)\) using a canonical fixed-point/integer implementation.
2. The worker publishes a commitment
   \[
   c_s=H_{\mathsf{prefill}}\!\left(
   \mathsf{qid}\parallel h_\theta\parallel
   \mathsf{version}\parallel\mathsf{canon}(s)
   \right)
   \]
   and a data-availability reference for the actual state \(s\).
3. Validators recompute \(M_1(x)\), serialize the result canonically, and check that its digest equals \(c_s\).
4. After acceptance, all \(M_2\) miners receive the same \(s\) and the \(M_2\) proof takes \(c_s\) (or an opening of it) as its prompt-boundary input.

This requires no \(M_1\) ZKP. It is a standard deterministic replicated-computation validity check. It is appropriate if the prompt length and \(M_1\) cost are small enough that every validating node can afford the recomputation.

It has two protocol consequences. First, \(M_1\) is no longer outside consensus: every validator performs it when validating the transaction. Second, posting only \(c_s\) is insufficient for \(M_2\), because miners need the state itself. The transaction must either carry the KV state (potentially very large) or carry a Merkle/data-availability root from which every miner can retrieve it.

#### B. Optimistic verification with a fraud-proof dispute

This is the most natural no-ZKP option when validator recomputation is too expensive. A worker posts a bonded claim and a commitment to \(s\). Verifiers recompute \(M_1\) off-chain during a challenge window. If a verifier disagrees, the parties use bisection/pinpointing to identify one incorrect graph operation or VM transition; the chain executes only that final step and slashes the party that is wrong.

This is exactly the model studied for ML inference by [Agatha](https://arxiv.org/abs/2105.04919) and [opML](https://arxiv.org/abs/2401.17555). Agatha uses a graph-based pinpoint protocol and cross-evaluator-consistent execution to make native DNN execution agree with on-chain arbitration. opML uses deterministic fixed-point execution, Merkle-committed VM state, a challenge period, and an interactive dispute game. These systems show that the chain need not execute the complete DNN in the normal case; it only arbitrates a disputed elementary operation.

For PoML, this maps cleanly to the proposed two-stage workflow:

\[
\mathsf{REGISTER\_QUERY}
\;\longrightarrow\;
\mathsf{PREFILL\_CLAIM}
\;\xrightarrow[\text{challenge window}]{\text{accepted}}
\mathsf{M}_2\text{ mining}.
\]

The tradeoff is that correctness is now optimistic/economic rather than an immediate validity-proof guarantee. The design needs at least one honest and active verifier (an AnyTrust-style assumption), sufficient data availability, anti-censorship for challenges, and rewards large enough to overcome the verifier's dilemma.

#### C. A dedicated \(M_1\) boundary ZKP

For a cryptographic guarantee without a challenge delay, define an \(M_1\) relation that proves

\[
s=M_{1,\theta}(x)
\quad\text{and}\quad
c_s=\mathsf{Com}(s).
\]

The \(M_2\) relation then takes \(c_s\) as a public input and proves that its suffix computation starts from the committed state. Intermediate activations do not need to be revealed. They only need to be constrained inside the \(M_1\) proof and consistently linked to the \(M_2\) proof.

[zkComposer](https://arxiv.org/abs/2607.08095) gives a relevant construction pattern: adjacent zkML sub-proofs share a fresh masked commitment to boundary activations, so the two proofs use exactly the same boundary values while preserving zero knowledge. A prompt-prefill/suffix split would require adapting this idea to the sequence boundary and the KV tensors; it is conceptually plausible but not provided by current DeepProve out of the box. A ZK-friendly commitment such as the paper's \(H_{\mathsf{zk}}\), or a polynomial commitment, is preferable to placing a general-purpose hash gadget inside the circuit.

### What the hash idea does and does not solve

Posting a hash of the \(M_1\) output is sound only together with a correctness mechanism:

- **Hash plus every-validator recomputation:** correctness follows from deterministic consensus execution.
- **Hash plus optimistic challenge/fraud proof:** correctness follows under the AnyTrust/economic assumptions.
- **Hash plus an \(M_1\) ZKP:** correctness follows from the proof system.
- **Hash alone:** only binds a state; it does not show that the state equals \(M_1(x)\).

The commitment must use a canonical serialization and domain-separate the query identifier, model hash, implementation version, prompt length, tensor shapes, precision, and positional offset. Fixed-point arithmetic and a fixed reduction order are important because otherwise two honest validators can compute different hashes from nominally identical floating-point executions. Agatha's XCE design and opML's fixed-point execution address precisely this cross-platform consistency issue.

The actual state also needs availability. A Merkle root is a compact on-chain commitment, but it is not a substitute for publishing the leaves. The protocol should require the KV state to be retrievable before the \(M_1\) claim becomes accepted, and should charge for bandwidth/storage or impose a prompt-length and state-size limit. Otherwise a malicious worker can post a valid-looking root and withhold the state, blocking all \(M_2\) miners.

### Recommendation

For the paper's small-model prototype, use **deterministic recomputation plus a canonical state commitment**, and make the state available to all miners. This is the easiest construction to specify and implement. If recomputing \(M_1\) at every validator is too costly, replace that step with an **opML/Agatha-style optimistic fraud-proof layer**. Reserve a dedicated \(M_1\) ZKP for the stronger cryptographic version; it requires a real boundary-commitment interface and should not be described as something current DeepProve already supplies.

The proposed \(M_1\) transaction should therefore contain, at minimum,

\[
(\mathsf{qid},h_\theta,\mathsf{version},
c_s,\mathsf{DAroot},\mathsf{size},\mathsf{bond},\sigma),
\]

where \(c_s\) commits to the canonical prompt state and \(\mathsf{DAroot}\) makes that state retrievable. The \(M_2\) proof must bind its initial prompt state to \(c_s\). If \(M_2\) instead starts from raw \(x\), the \(M_1\) claim may still be useful for acceptance or availability, but it does not actually reduce the \(M_2\) prover's work.

## Variable-length queries, token-based fees, and a fair PoML lottery

### Why a constant lottery probability is unfair

If every completed query receives the same lottery probability \(p\), then a short response and a long response produce the same chance of winning even though their inference/proof costs differ. A miner could therefore prefer the cheapest eligible queries and obtain more lottery attempts per unit of computation.

The current API convention is a useful user-facing analogy: providers distinguish input/prompt tokens, output tokens, and sometimes cached input tokens, and expose a maximum output-token setting. [OpenAI's token accounting guidance](https://help.openai.com/en/articles/4936856) explicitly separates these categories, while [Anthropic's current price tables](https://docs.anthropic.com/en/docs/about-claude/pricing) also distinguish input, output, and cache-hit pricing. However, token count should be treated as a billing proxy, not automatically as the consensus work measure: transformer and proof costs depend on prompt length, response length, attention structure, quantization, and the particular proving system.

### Define a public work schedule

For a query with prompt length \(p\) and actual response length \(k\), define a deterministic, publicly computable work value

\[
C(p,k)
 =
C_{\mathsf{M}_1}(p)
+\;C_{\mathsf{M}_2}(p,k)
+\;C_{\mathsf{proof}}(p,k)
+\;C_{\mathsf{VRF}}(k).
\]

The model registry should publish the circuit, precision, and a cost schedule \(C\) (ideally the exact constraint count or a conservative calibrated upper/lower bound). A simple token-linear schedule is acceptable only after benchmarking shows that it approximates the real prover cost over the allowed range. If \(M_1\) is canonically computed and supplied to all \(M_2\) miners, \(C_{\mathsf{M}_1}\) is a service/acceptance cost rather than \(M_2\) mining work and should not be included in the \(M_2\) lottery weight. If \(M_2\) recomputes the prompt, it must be included.

The relevant quantity for fairness is not the miner's wall-clock time, which is hardware-dependent, but a protocol work unit derived from \(C(p,k)\). This is analogous to the constraint-count metric in [Zk-SNARK Marketplace with Proof of Useful Work](https://arxiv.org/abs/2510.09729), which makes proof fees complexity-proportional and sets the lottery probability from the associated circuit complexity. That paper also emphasizes that proving time must be predictable from circuit complexity and that adding a cumulative-chain-complexity bonus can increase centralization; PoML should use the cost of the current query, not a bonus for accumulated work already performed in the current block.

### Ethereum-style fee cap and stopping rule

The user can specify a maximum output length \(K\) or, equivalently, a maximum work/fee cap. Let \(C_{\max}=C(p,K)\). The registration transaction escrows

\[
F_{\max}
 =
B_0
+\;(P_{\mathsf{base}}+P_{\mathsf{tip}})C_{\max}
+\;F_{\mathsf{storage}}(K),
\]

where \(B_0\) is a nonrefundable base burn and \(P_{\mathsf{base}},P_{\mathsf{tip}}\) are protocol and priority prices per work unit. At completion with actual length \(k\), the protocol charges

\[
F_{\mathsf{actual}}
 =
B_0
+\;(P_{\mathsf{base}}+P_{\mathsf{tip}})C(p,k)
+\;F_{\mathsf{storage}}(k)
\]

and refunds \(F_{\max}-F_{\mathsf{actual}}\). This follows the Ethereum gas-limit idea: a transaction specifies a limit, and unused gas is returned. See the [Ethereum gas documentation](https://ethereum.org/developers/docs/gas/).

The protocol should not allow a miner to stop at an arbitrary point merely because it chooses to spend no more fee. The valid termination rule should be deterministic:

- generation stops early only when the model samples the protocol-defined EOS token;
- if \(K\) is reached, the response is marked with a protocol-defined \(\mathsf{TRUNCATED}\) flag;
- a proof must certify either EOS or exactly \(K\) generated tokens followed by truncation.

Thus a miner cannot claim a cheap, short response by omitting a difficult suffix. The base burn \(B_0\) is useful for the chosen-prompt analysis: a very short query still incurs a nonzero cost for creating and registering an attack query. The variable component can then be proportional to the actual proven work.

There is a related optional-stopping issue in the lottery. If the threshold is computed from the realized length \(k\), then \(k\) must be fixed by the registered query, the model's VRF transcript, and the EOS/truncation rule before the lottery hash is evaluated. A miner must not be allowed to try several prefixes and retain whichever one gives the most favourable threshold. The conservative alternative is to use the registered maximum \(K\) for the lottery weight while charging/refunding according to the realized \(k\); this is simpler to reduce, but it makes early-EOS queries less capital-efficient.

### Complexity-weighted lottery

Let \(C_0\) be a reference work unit and let

\[
w_i=\frac{C_i}{C_0}
\]

be the normalized cost of query \(i\), represented with a fixed-point or integer scale. Let \(p_0\) be the success probability of one reference work unit. A robust weighted lottery is

\[
p_i=1-(1-p_0)^{w_i}.
\]

The verifier accepts the query's lottery hash when

\[
H(\mathsf{bind}_i\parallel \mathsf{qid}_i\parallel
\mathsf{statement}_i)
<
\left\lfloor 2^\kappa p_i\right\rfloor.
\]

When \(p_0w_i\) is small, \(p_i\approx p_0w_i\), giving the simpler rule used in SNARKChain-style designs: lottery odds are proportional to current-proof complexity. The exponential form avoids probabilities above one and has an exact compositional interpretation. For independent queries,

\[
\prod_i(1-p_i)
=
(1-p_0)^{\sum_i w_i},
\]

so a miner's total chance depends on total normalized work, not on whether that work was packaged as one long query or many short queries. The query identifier, block binding, response length, and proof statement must all be fixed before the lottery comparison, so a miner cannot choose a cheaper target after seeing a favorable hash.

For the cleanest reduction, choose an integer work scale and regard query \(i\) as \(w_i\) virtual reference-work trials. The protocol can derive the trials as \(H(\mathsf{bind}_i\|\mathsf{statement}_i\|j)\) for \(j=1,\ldots,w_i\), and accept the first successful \(j\). This changes the old one-ticket-per-inference lemma to a work-bounded statement (at most \(\sum_iw_i\) virtual trials), but it restores the exact random-oracle structure used by the backbone proof. A single hash with threshold \(p_i\) is cheaper to specify, but then the reduction must handle independent Bernoulli trials with non-identical probabilities rather than literally reuse the current lemma.

### How the backbone reduction changes

The existing \(q_{\mathsf{ML}}\) theorem counts completed inference-proof pairs. With variable lengths, the natural bound is instead a per-round work budget. Let \(B_{\mathcal A}\) be the maximum normalized work a miner can complete in one round, and let

\[
C_i^{\mathsf{eff}}\geq(1-\varepsilon_{\mathsf{total}})C_i
\]

be the effective cost after all allowed caching and precomputation advantages. Then

\[
\sum_{i\in\mathcal I_{\mathcal A}}w_i
\;\leq\;
\frac{B_{\mathcal A}}{1-\varepsilon_{\mathsf{total}}}.
\]

The weighted lottery is equivalent to that many virtual reference-work Bernoulli trials (exactly for integer \(w_i\), and up to rounding otherwise). Therefore the backbone argument can be rebuilt using a work-unit oracle and its total per-round lottery mass. The structural assumptions remain the same: independent block-bound lottery events, public verifiability, no free extra tickets, and a bound on the adversary's total completed work.

This is not literally the current theorem with \(q_{\mathsf{ML}}\) renamed. The theorem and its condition must be rewritten in terms of \(B_{\mathcal A}\) or a normalized work bound. An analogue of the current condition is needed to ensure that \(\varepsilon_{\mathsf{total}}\) cannot buy one additional reference-work unit in a round; the exact inequality depends on the chosen work-unit granularity and rounding. If retaining the current theorem verbatim is more important than supporting arbitrary lengths, use fixed-cost buckets or require a fixed \(K\) per mining attempt.

### Recommended PoML fee and lottery design

Use the following separation:

1. **User interface:** input-token/prompt fee, output-token fee, maximum output \(K\), and a maximum fee cap.
2. **Consensus accounting:** a public work schedule \(C(p,k)\) based on measured model-plus-proof complexity, not merely token count.
3. **Fee settlement:** charge the actual proven \(C(p,k)\), refund unused escrow, and retain a fixed base burn \(B_0\).
4. **Lottery:** use \(p_i=1-(1-p_0)^{C_i/C_0}\), or its small-probability linear approximation.
5. **Termination:** EOS or a deterministic truncation at \(K\); no arbitrary miner-selected cutoff.
6. **M1 treatment:** if canonical \(M_1\) is supplied to all miners, charge it as a separate prompt-prefill service and weight only the \(M_2\) work in the mining lottery. If miners recompute \(M_1\), include it in \(C_i\).

The burn condition for chosen prompts must hold for every allowed work class, especially the shortest one:

\[
B_0+B_{\mathsf{burn}}(C_i)
\;>\;
p_iV_{\max}
\]

after accounting for any precomputation cost and the value of block control. A purely proportional burn can be insufficient for very short queries, which is why a fixed \(B_0\) is useful.

### Experiments needed before claiming fairness

For \(p\) and \(k\) grids, measure \(M_1\) time, native inference time, proof time, VRF/sampler time, peak memory, and proof size. Fit and publish \(C(p,k)\), then simulate miners with different hardware rates and different query-length mixes. The key tests are:

- block-win share versus normalized work rate, not query count;
- invariance of reward share when miners prefer short versus long queries;
- sensitivity to EOS-heavy workloads and forced truncation;
- effect of cached \(M_1\) state and residual \(M_2\) reuse;
- rounding error in the weighted threshold;
- fee-cap refunds and the profitability of chosen-prompt attacks.

The main claim should be that variable-length queries are compatible with PoML through a complexity-weighted lottery and a work-budget reduction. A constant per-query lottery probability is not fair once response lengths vary.

## Experiments required for an LLM PoML extension

The following experiments cover the empirical claims, protocol parameters, and implementation risks identified in the discussion.

### A. Model randomness and computational independence

1. **Honest prefix-collision curves.**  Select prompts from common language-model benchmarks, run many independent VRF/random seeds per prompt, and record for each prefix length \(\ell\) whether two traces share the same input state \(C_\ell(x)\). Report both per-prompt curves and aggregate quantiles; do not report only a single average.

2. **Token-distribution and entropy profile.**  For every decoding step, record top-1 probability, entropy, top-\(k\)/nucleus mass, and EOS probability. Compare greedy, temperature, and the proposed VRF sampler. This identifies prompts with highly concentrated distributions and gives parameters for collision and chosen-prompt analyses.

3. **Response-length distribution.**  Sample the same benchmark prompts under the exact protocol sampler and measure the distribution of EOS positions and truncation rates for several values of \(K\). This is needed to estimate typical work, fee refunds, and how often actual-length weighting is exercised.

4. **Cache and precomputation advantage.**  Compare a fresh miner with one holding prompt embeddings, prompt KV state, decoder KV state, or partial traces from earlier queries. Use exact, near, and unrelated prefixes. Measure saved FLOPs/constraints and report the largest fractional saving as the empirical \(\varepsilon_{\mathrm{total}}\) bound.

### B. M1/M2 and proof feasibility

5. **M1 prefill benchmark.**  Measure prompt-prefill latency, memory, and state size as a function of prompt length. Compare recomputing \(M_1\) per miner with a canonical \(M_1\) transaction and state commitment distributed to all \(M_2\) miners.

6. **M1 verification alternatives.**  Prototype deterministic validator recomputation plus a canonical hash, an optimistic challenge/fraud-proof path, and (if practical) an \(M_1\) ZK proof. Measure verifier work, bandwidth, storage/availability overhead, and the time until the state can safely enter the mining pool.

7. **End-to-end DeepProve/zkML scaling.**  For the selected small LLM, benchmark proofs for several prompt lengths \(p\) and output lengths \(k\). Record proving time, verification time, peak memory, proof size, and circuit constraints separately for prefill, one-token decoding, and full \(x\|y\) generation.

8. **Randomized decoding integration.**  Replace argmax with sampling from the model distribution using the protocol VRF. Prove that the sampled token is the correct inverse-CDF/top-\(k\) selection for the VRF value, and measure the additional circuit constraints and proving time per token. Check that the result remains deterministic for a fixed model, query, VRF key, and chain binding.

9. **Numerical and cross-hardware reproducibility.**  Run the canonical inference and proof-generation code on different CPU/GPU types and parallelization settings. Verify identical token choices, commitments, and proof statements, or quantify and eliminate discrepancies using fixed-point arithmetic and a fixed reduction order.

### C. Variable-cost fees and lottery fairness

10. **Work-schedule calibration.**  On a grid of \((p,k)\), measure model FLOPs, memory traffic, VRF cost, proof constraints, proving time, and verification time. Fit the public schedule \(C(p,k)\), test whether a token-linear approximation is adequate, and choose conservative rounding rules.

11. **Weighted-lottery fairness simulation.**  Compare the existing flat lottery, a single weighted hash \(p_i=1-(1-p_0)^{w_i}\), and integer virtual work-unit tickets. Simulate miners processing mixtures of short and long queries and test whether block-win share tracks normalized completed work rather than query count.

12. **Heterogeneous-miner consensus simulation.**  Give miners different hardware rates and query-length mixes. Measure honest/adversarial lottery shares, block intervals, chain growth, common-prefix violations, and chain quality under the work-budget reduction. Verify that the honest-majority condition must be stated in normalized work, not simply in the number of miners.

13. **Optional-stopping and grinding tests.**  Test whether a miner can improve its odds by choosing among registered queries, reordering queries, trying multiple VRF keys, stopping at a favorable prefix, or declaring an excessive \(K\). Bind all such choices to the query, VRF transcript, response length, and proof before threshold evaluation; measure any remaining advantage.

14. **Fee-cap and termination tests.**  Exercise EOS, exact-\(K\) truncation, underfunded registrations, expiry, and refunds. Confirm that a miner cannot omit a costly suffix, claim a cheaper length, or receive a reward without a valid proof of EOS/truncation.

15. **Chosen-prompt economic attack.**  Search for prompts with highly concentrated outputs, let an adversary precompute under arbitrary randomness, and estimate the probability of matching the later VRF trace. Combine this with query fees, base burn, proof cost, and block-control value to verify negative expected value across all allowed lengths.

### D. System-level feasibility

16. **Throughput and queueing benchmark.**  Measure end-to-end time from registration to an accepted \(M_1\) state and then to an \(M_2\) proof/block. Vary query arrival rate, maximum \(K\), number of miners, and proof parallelism; report whether the system remains live under realistic load.

17. **Bandwidth, storage, and availability.**  Measure the size and dissemination time of prompts, committed KV states, ciphertexts, VRF transcripts, and ZK proofs. Test whether a malicious worker can post a commitment while withholding the \(M_1\) state, and size the availability bond or data-availability requirement accordingly.

The minimum experimental package for claiming an LLM PoML candidate is experiments 1--4, 7--8, 10--12, and 15. The remaining experiments are needed to turn the prototype into a deployable protocol and to justify its operational parameters.

## Chosen-prompt precomputation under DeepProve and a VRF

It is important to distinguish three objects that an adversary might precompute:

1. the output token sequence \(y\);
2. the model-side witness, including logits or activations for \(x\|y\);
3. a complete DeepProve proof.

Precomputing only \(y\) is not sufficient to publish a valid PoML response. If the later VRF transcript happens to sample exactly \(y\), the adversary can skip the ordinary autoregressive generation phase, but a validator still requires a valid ZK proof. DeepProve certifies the full prompt-plus-response sequence by evaluating the model over \(x\|y\) and proving that every position's prediction is consistent with the next claimed token. Although intermediate activations are not revealed or separately committed, the prover still has to perform the layer-by-layer sumcheck/GKR proving work. See [Lagrange's end-to-end DeepProve description](https://lagrange.dev/engineering-updates/inside-deepprove-proving-an-llm-end-to-end).

The outcome depends on how the PoML proof statement is constructed:

- If the statement contains only \((h_\theta,x,y)\), a proof precomputed for that same pair may be replayable. Separately publishing a fresh VRF transcript does not establish that the proof's token choices were derived from that transcript.
- If the statement binds \(\mathsf{qid}\), the parent/block binding, a commitment to the fresh VRF outputs, \(x\), \(y\), and the sampling rule, then an old proof is not a proof for the new statement. The miner must generate a fresh Fiat--Shamir transcript and proof.
- The VRF itself can be verified outside the zkML proof, but the zkML proof must take the verified VRF output or its commitment as a public input and prove that every \(y_t\) is the result of applying the specified sampler to the model logits and the corresponding VRF-derived value.

A suitable public statement is schematically

\[
\mathsf{stmt}
=
(h_\theta,\mathsf{qid},\mathsf{bind},c_x,c_{\mathcal R},c_y,K),
\]

where validators separately verify the VRF transcript opening \(c_{\mathcal R}\), and the zkML relation proves

\[
\forall t\leq k:
\quad
\ell_t=M_\theta(x,y_{<t}),
\qquad
y_t=\mathsf{Sample}(\ell_t,z_t),
\]

for the committed VRF value \(z_t\).

The adversary's advantage therefore has three cases:

| Precomputation | Work saved after a VRF match | Work still required |
|---|---|---|
| Output \(y\) only | Ordinary autoregressive generation | Full model witness/prover work and a fresh proof |
| \(y\) plus logits/activations | Generation and possibly witness construction | Fresh sumcheck/GKR/Fiat--Shamir proof work |
| \(y\) plus an old proof | Potentially everything only if the statement is replayable | A fresh proof if the statement is bound to the new query/block/VRF |

DeepProve's one-pass construction makes this distinction particularly relevant: an honest miner first generates \(y\) autoregressively and then proves \(x\|y\) in one pass, whereas an attacker whose precomputed \(y\) matches can omit the first phase but cannot omit the second. The official DeepProve description says that it proves every layer and generated token and currently describes deterministic argmax plus seed-derandomized sampling, but PoML's per-token VRF sampler would still require an explicit integration and soundness argument.

Thus the chosen-prompt attack is real, but the claim “publish only the VRF and precomputed output” is false in a correctly designed protocol. The residual advantage is the generation/witness work that can be reused before producing a fresh bound proof. How much of DeepProve proving can itself be reused is not established by output correctness and must be treated as a separate proof-computational-independence assumption or measured experimentally.

## Preventing replay of a precomputed DeepProve proof

There is no cryptographic way to tell when a valid proof was generated if it is a proof for exactly the same public statement. A proof system normally allows anyone to copy and replay a valid proof for that statement. Therefore, requiring fresh prover coins or a new local proof nonce is not sufficient: an old proof remains valid.

Replay prevention must instead make the statement itself fresh and unpredictable before the query becomes mineable. A suitable two-stage design is:

1. The query and canonical \(M_1\) state are registered and accepted, with the prompt commitment and fee fixed.
2. A beacon \(b\), revealed only after that acceptance, is used in the mining seed:
   \[
   r_i=H(b\parallel\mathsf{qid}_i\parallel\mathsf{pos}_i\parallel\mathsf{vk}_m).
   \]
3. The inference VRF outputs \(z_{i,t}\) are derived from \(r_i\), and the proof statement binds \(b\), \(r_i\), all \(z_{i,t}\) (or a commitment to them), \(\mathsf{qid}_i\), the block/position binding, \(x_i\), and \(y_i\).
4. The zkML relation proves both the model computation and the sampling relation \(y_{i,t}=\mathsf{Sample}(M_\theta(x_i,y_{i,<t}),z_{i,t})\).

The beacon must be unpredictable to the adversary before query acceptance and not controllable by the miner. A parent-block hash that is already known when the query is registered does not provide this property. If the miner can choose the entire future statement before doing its “precomputation,” replay is unavoidable; the work must simply be counted as legitimate mining work.

Under this design:

- a proof precomputed for \((x,y)\) but not bound to \(b\) and the VRF transcript is rejected;
- a proof precomputed for an old \(b\), query identifier, or block position is rejected;
- a proof for the same token sequence but a different VRF transcript is rejected unless the proof relation deliberately omits the sampling check;
- an old proof cannot be made fresh merely by attaching a new VRF proof outside the zkML relation.

The verifier's canonical transcript must include every public statement field, including the beacon and VRF commitment. Otherwise a field that is present in the protocol message but omitted from Fiat--Shamir domain separation may not actually prevent replay.

This construction prevents replay of a complete proof, but it does not prove that the prover recomputed every internal value from scratch. An attacker may still precompute the output, logits, activations, or model-specific proving data and reuse part of the fresh proof computation. The remaining saving belongs in the proof-computational-independence parameter (or in an explicit empirical bound), while the unpredictable beacon prevents the strongest “precompute the entire accepted proof” attack.

## Distributional CIA is not enough for the adversarial reduction

The honest-prompt prefix-collision experiment and the cryptographic CIA assumption answer different questions. The experiment estimates reuse on a specified prompt distribution. The security theorem must quantify over an adversary that can choose prompts adaptively, including prompts with extremely concentrated token distributions. Therefore a low average collision curve cannot establish the CIA premise used in the current reduction.

Fee burns and negative expected value are also not substitutes for this premise in an unconditional backbone theorem. They may deter a rational economic adversary under an explicit bound on the value of block control, but a cryptographic adversary may execute a loss-making strategy. Burns should be presented as an economic mitigation, not as proof that the attack is impossible.

### A reduction-preserving replacement: uniform bounded amortization

Replace distributional model CIA with a worst-case assumption that holds for every admissible prompt, including adversarially chosen prompts:

\[
\Pr\!\left[
C_{\mathcal A}(x,\sigma,b)
<
(1-\varepsilon_{\mathrm{tot}})\,
C_{\mathrm{fresh}}(x)
\right]
\leq
\nu(\lambda),
\]

where \(x\) is any registered prompt, \(\sigma\) is any adversarial precomputation state, \(b\) is the fresh mining challenge, \(C_{\mathrm{fresh}}\) is the cost of a fresh valid inference-proof pair, and \(\nu\) is negligible. This is a uniform bounded-amortization (or worst-case CIA) assumption. It is much stronger than an empirical benchmark result and should be stated as an assumption unless it is proved for the chosen model/prover.

The cost should be decomposed rather than bounded by the maximum of two fractional savings:

\[
C_{\mathrm{fresh}}=C_{\mathrm{inf}}+C_{\mathrm{proof}},
\qquad
\varepsilon_{\mathrm{tot}}
\leq
\frac{\varepsilon_{\mathrm{inf}}C_{\mathrm{inf}}
      +\varepsilon_{\mathrm{proof}}C_{\mathrm{proof}}}
     {C_{\mathrm{inf}}+C_{\mathrm{proof}}}.
\]

This leaves one possible LLM-specific route. A chosen prompt may make \(\varepsilon_{\mathrm{inf}}\) close to one because the token sequence is reusable, while a fresh challenge-dependent DeepProve proof may still have \(\varepsilon_{\mathrm{proof}}\) close to zero. If proof generation dominates total work, the weighted \(\varepsilon_{\mathrm{tot}}\) can remain below the threshold needed to prevent an additional effective lottery attempt. This must be demonstrated as a worst-case bound on the prover, not inferred from average prompt collisions.

For the current fixed-cost backbone theorem, the condition remains an explicit bound such as

\[
\varepsilon_{\mathrm{tot}}<\frac{1}{q_{\mathrm{ML}}+1},
\]

with \(q_{\mathrm{ML}}\) interpreted using the fresh inference-plus-proof cost. For variable-length queries, replace \(q_{\mathrm{ML}}\) by a normalized work budget and use the corresponding weighted-lottery reduction. The proof must count every virtual work unit that a valid response can create.

### Protocol conditions needed for this route

1. The query must be accepted before an unbiasable beacon \(b\) is revealed.
2. The beacon, query identifier, mining position, VRF outputs, prompt commitment, response, and maximum length must be included in the canonical DeepProve statement and Fiat--Shamir transcript.
3. The zkML relation must prove the VRF sampling rule, not merely model consistency for a claimed output.
4. A proof precomputed for another beacon or VRF transcript must be invalid.
5. The proof implementation must expose or experimentally bound any reusable model/prover preprocessing.

These conditions prevent complete-proof replay, but they do not automatically prove the weighted amortization inequality. If a miner can reuse enough model or prover work that the inequality fails for some chosen prompt, the current backbone reduction does not apply.

### What can honestly be claimed for GPT-2

There are three defensible levels of claim:

- **Formal instantiation:** prove or assume the uniform bounded-amortization condition for the GPT-2 plus DeepProve pair, with a concrete \(\varepsilon_{\mathrm{tot}}\) bound.
- **Conditional instantiation:** state the backbone theorem under that assumption and use collision/cache/proof experiments only to show plausibility and estimate parameters.
- **Feasibility result:** demonstrate VRF-bound randomized decoding and end-to-end proofs for GPT-2, while explicitly stating that consensus security against arbitrary chosen prompts remains open.

The honest prefix-collision experiment remains valuable for estimating normal workload reuse and for fee/throughput engineering. It should not be described as evidence that the adversarial CIA assumption holds universally. If no uniform bound can be established, GPT-2 can still be a useful engineering candidate, but it should not be presented as an unconditional model-class instantiation of the current \(5\_reduction\) theorem.

## Quantitative answer for a specialized low-entropy prompt

The prompt "Repeat after me: The sky is blue." is a useful counterexample to an overly optimistic interpretation of VRF sampling. There is no protocol-independent percentage of work that can be declared reusable from the prompt alone. The relevant quantity is the work that can be done before the fresh mining challenge is revealed, conditional on the challenge selecting a response that was precomputed.

Under DeepProve's current deterministic-argmax path, a fixed quantized model and prompt give a fixed \(y^\star\), so the candidate match probability is one. A VRF sampler is therefore not an optional implementation detail for PoML: it must be part of the challenge-bound relation, and its effect on the model/prover trace must be analysed separately.

Let \(P_x(y)\) denote the probability that the protocol sampler emits a particular continuation \(y\) for prompt \(x\). For a top-\(p\), top-\(k\), or inverse-CDF sampler, many different VRF values lie in the same token-selection interval. Consequently, the VRF values can be different while the selected token is identical. For a fixed candidate \(y^\star\), the match probability is

\[
\Pr[Y=y^\star\mid x]=P_x(y^\star),
\]

and for a precomputed set \(S\) of candidates it is \(\sum_{y\in S}P_x(y)\). This is the chosen-candidate probability, not the two-sample collision probability \(C_k(x)\). A highly specialized prompt can make \(P_x(y^\star)\) large, potentially close to one for the first several tokens, even though two VRF bit strings almost never coincide.

The work accounting should be stated separately from this match probability. For one query, write the fresh cost as

\[
C_{\mathrm{fresh}}
  = C_{\mathrm{gen}}+C_{\mathrm{proof}},
\qquad
C_{\mathrm{proof}}
  = C_{\mathrm{wit}}+C_{\mathrm{chal}}.
\]

Here \(C_{\mathrm{gen}}\) is the ordinary autoregressive generation and sampler work, \(C_{\mathrm{wit}}\) is the query-specific model/witness work needed by DeepProve, and \(C_{\mathrm{chal}}\) is the part of commitment, sumcheck/GKR, Fiat--Shamir, sampler, and aggregation work that depends on the fresh challenge. Model parsing, quantization, proving keys, and static lookup tables are one-time setup and should either be supplied to every miner or excluded from the per-query cost; otherwise the work schedule counts a cacheable common setup as if it were fresh mining work.

Conditional on a match, the useful saving is

\[
\alpha_{\mathrm{match}}
  =1-\frac{C_{\mathrm{online}}}{C_{\mathrm{fresh}}},
\]

where \(C_{\mathrm{online}}\) is the work that must still be done after the beacon/VRF is available. The corresponding throughput factor is \(1/(1-\alpha_{\mathrm{match}})\), before charging the offline precomputation and storage. This is a conditional speed-up; the expected amortized cost also includes the probability of a match and the cost of building the cached state.

The three important precomputation levels are therefore:

| State held before the beacon/VRF | Condition for use | Potential saving on a match | Work that remains fresh |
|---|---|---|---|
| Candidate tokens \(y^\star\) | The VRF sampler selects exactly \(y^\star\) | \(C_{\mathrm{gen}}\) | DeepProve witness/prover work and the bound sampler/VRF relation |
| Candidate tokens plus model trace for \(x\Vert y^\star\) | The selected sequence is \(y^\star\) | \(C_{\mathrm{gen}}\) plus any reusable \(C_{\mathrm{wit}}\) | Challenge-dependent commitments, sumchecks, Fiat--Shamir transcript, and aggregation |
| A complete old proof | Only if the public statement is unchanged | Nearly all work | Nothing, if replay is allowed; otherwise a fresh proof/transcript |

The second row is the important one for the proposed DeepProve extension. DeepProve's end-to-end construction feeds the complete claimed sequence \(x\Vert y\) through the transformer once and proves every operator; it does not use the VRF value as an input to the transformer graph merely because the final token was sampled with that value. If the same \(y^\star\) is selected for many different VRF outputs, the sequence \(x\Vert y^\star\), its model activations, and its model-side witness are the same. Thus replacing deterministic argmax by VRF-based sampling does not, by itself, make the model computation computationally independent. It only changes the sampling relation (and, if correctly bound, the proof statement). DeepProve's description also notes that intermediate activations are not committed as public values; that limits what a verifier learns, but it does not prevent a prover from retaining those values privately for later work ([DeepProve engineering description](https://lagrange.dev/engineering-updates/inside-deepprove-proving-an-llm-end-to-end)).

The old proof itself should nevertheless be rejected. If \(z_t\), the beacon, the query identifier, and the block/position binding are public inputs to the relation and to Fiat--Shamir domain separation, a proof generated for another \(z_{1:k}\) is a proof for a different statement. A new VRF proof attached outside the zkML relation is not enough: the relation must prove

\[
y_t=\mathsf{Sample}(\ell_t,z_t),
\qquad
\ell_t=M_\theta(x,y_{<t}),
\]

for the same \(z_t\) that is verified by the protocol. Binding \(z_t\) makes complete-proof replay infeasible, but it does not imply that all prover work is fresh. A miner can still precompute the model witness and any challenge-independent proof data. Fiat--Shamir makes the challenges unpredictable; it does not, on its own, erase a witness or prohibit precomputation of the first part of a prover transcript.

This gives a direct answer to the question about proportions:

- If the attacker stores only the likely response, the saving is at most \(C_{\mathrm{gen}}/C_{\mathrm{fresh}}\). In a DeepProve deployment where proof generation dominates native decoding, this may be a small wall-clock fraction, but it is not zero.
- If the attacker stores the model trace or a reusable prover state for \(x\Vert y^\star\), the saving can include most of \(C_{\mathrm{wit}}\) and any challenge-independent prover prefix. The fraction can therefore be large, even close to one in an implementation where the fresh challenge-dependent tail is small. No numerical percentage is justified until DeepProve is instrumented at this boundary.
- If the attacker stores a complete proof and the statement omits the VRF relation, the saving is effectively 100% and the protocol is replayable. If the statement is correctly fresh, the old proof is unusable, but partial prover-state reuse remains an open quantitative question.

The empirical quantity to report is consequently not just proof time, but the online proof time after a beacon reveal. For each adversarially selected prompt and candidate set, record (i) offline work before the beacon, (ii) online work after the beacon, (iii) which DeepProve phases were reused, (iv) storage and preparation cost, and (v) the match probability. The experiment should instrument model execution, witness construction, polynomial commitments, each Fiat--Shamir round, the VRF-sampler subrelation, and chunk aggregation separately. Report the worst-case or a stated high quantile over prompts, not only an average over natural-language benchmarks.

## What this requires from the Section 5 reduction

The current reduction can remain useful for an LLM only if its computational-independence premise is changed from an informal claim about different random seeds to a uniform online-cost bound. Define

\[
\varepsilon_{\mathrm{tot}}(x,\sigma,b)
  =1-
    \frac{C_{\mathrm{online}}(x,\sigma,b)}{C_{\mathrm{fresh}}(x)},
\]

where \(x\) is any admissible prompt, \(\sigma\) is any state produced by adversarial precomputation before the beacon, and \(b\) is the fresh challenge. The assumption needed by the backbone argument is a bound of the form

\[
\Pr_b\!\left[
  \varepsilon_{\mathrm{tot}}(x,\sigma,b)>\bar\varepsilon
\right]
\leq \nu(\kappa)
\quad\text{for every admissible }(x,\sigma),
\]

with negligible \(\nu\) and

\[
\bar\varepsilon < \frac{1}{q_{\mathsf{ML}}+1}.
\]

Equivalently, if separate bounds are established for generation and proving, then

\[
\bar\varepsilon
\geq
\frac{\varepsilon_{\mathrm{gen}}C_{\mathrm{gen}}
      +\varepsilon_{\mathrm{proof}}C_{\mathrm{proof}}}
     {C_{\mathrm{gen}}+C_{\mathrm{proof}}},
\]

and the weighted quantity, rather than either component in isolation, must satisfy the Section 5 threshold. If the bound fails on a chosen prompt, the correct conclusion is that the adversary has an effective query rate

\[
q_{\mathsf{ML}}^{\mathsf{adv}}
  =\left\lfloor
     \frac{q_{\mathsf{ML}}}{1-\bar\varepsilon}
    \right\rfloor,
\]

so the honest-majority condition must be stated in normalized work using this rate (or the variable-cost lottery), rather than silently retaining the honest rate.

Two textual repairs are therefore needed in the paper. First, the assertion that Fiat--Shamir automatically gives \(\varepsilon_p=\mathsf{negl}(\kappa)\) should be presented as an explicit assumption about the *online* DeepProve prover, not as a consequence of transcript unpredictability. Second, the LLM should belong to a restricted class \(\mathcal{K}_{\mathrm{LLM}}(\bar\varepsilon)\) only after the bound is proved or measured under the stated precomputation and storage model. Without that bound, the GPT-2 result is a feasibility demonstration for VRF-bound randomized proving, not an unconditional instantiation of the backbone theorem.

## Can reuse be eliminated?

Complete-proof replay can be prevented with the two-stage registration/beacon design already described: accept and commit the query first, reveal an unbiasable beacon second, derive per-token VRF values from it, and bind all of those values into the DeepProve statement and Fiat--Shamir transcript. Absolute zero reuse is not achievable, because model-specific setup, quantization tables, and other public preprocessing are inherently reusable and because any proof for an unchanged statement is copyable.

To make the *query-specific* reusable fraction small, the fresh challenge must affect the model-side trace, not only the final token-selection check. Possible constructions include a protocol-defined stochastic model whose per-token randomness changes hidden states, or a challenge-keyed randomized encoding/masking applied throughout the proven computation and checked for correct unmasking. Both options have to be analysed for algebraic preprocessing, model quality, quantization, and proof overhead; a mask that can be stripped outside the circuit does not provide the desired bound. If changing the model trace is unacceptable, the conservative alternatives are to keep the theorem conditional on the uniform bounded-amortization assumption, restrict the allowed prompt distribution and say so explicitly, or add a separate prehash/hash-sandwich mechanism that supplies the missing unpredictability at the cost of additional non-useful work.

The effective challenge entropy is the entropy that changes the proven computation, not the nominal length of the VRF output. If the sampler consumes only a few bits, or if a broad interval of VRF values produces the same token sequence and the same witness, an attacker may precompute a table over those equivalence classes. The protocol should therefore measure the number and cost of distinct challenge-dependent traces under the actual sampler and circuit, rather than infer a \(2^\kappa\) anti-precomputation factor from the VRF output length alone.

The key conclusion is therefore: binding the VRF values into the public statement and Fiat--Shamir transcript replay-protects a DeepProve proof, while placing the sampler relation inside the proof is additionally required to establish that the output was derived from those values. VRF randomness at the sampler does not by itself prevent reuse of the transformer computation. For the specialized prompt, the inference saving can be nearly complete at the output-selection level, and the proof saving can be anywhere from negligible to dominant depending on how much of the witness/prover pipeline remains reusable. The paper should claim a concrete percentage only after measuring the online fraction and then use that measured or assumed worst-case value in the \(5\_reduction\) bound.

## Current framework audit: does the VRF stop ZKP precomputation?

The parent-derived binding is already a delayed challenge relative to work performed before the parent block is published. The important qualification is that it is delayed only for queries that are registered before that parent and for which the transaction snapshot is fixed at the start of the mining round.

### Current protocol as written

In the current block-production algorithm, the first binding is \(\mathsf{bind}_1=G(s,\mathsf{tx}_{\mathrm{std}})\), and the miner computes

\[
r_i=H(\mathsf{bind}_i\parallel h_{u,i}\parallel\mathsf{qid}_i\parallel\mathsf{vk}^{\mathrm{sig}}_m)
\]

before evaluating the inference VRF and invoking the prover. The parent hash \(s\) is known, the miner chooses \(\mathsf{tx}_{\mathrm{std}}\), and the miner controls the VRF secret key. Therefore the miner can evaluate \(r_i\), all \(z_{i,t}\), and the corresponding DeepProve statement before doing the proof work. For \(i\geq2\), the miner can continue sequentially after computing \(\pi_{i-1}\). The VRF gives uniqueness and verifiability for a fixed input; it does not give the key holder an externally hidden challenge.

Before \(B_{\mathsf{prev}}\) is published, \(s=H(B_{\mathsf{prev}})\) is unknown, so an attacker cannot know the exact \(\mathsf{bind}_1\), \(r_1\), VRF outputs, ciphertext, or complete proof for the eventual canonical parent, except by guessing the parent or privately building an alternative fork. Once the parent is published and \(\mathsf{tx}_{\mathrm{std}}\) is frozen, the miner can evaluate \(r_i\) and generate the complete proof before evaluating the lottery hash. That is ordinary mining work, not by itself an extra ticket.

The parent binding therefore blocks exact cross-round proof replay, but it does not establish that the work is computationally independent across candidate bindings. A low-entropy prompt may allow the attacker to prepare the same output, model trace, or prover state for many possible \(s\) values. The \(i\geq2\) rule \(\mathsf{bind}_i=H(\pi_{i-1})\) additionally serializes the proof chain, so the exact next statement is unavailable until the preceding proof exists; it still does not prevent reuse of state that is independent of that statement. The NIZK computational-independence game must model this timing explicitly: the fresh challenge is the parent-derived \(s\), not merely a random value sampled after an abstract adversary phase.

There is an additional distinction for \(\mathsf{RESPONSE\_QUERY}\) transactions. The current validity rule explicitly does not require the losing response's \(r\) to be derived from the mining binding. Such a response proof can therefore be fully precomputed by construction. This need not create an extra lottery ticket if response transactions are excluded from the mining lottery, but it must be treated separately in any claim about fair compensation or non-reuse of useful work.

### When an additional beacon is needed

The parent hash can serve as the delayed beacon if query registration occurs before the parent is known, the query is not eligible for same-block registration, and the parent-selection process is not controlled by the prover. An additional beacon is needed if a query can be registered after \(s\) is known, if the miner can choose the relevant parent/fork after precomputation, or if the transaction snapshot remains mutable after the supposed cutoff. In that case, use an independent beacon \(b\) revealed after query acceptance, with

\[
r_i=H(b\parallel\mathsf{qid}_i\parallel\mathsf{pos}_i\parallel\mathsf{vk}_m),
\]

then the attacker cannot precompute a proof for the exact future statement except by guessing \(b\) or building a table over many possible beacons. If \(b\), \(r_i\), \(z_{i,t}\), the query identifier, and the block binding are all public inputs to the DeepProve relation and Fiat--Shamir transcript, a proof made for another beacon is rejected. The same complete-proof freshness property is obtained from \(s=H(B_{\mathsf{prev}})\) under the parent-registration timing above.

It does not prevent all ZKP-related precomputation. Before \(b\) is revealed, the attacker can still prepare model setup, lookup tables, candidate outputs, the \(x\Vert y\) model trace, witness data, and any challenge-independent prover prefix. For a low-entropy prompt, the candidate \(y\) and the model trace may remain valid for many different \(b\)'s because \(b\) affects only the sampler relation. The fresh work is then the challenge-dependent proof tail, whose fraction is implementation-specific and could be small or large. Fiat--Shamir unpredictability alone supplies no lower bound on that online fraction.

The strongest defensible quantitative statement is therefore:

| Protocol variant | Complete proof precomputation | Partial model/prover precomputation |
|---|---|---|
| Parent-derived \(s\), with registration before the parent | Exact proof for the eventual parent cannot be prepared beforehand; it can be generated after \(s\) is published | Output, model trace, and challenge-independent prover state may be reused across parents |
| Additional delayed beacon, VRF bound only at statement/sampler | Prevented except for guessing or a large beacon table | Still allowed; must be measured or assumed |
| Delayed challenge plus challenge-dependent model trace | Prevented for the challenge-specific trace, subject to the encoding's security argument | Static setup remains reusable; residual fraction still needs a bound |

For Section 5, the relevant assumption is an online-prover game. For every prompt \(x\) and precomputation state \(\sigma\) produced before \(b\) is revealed, require

\[
\Pr_b\!\left[
\begin{array}{c}
\mathsf{Verify}(\mathsf{stmt}(x,b),\pi)=1\\[-1pt]
\land\ \mathsf{Depth}(\mathsf{OnlineProver}(\sigma,x,b))
< (1-\varepsilon_p)T_{\mathrm{proof}}(x)
\end{array}\right]\leq \nu(\kappa).
\]

Under the intended parent-registration timing, the current protocol does satisfy the *complete-proof freshness relative to the parent* part. It does not, however, establish a small \(\varepsilon_p\) for partial reuse of model or prover work. Thus the corrected answer is: the parent hash can play the role of a delayed beacon for exact-proof freshness, but preventing reuse of the underlying DeepProve computation still requires an explicit bound or a challenge-dependent model trace.

## Exact-output collisions under VRF sampling

This is the concrete worst case for the proposed VRF extension. Let
\[
p_x=\Pr_{R,R'}[M_\theta(x,R)=M_\theta(x,R')]
\]
for independent VRF-derived randomness values \(R,R'\). If the transformer itself is deterministic and the VRF is consumed only by the final token sampler, then whenever the sampled token sequence is the same, the autoregressive prefixes are the same at every step. Consequently the entire transformer trace (logits, hidden states, and the witness values used for those layers) is identical. A prompt such as ``Repeat after me: The sky is blue'' can have \(p_x\) close to one.

The attacker can therefore cache the inference trace and all challenge-independent preprocessing once, then run only the VRF evaluations, the sampler checks, encryption, and the challenge-dependent part of the prover for each new parent. The complete proof \(\pi\) should still be non-replayable when \(R\), the binding, and the Fiat--Shamir transcript are public inputs: a proof generated for another \(R\) is rejected. That statement-level freshness must not be confused with freshness of the expensive witness or prover work. In a monolithic DeepProve implementation, changing the public input may also change many Fiat--Shamir challenges, so the attacker may have to recompute substantial sumcheck/commitment work; nevertheless, the cached trace and polynomial data can still be valuable. In a modular implementation, a transformer subproof independent of the sampler could be reusable exactly. The fraction is implementation-specific and cannot be inferred from the VRF output length.

A useful accounting variable is
\[
\rho_x = p_x\,\frac{C_{\mathrm{reusable}}(x)}{C_{\mathrm{fresh}}(x)},
\qquad
C_{\mathrm{fresh}}=C_{\mathrm{trace}}+C_{\mathrm{proof}}+C_{\mathrm{VRF}}+C_{\mathrm{sampler}}.
\]
The current framework gives no non-trivial protocol-level upper bound on \(\rho_x\): in the worst case \(p_x\to1\) and the reusable trace dominates, so \(\rho_x\) can approach one. We should not insert a percentage in Section~5 without profiling the actual prover phases and then making the resulting worst-case online fraction an explicit assumption.

### What would actually prevent this reuse?

1. **Challenge-dependent model computation.** Make the delayed beacon affect the expensive trace, not only the final sampler. For example, include a beacon-derived nonce in the model input and require the output to depend on it, or use a challenge-keyed internal re-randomisation/masking at every layer. The relation must prove the transformed computation and its correct unmasking. A merely algebraic mask that can be stripped from a cached trace does not help; the construction needs a security and cost argument showing that a fresh trace (or a specified fraction of it) is required.

2. **Challenge-dependent useful work with a changed semantics.** Requiring several beacon-seeded inferences, or sampling at internal model layers, can force fresh computation, but it changes the service being proved and may make outputs incomparable across miners. This is a protocol-design trade-off, not a free anti-precomputation property.

3. **Separate online sequential work.** A beacon-seeded VDF or other sequential computation can guarantee a lower bound on post-beacon work even if the model trace is cached. This preserves a quantitative mining bound, but the guaranteed fresh work is then auxiliary work rather than fresh LLM inference.

Random spot checks, delayed commitments, or putting \(R\) only into Fiat--Shamir do not solve the exact-output case: they can reject an incorrect or replayed proof, but they cannot tell whether a valid deterministic trace was computed before or after the beacon. If the model semantics must remain unchanged and VRF randomness is used only for token selection, the honest Section~5 assumption is therefore the conservative one: specialized prompts may have nearly complete inference reuse, and the theorem must bound (or explicitly tolerate) that reuse rather than claim that the delayed parent hash eliminates it.

## Literature-guided design space for challenge-dependent inference

The literature does not contain a standard LLM technique whose purpose is to make every inference instance computationally fresh for a verifier. It does, however, give useful building blocks. The relevant distinction is whether randomness changes only the sampled answer, changes the input representation, or changes the hidden-state computation throughout the network. Only the latter two can plausibly reduce the reusable DeepProve trace.

### Closest prior work

| Work | Where randomness enters | What it establishes for this design | Important limitation |
|---|---|---|---|
| [DeepProve](https://lagrange.dev/engineering-updates/inside-deepprove-proving-an-llm-end-to-end) | A full transformer computation is proved with GKR/sumchecks; the original system handles sampling by a deterministic seed/argmax relation | The entire prompt-to-response trace can be put in one proof, so a VRF-dependent transformation has to be included in that relation rather than bolted on after proving | Public-input binding and Fiat--Shamir stop proof replay, but do not by themselves prove that the trace was computed after the beacon |
| [Random Soft Prompts (RSPs)](https://arxiv.org/abs/2605.11936) | Fresh Gaussian-like embedding vectors are appended to an input at inference time; no learned content is required | A recent, training-free precedent for injecting random vectors into an LLM and obtaining different reasoning trajectories | The paper is a recent preprint, not an anti-amortization result; it appends vectors, whereas our causal setting should test a *prefix* so that all later user-token states depend on the beacon; output quality and beacon sensitivity must be measured |
| [RESTA / Smoothed Embeddings](https://arxiv.org/abs/2501.16497) | Random noise is added to embedding vectors and outputs are aggregated across noisy runs | Embedding perturbation is an established inference-time robustness pattern and can be wrapped around a frozen model | It normally requires multiple complete forward passes and an aggregation rule; it certifies robustness, not fresh-work lower bounds |
| [SmoothLLM](https://arxiv.org/abs/2310.03684), [SAFER](https://aclanthology.org/2020.acl-main.317/), and [randomized masking](https://aclanthology.org/2023.cl-2.5/) | Random prompt perturbations (characters, words, or masks) are applied to several copies and the outputs are voted/aggregated | Demonstrates that a model can sometimes be used as a black-box under randomized input transformations, without changing its weights | These methods target adversarial robustness; they can change semantics, and the repeated copies increase cost rather than prove a per-instance online lower bound |
| [MC dropout for Transformers](https://aclanthology.org/2021.eacl-main.157.pdf) | Dropout masks are sampled at inference and predictions are aggregated | Direct evidence that stochastic inference is possible, and that *where* dropout is placed determines amortization: last-layer dropout lets almost the entire transformer body be reused, while all-layer dropout changes much more computation | The cited experiments are classifiers, not autoregressive ZK-proven LLMs. All-layer sampling has substantial multi-pass overhead, and an unmodified pretrained LLM may not preserve quality under inference-time dropout |
| [LayerDrop](https://arxiv.org/abs/1909.11556), [DropHead](https://aclanthology.org/2020.findings-emnlp.178/), and [DropKey](https://openaccess.thecvf.com/content/CVPR2023/html/Li_DropKey_for_Vision_Transformer_CVPR_2023_paper.html) | Structured layer/head/key masking, primarily during training | Supplies design precedents for structured masks in a transformer and suggests that masks can be made layer- or head-specific | These are mainly regularisation techniques. Randomly omitting computation may reduce the work that must be done, and a mask applied only to keys/values can leave reusable projections or a cached base trace |
| [NEFTune](https://arxiv.org/abs/2310.05914) and [R-Drop](https://papers.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) | Noise on embeddings or dropout consistency during *training* | Shows that models can be trained to tolerate controlled stochasticity and retain useful outputs | It is not evidence that arbitrary inference-time noise can be added safely to an already deployed model; a light fine-tuning stage may be needed |
| [Verifiable Dropout](https://arxiv.org/abs/2512.22526) | A nonce/context derives dropout masks through a VRF-like seed; a zkVM proves that the masks were generated and applied correctly | This is the closest cryptographic precedent for beacon-keyed stochastic neural computation and for putting the randomness/application inside the proof | It proves correct use of randomness, not that a prover must redo a lower-bounded fraction of useful model work. It also reports proof-generation overhead that grows with the masked tensor, so integration with DeepProve needs a separate cost analysis |
| [Proofs of Useful Work](https://eprint.iacr.org/2017/203) and [arbitrary-matrix-multiplication PoUW](https://eprint.iacr.org/2025/685) | The useful computation is designed to have limited amortisation across independently challenged instances | Gives the right conceptual standard: anti-reuse is a property of the underlying computation/task, not a consequence of attaching a proof system to a reusable computation | These are general PoUW constructions, not LLM/DeepProve theorems. They motivate an explicit anti-amortization assumption or reduction, but do not supply one for a transformer today |

The most useful combination for PoML is therefore: use the *training-free random-embedding* literature to select a challenge-dependent model input, use the *verifiable-dropout* literature for binding and proving the stochastic operation, and use the *PoUW* literature for the statement that a fresh-work bound must be proved separately from proof soundness.

### Which injection point changes the reusable trace?

For a decoder-only transformer, write \(h^0_t=E(x_t)+p_t\), followed by per-layer projections \(Q^l=h^{l-1}W_Q^l\), \(K^l=h^{l-1}W_K^l\), \(V^l=h^{l-1}W_V^l\), attention, residual connections, and the feed-forward block. The following ranking is a hypothesis to test, not a theorem:

| Injection point | Expected effect on precomputation | Assessment |
|---|---|---|
| Final logits, temperature, or token sampler only | Hidden states, logits, and transformer witness remain reusable whenever the output sequence collides | Reject as an anti-reuse mechanism; retain only for output randomness |
| Random suffix after the user prompt | Earlier prompt states and much of the cached prefix remain unchanged in a causal model | Usually insufficient; if using random soft prompts, place them before the user query |
| Random prefix embeddings | Every subsequent prompt and generated-token state is conditioned on the beacon; a collision now requires the perturbed computation to converge to the same trace/output | Best first training-free candidate, inspired by RSPs; semantic drift and model insensitivity remain open risks |
| Additive noise to embeddings or residual stream at every layer | Perturbs all downstream states and is difficult to undo from a cached trace if the noise is dense and nonlinearities are subsequently applied | Stronger than a one-time input perturbation, but likely needs calibration or stochasticity-aware fine-tuning |
| Noise to \(Q\) and \(K\) (or attention logits) before softmax | Changes attention weights nonlinearly and therefore the downstream residual stream; per-layer masks can force fresh attention work | Promising second candidate. It must be dense enough to prevent cached base attention from being cheaply corrected; Q/K is preferable to V-only noise |
| Noise to \(V\) only | Base \(QK^\top\), attention weights, and much of the trace can be cached; the value perturbation may be a cheap delta computation | Weak on its own |
| Dropout/masking at every attention and feed-forward layer | Changes intermediate activations throughout the proof; a fresh mask can prevent exact trace reuse | Cryptographically natural and supported by Verifiable Dropout, but expensive and potentially harmful without training |
| Random layer/head selection | Can change the executed subnetwork, but may let the attacker compute only the selected cheap subset or exploit a cached superset | Useful as a regulariser or diversity mechanism, not a standalone lower-bound construction |
| Challenge-keyed model weights or channel permutations | Appears to change the model, but a linear/invertible transformation may be pushed through cached matrices or undone outside the circuit | Treat as unsafe until an algebraic non-amortization argument is supplied |

The prefix/suffix distinction is important. In a causal model, appending a random vector after \(x\) does not alter the hidden states already computed for \(x\); prepending it does. Even a prefix is not automatically sufficient: the model may learn to ignore a small or out-of-distribution prefix, and an attacker may cache a distribution of nearby traces. The security experiment should therefore measure trace divergence and online work, rather than merely checking that the final token differs.

### Recommended first construction

The least invasive experiment is a VRF-derived random soft-prefix construction:

1. After the parent-derived beacon is fixed, expand \(R_i\) with a domain-separated PRG into \(m\) vectors of the model's embedding dimension. Sample them from a calibrated distribution (for example, an isotropic Gaussian whose scale is fitted to the public embedding table, as in the RSP line of work), or use a cryptographically generated bounded perturbation of public embedding vectors.
2. Prepend those vectors to the registered user prompt. Do not append them after the prompt if the goal is to invalidate the cached prompt trace.
3. Define the service semantics explicitly. The simplest relation is "the model answers the beacon-conditioned prompt"; if the original answer must be preserved, use an output-format/semantic-equivalence relation or aggregate several beacon-conditioned runs. That extra relation changes both the utility and the proof cost.
4. Include the beacon, generated prefix, and the complete transformed inference in the DeepProve public statement. The circuit must prove generation of the prefix from the VRF output; otherwise the prover can choose a benign prefix and recover determinism.
5. Measure the worst-case (not only average) output collision probability and the residual prover cost after caching. A high collision rate for "repeat after me" prompts is a direct falsification of the anti-reuse claim.

If quality or sensitivity is inadequate, the next construction to test is a VRF-keyed dropout/noise mask at every attention and feed-forward block. This follows the verifiable-dropout architecture more closely, but it almost certainly needs a model trained or calibrated for the stochastic forward pass. Q/K or attention-logit perturbations are preferable to V-only perturbations because they alter the nonlinear routing of information. A hybrid prefix plus sparse per-layer mask may give a better utility/security trade-off than either alone, but it increases the circuit and proof complexity.

### What the experiments must report for Section 5

For each candidate and prompt family, report at least:

- \(p_x=\Pr_{R,R'}[M(x,R)=M(x,R')]\), including exact token equality and a semantic-equivalence variant;
- layerwise divergence of hidden states, attention maps, and logits;
- the fraction of the trace and DeepProve prover phases that an optimal caching attacker can reuse (model evaluation, witness construction, commitments, polynomial/lookup data, and sumcheck/GKR rounds separately);
- fresh online time and memory after arbitrary pre-beacon preprocessing;
- utility: task accuracy, perplexity, format validity, and semantic similarity to the unperturbed answer;
- proof size, verifier time, and prover time attributable to proving the VRF-derived randomisation.

The quantity to insert into the reduction is the *worst-case online fraction* over the allowed prompt class. If \(C_{\rm fresh}(x)\) is the cost of one fresh instance and \(C_{\rm online}(x,\sigma,R)\) is the cost after pre-beacon state \(\sigma\), define

\[
\Pr_R\left[
 \frac{C_{\rm online}(x,\sigma,R)}{C_{\rm fresh}(x)}
 \geq 1-\varepsilon_p
 \right]\geq 1-\nu(\kappa)
\quad\text{for every allowed }(x,\sigma),
\]

or use the corresponding quantile/tail formulation if cost is random. The Section~5 theorem should assume this bound (or derive it from a proved anti-amortization property). A proof that the VRF mask is correctly sampled is not a proof of this inequality. In particular, if the prefix or internal mask can be ignored, linearly corrected, or represented by a cached superset, the measured \(\varepsilon_p\) will be large even though every proof is statement-bound.

### Research gap and likely paper contribution

The literature supports the components but not the desired end-to-end claim. Randomized smoothing and random soft prompts show how to perturb a frozen LLM at inference; stochastic-transformer work shows that all-layer randomness is materially different from last-layer randomness; Verifiable Dropout shows how to bind a nonce-derived mask to a zero-knowledge proof; and PoUW work makes anti-amortization an explicit computational property. None of these, by itself, gives a theorem that a DeepProve prover must redo a constant fraction of an LLM trace after a hidden VRF beacon.

That missing result is likely the cleanest formulation of our contribution: define a challenge-dependent transformer variant and a concrete attacker-with-preprocessing model, then either prove a lower bound for a restricted circuit/prover architecture or honestly report a measured worst-case bound. Until that is done, the defensible claim is only complete-proof freshness for the parent/beacon-bound statement, together with a separately stated empirical or assumed bound on reusable inference and prover work.

## Reassessment after an LLM-specific literature review

The earlier design-space table should not be read as saying that inference-time dropout, Q/K noise, or layer masking already have strong LLM evidence. Those entries are architectural hypotheses (and, in some cases, training-time regularizers), not established drop-in methods for a frozen autoregressive LLM. The literature that is directly relevant to the proposed PoML intervention is narrower. It is useful to separate four classes.

### 1. Random soft prefixes and continuous prompt vectors

The closest precedent is [Random Soft Prompts (RSP)](https://arxiv.org/abs/2605.11936). RSP samples a sequence of embedding vectors from a Gaussian fitted to the pretrained embedding table, appends the vectors at inference time, and uses one ordinary greedy forward pass. The paper reports near-baseline or improved accuracy in several aligned-model settings, and attributes the effect to early branching followed by dilution and later commitment. It also reports prefix, suffix, and infix placements. This is valuable evidence that an untrained random vector can be accepted by a frozen LLM without an obligatory denoising network. It remains a recent preprint and does not study adversaries, trace reuse, or a single challenge-conditioned PoML computation.

The placement used for RSP's main results is not automatically the right placement for PoML. In a causal decoder, a random suffix cannot alter hidden states already computed for the user prompt, so a miner can cache the entire prompt prefill. A random prefix is security-relevant because every later prompt and generated-token state is causally conditioned on it. The prefix may still be ignored or diluted, however; different prefixes can produce the same answer and possibly a cheaply correctable hidden trajectory. Thus RSP supports the *interface* and gives a utility starting point, but not an anti-amortization theorem.

[Prefix-Tuning](https://arxiv.org/abs/2101.00190) and [Prompt Tuning](https://arxiv.org/abs/2104.08691) are older, peer-reviewed precedents for prepending continuous vectors to a frozen language model. Their vectors are learned rather than random, so they do not establish freshness. They do establish that ordinary transformer implementations can consume a variable-length continuous prefix and that a prefix can condition generation without modifying the model weights. In an implementation that accepts an inputs_embeds interface, a VRF-derived prefix is therefore a modest interface change; in a token-ID-only circuit, the embedding input relation must be extended.

[Soft Reasoning](https://arxiv.org/abs/2505.24688) is another relevant, now peer-reviewed, direction. It explores Gaussian perturbations of an embedding associated with the answer-generation boundary and uses the resulting trajectories for reasoning search. The paper finds that perturbing more than a small number of positions can damage quality, and that a position near the end of the prompt often preserves semantics better than an early position. That utility result points in the opposite direction from the strongest freshness requirement: an end-position perturbation is less disruptive, but it leaves more of the prompt prefill reusable. We should treat this as a utility/security trade-off to measure, not as a reason to choose the end position by default.

**PoML assessment.** A challenge-derived continuous prefix is the best drop-in candidate. It needs no model retraining if the model and DeepProve circuit already support continuous input embeddings. It is also easy to bind: the public statement contains the beacon and query, the circuit derives the prefix with a domain-separated PRG, and the proven trace starts with the resulting vectors. The main open questions are (i) how long and how large the prefix must be before it materially changes the trace, and (ii) how much accuracy is lost on short, high-margin prompts such as "Repeat after me: The sky is blue."

### 2. Randomized embedding smoothing and perturbation

[RESTA](https://arxiv.org/abs/2501.16497) is the closest match to the proposed embedding-randomization idea. It considers isotropic Gaussian, hard-directional, soft-directional, and orthogonal perturbations. Its implementation perturbs user-content embeddings while keeping system/template content fixed, generates \(k\) responses, and aggregates tokens by majority vote (with smoothing applied to an initial response prefix of length \(l\)). The reported experiments use \(k=10\) and \(l=20\). This is a robustness/jailbreak defense, not a lower-bound construction, and the paper does not report a \(k=1\) utility ablation.

The more recent [Embedding Perturbation for LLM uncertainty quantification](https://arxiv.org/html/2602.02427) supplies useful mechanistic evidence. It adds small i.i.d. Gaussian noise to token embeddings and observes that token probabilities, especially at uncertain reasoning steps, are sensitive to the perturbation. The authors relate the sensitivity to distance from a token-selection decision boundary. This supports the claim that small embedding noise can alter internal decisions, but it does not show that one noisy generation remains accurate, nor that a cached trace cannot be updated cheaply.

[Discrete-Continuous Randomized Smoothing for generative LLMs](https://openreview.net/pdf/1703cc89b7cea69d54953f0a355521f86fd62365.pdf) samples a subset of token positions, adds Gaussian noise in embedding space, and projects each perturbed vector to the nearest vocabulary embedding before running a standard generative LLM. It is a useful no-retraining implementation precedent, but its certification experiments use thousands of randomized generations and the nearest-neighbour projection creates large collision regions: many different noise values can map to the same token. Projection is therefore unattractive as the main anti-precomputation mechanism.

**PoML assessment.** RESTA-style perturbation is compatible with a one-pass protocol: choose one beacon-derived perturbation, run one perturbed generation, and prove that computation. Majority voting is not required for cryptographic correctness; it is a utility/robustness device. If \(k>1\) is retained, the protocol must prove all \(k\) traces and the aggregation relation (or prove a recursive aggregate), multiplying prover work and creating a different service semantics. The honest statement about \(k=1\) is currently "unknown but testable": for high-margin prompts it may preserve the answer, while prompts close to a token decision boundary may change substantially. RESTA's \(k=10\) results cannot be extrapolated to a one-sample guarantee.

For anti-amortization, perturbing only user tokens is better than perturbing a response suffix, but keeping a static system/template prefix unchanged still permits that portion to be cached. A dense perturbation of all registered-query tokens, or a random prefix before the query, is the more relevant PoML variant. A nearest-token projection should be avoided unless the protocol specifically wants discrete semantics. Continuous vectors give a much larger challenge space and do not introduce an obvious finite lookup table.

### 3. Randomized discrete or semantic prompt transformations

Work such as [SmoothLLM](https://arxiv.org/abs/2310.03684), [SAFER](https://aclanthology.org/2020.acl-main.317/), [Semantic Smoothing](https://arxiv.org/abs/2402.16192), and self-denoising smoothing applies random character, token, mask, or semantic transformations to several copies of a prompt and aggregates the outputs. These papers provide evidence that LLM behavior can be stabilized under randomized input transformations, but they generally require multiple model calls, may change the task semantics, and often use a classifier or an auxiliary LLM to judge/aggregate responses. They are not attractive as the primary PoML construction. They may still be useful as baselines for measuring the utility loss of a single random transformation.

### 4. Randomized internal weights or activations

[PaRaFormer](https://arxiv.org/abs/2311.10943) inserts randomly initialized, frozen Q/K/V and feed-forward layers into a transformer and trains the remaining model around them. It reports more diverse dialogue with comparable fluency and coherence. This is the strongest literature precedent for randomness inside the transformer rather than only at its input, but it is a redesigned and retrained architecture. Reinitializing or changing such layers per beacon in an already-trained LLM is not a drop-in operation: the model has not learned to interpret those distributions, and the DeepProve circuit would need a new parameterization. PaRaFormer therefore supports a *future trained variant* if the prefix/embedding route fails, not the first PoML prototype.

Inference-time residual-stream steering papers establish that frozen LLMs can be modified by adding vectors at selected layers, but the vectors are normally learned semantic directions rather than fresh random challenges. They are useful engineering precedents for exposing internal injection points, not evidence that random residual noise preserves generation quality or forces fresh prover work. The same caution applies to Q/K noise and all-layer dropout: these remain experiments until tested on the target model and proved to have a nontrivial online-cost effect.

### Does one RESTA-style sample preserve useful output?

There are three different questions that must not be conflated:

1. **Does one sample produce a syntactically valid response?** Usually this is the easiest property and should be checked first.
2. **Does it preserve the intended answer?** This depends on the perturbation scale, the prompt, and the token margins. No paper located here gives a general \(k=1\) guarantee for autoregressive LLMs.
3. **Does it create a sufficiently different trace?** Output disagreement is neither necessary nor sufficient. Two runs can produce different answers while sharing a large prefill trace, or the same answer while having different hidden states and requiring fresh computation.

For PoML, define the one-sample stability and collision statistics explicitly:

\[
p_{\mathrm{exact}}(x)=\Pr_{R,R'}[M_\theta(T_R(x))=M_\theta(T_{R'}(x))],
\qquad
p_{\mathrm{sem}}(x)=\Pr_{R,R'}[\operatorname{Sem}(M_\theta(T_R(x)))=\operatorname{Sem}(M_\theta(T_{R'}(x)))].
\]

Measure these on the high-margin "repeat after me" family as well as on ordinary benchmark prompts. A one-sample protocol is reasonable if \(p_{\mathrm{sem}}\) remains high for the intended service semantics while the trace-level reuse fraction remains low. If exact canonical answers are required, a single noisy sample may be unacceptable; voting then becomes part of the application definition and its additional proof cost must be acknowledged.

The RSP mechanism gives a warning: the random prefix/suffix influence is reported to be strongest early and to dilute as the KV cache grows. This means a model can converge to the same late answer even when the initial random vectors differ. That convergence is not a failure for freshness if the hidden-state computation is genuinely recomputed, but it is a failure if the implementation lets the miner reuse a base trace and cheaply "wash out" the perturbation. We need to measure both.

### A distinct transformed input is not a computational-independence proof

Guaranteeing that \(T_R(x)\) is different from every previously seen byte string is insufficient. A neural network is not injective: distinct embedding sequences can produce the same token sequence, the same late hidden states, or states that are close enough for a cached approximation to be corrected cheaply. In particular:

- a suffix perturbation leaves the causal prefix exactly unchanged;
- a small prefix/embedding perturbation can be attenuated by attention, residual mixing, and layer normalization;
- a discrete nearest-neighbour projection can map many random vectors to one vocabulary token;
- a high-margin prompt may map a large region of challenge space to the same greedy output;
- even when outputs coincide, the attacker may reuse the model weights, token projections, static system prefix, or a cached base trace and recompute only the challenge-dependent deltas.

The relevant security property is therefore a trace/cost property, not an input-uniqueness property. Let \(\sigma\) be arbitrary state computed before the beacon and let \(C_{\mathrm{online}}(x,\sigma,R)\) be the minimum online work of an optimal caching attacker. Section 5 should quantify the worst-case tail

\[
\Pr_R\!\left[
\frac{C_{\mathrm{online}}(x,\sigma,R)}{C_{\mathrm{fresh}}(x)}
\ge 1-\varepsilon_p
\right]\ge 1-\nu(\kappa),
\]

or explicitly leave \(\varepsilon_p\) as a measured assumption. The experiment should compare not just final outputs but layerwise hidden states, attention maps, logits, witness construction, commitments, and the GKR/sumcheck phases. "Different input" can be recorded as a diagnostic; it cannot substitute for this bound.

### Recommended PoML order after this review

1. **VRF-derived random prefix, continuous and dense.** Use RSP's training-free distribution as the initial calibration, but prepend the vectors before the registered query. Test several prefix lengths and norms, and include the prefix-generation relation in DeepProve.
2. **Dense RESTA-style perturbation of the entire registered query.** Start with isotropic and orthogonal noise, one sample, no vote. Keep the system template fixed only if its cacheability is explicitly excluded from the claimed fresh-work fraction.
3. **A small perturbation at the answer boundary (Soft Reasoning-inspired).** This is a utility-oriented fallback when a full prefix damages quality; it is weaker for prefill freshness and should not be presented as solving anti-amortization by itself.
4. **A trained all-layer stochastic variant (PaRaFormer-like or challenge-keyed residual noise).** Pursue only if the first two fail the measured trace-divergence/utility frontier. It offers stronger potential freshness but changes the model, training, and proof circuit.
5. **Discrete/semantic smoothing and majority voting.** Treat as a robustness baseline or an optional service mode, not the core one-pass PoML mechanism.

For every candidate, the protocol should derive the randomization only after the parent-bound beacon is fixed, prove the derivation and application inside the model relation, and report the worst-case online fraction for specialized prompts. The literature supports using random embeddings as a plausible intervention; it does not support claiming that one random sample, a unique transformed input, or a correctly proved VRF mask automatically gives the Section~5 computational-independence bound.
## RSP-specific literature chase: what is established, and what is still a new claim

The reference trail gives a more precise answer to the question of whether RSP is an established practice. Two claims that are often conflated should be separated:

1. **Continuous vectors can be supplied to a frozen language model as pseudo-tokens.** This is established. Prefix-Tuning, Prompt Tuning, and the NAACL best short paper *Learning How to Ask* all use continuous "soft words" or prefixes. The latter is especially relevant because it initializes soft prompts randomly, but the vectors are then optimized; it does not show that a fresh random vector is useful at inference without optimization. See [Prefix-Tuning](https://arxiv.org/abs/2101.00190), [Prompt Tuning](https://arxiv.org/abs/2104.08691), and [Qin and Eisner (NAACL 2021)](https://aclanthology.org/2021.naacl-main.410/).

2. **A newly sampled, untrained random prefix is itself a useful inference-time intervention.** I did not find an older peer-reviewed LLM paper establishing this exact claim. RSP appears to be a recent and genuinely novel isolation of that mechanism. It should therefore be cited as promising preliminary evidence, not as an established engineering primitive.

The most useful adjacent literature is as follows.

| Work | What it establishes | PoML limitation or opportunity |
|---|---|---|
| [Prefix-Tuning (ACL 2021)](https://arxiv.org/abs/2101.00190), [Prompt Tuning (EMNLP 2021)](https://arxiv.org/abs/2104.08691) | A frozen transformer can condition all subsequent computation on a continuous prefix. | The vectors are learned and reusable, so there is no freshness or anti-amortization result. They nevertheless support the feasibility of an `inputs_embeds`-style DeepProve interface. |
| [Learning How to Ask (NAACL 2021)](https://aclanthology.org/2021.naacl-main.410/) | Random initialization of soft prompts is a standard starting point for optimization; random and informed initialization can be similarly good after training. | This is evidence about optimization, not about one-shot random inference. It warns us not to cite random initialization as evidence that arbitrary vectors will be semantically understood. |
| [Pause Tokens (ICLR 2024)](https://arxiv.org/abs/2310.02226) and [Let's Think Dot by Dot](https://arxiv.org/abs/2404.15758) | Extra positions can provide computation even when their surface content is meaningless. | The model must be trained to use filler/pause positions; arbitrary filler is not automatically useful. A finite discrete filler alphabet is also easy to precompute and replay. |
| [NEFTune (ICLR 2024)](https://arxiv.org/abs/2310.05914) | Gaussian noise on embeddings can be tolerated and can improve a model when used during instruction fine-tuning. | The noise is training-only; the authors explicitly do not add it during ordinary generation. It supports a future noise-aware model, not a drop-in security guarantee. |
| [StreamingLLM (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html) and [Massive Activations (COLM 2024)](https://arxiv.org/abs/2402.17762) | Initial or special tokens can attract attention for positional or bias reasons even when semantically empty; some large activations are nearly input-independent. | A random prefix may receive attention without carrying challenge-specific information. A position-only or bias-only effect can be cached, so attention mass is not a freshness proof. |
| [Meaningless is better](https://arxiv.org/abs/2411.17304) and [Meaningless Tokens, Meaningful Gains](https://arxiv.org/abs/2510.01032) | Semantically empty discrete strings can change reasoning and first-layer activations. The latter reports that insertion between the system prompt and question was the best of its tested positions, while insertion at the end could make accuracy collapse. | Both are recent/preprint evidence, not a general theorem. They support testing a challenge block between system and query, but discrete strings have a finite replay space and the observed first-layer effect may be cheap to correct. |
| [Smoothed Embeddings for Robust Language Models (RESTA)](https://arxiv.org/abs/2501.16497) | Inference-time Gaussian embedding perturbations can improve robustness when several perturbed generations are aggregated. | The reported protocol uses multiple samples and majority voting. It does not establish that a single sample preserves the answer or forces a large amount of new computation. |
| [From Noise to Diversity (RSP)](https://arxiv.org/html/2605.11936) | Fresh Gaussian pseudo-tokens can change early token distributions and improve Pass@N in several math-reasoning settings without training the prompt. | It is a current preprint, is evaluated mainly on reasoning benchmarks, and does not measure an adversarial cached-trace cost. Its default suffix placement is also not security-relevant for causal prefill freshness. |

### What RSP actually tells us about output behavior

RSP samples each pseudo-token from an isotropic Gaussian calibrated to the mean and variance of the model's embedding table. Its experiments use short blocks (principally 10--20 vectors), one fresh block per rollout, and report a two-stage effect: early next-token distributions become flatter and trajectories branch; the effect attenuates as the KV cache grows and later tokens commit. This is useful evidence that a random block can alter generation without conveying a task instruction, but it is also a warning for PoML: convergence to the same late answer is expected and does not imply that the earlier expensive states were non-reusable.

Position is not a cosmetic choice. RSP's suffix is the utility-oriented default because it often preserves the original prompt behavior, but a causal decoder has already computed the user-query prefill before a suffix is seen. In the reported Llama3.1 ablation, a prefix substantially reduced some benchmark scores (approximately 7 percentage points on MATH and 9 points on GSM), whereas suffix results were closer to baseline; other Qwen models were less sensitive. Thus there is no literature-supported assumption that a prefix is harmless, nor that a suffix gives freshness. The proposed protocol must measure the target model and target prompt family directly.

The independent *Meaningless Tokens, Meaningful Gains* study is valuable because it tests placement explicitly. Its strongest utility result came from inserting tokens between the system instruction and the question; insertion at the beginning, at the end, or at a random internal position could be substantially worse. This suggests a practical PoML experiment: compare a VRF-derived continuous block immediately before the registered query with a block between the system template and query. The latter may cause the query prefill to depend on the challenge while preserving more instruction semantics. It remains an empirical hypothesis, not a security result.

### Which parts of this literature are genuinely reusable for PoML?

The literature supports the following engineering statements:

- A continuous challenge block is a normal input type for transformer implementations, provided the model relation exposes the embedding input rather than only token IDs.
- Random initialization and embedding noise are operationally straightforward. They do not require a vocabulary projection, and a cryptographic PRG can expand the beacon into a high-dimensional block inside the circuit.
- A challenge block placed before the user query can influence every later causal state. A suffix cannot influence the already-computed query prefill.
- Utility is controlled by block length, norm, position, and model family. Larger or badly placed blocks can cause repetition, degraded accuracy, or a changed response style.
- Output agreement is not the right security metric. RSP's own mechanism predicts that different early trajectories may eventually yield the same answer.

The literature does **not** support the following stronger claims:

- that one RSP/RESTA sample has a general answer-preservation guarantee;
- that a random embedding is necessarily interpreted as a rich, challenge-dependent semantic signal;
- that a different byte string or a different final answer prevents reuse of a cached trace; or
- that any fixed percentage of the model computation must be redone.

The last point is the key one for Section~5. Existing papers measure accuracy, entropy, diversity, or robustness. They do not model an adversary who caches a base trace and computes only the challenge-dependent correction. In fact, the RSP and meaningless-token evidence is compatible with a potentially cheap attack: if most of the intervention is absorbed in an early attention normalization or an approximately affine first-layer shift, the attacker could precompute the unperturbed model and update only a small portion of the trace. This is a PoML risk inference, not an attack demonstrated by those papers, and it is exactly why a measured anti-amortization experiment is required.

### Recommended RSP-centered experiment before committing to the construction

Use the actual DeepProve target model and include specialized high-margin prompts (for example, exact-copy or deterministic-format prompts) as a separate stratum. For each prompt, compare:

1. no intervention;
2. an RSP-style Gaussian block before the query;
3. the same block between system prompt and query;
4. an RSP-style suffix (utility control, not a security candidate);
5. dense Gaussian perturbation of all registered-query embeddings (RESTA-style); and
6. a discrete meaningless-token control.

Sweep the block length and RMS norm rather than importing RSP's 10--20-token setting unchanged. For every condition, record exact and semantic answer stability, token-level logit changes, layerwise hidden-state distance, attention mass on the challenge block, and the first layer at which the two traces become distinguishable. Then implement the caching adversary explicitly: cache all beacon-independent weights, system-prefix states, and the unperturbed trace, and allow it to apply the cheapest measured correction. Report the online FLOPs (or prover constraints) as a fraction of a fresh run, including the cost of the DeepProve witness and commitment phases.

The cryptographic implementation should derive the block only after the parent-bound beacon is fixed, domain-separate it from the VRF used for output sampling, and include both the derivation and the resulting continuous vectors in the proved statement. Quantize only in a way that is part of the relation; otherwise the attacker may exploit a small effective challenge space. Avoid nearest-vocabulary projection in the first construction because many continuous challenges would then collapse to the same discrete sequence.

The most defensible present conclusion is therefore: **RSP supplies a plausible, training-free candidate and a useful calibration distribution, while Prefix-Tuning/Prompt-Tuning establish the continuous-prefix interface. Neither establishes the Section~5 lower bound.** We should retain the RSP construction only together with an explicit measured parameter \\(\varepsilon_p\\) for the optimal cached-trace attacker, and state clearly that proving or empirically validating a nontrivial lower bound is part of the PoML contribution rather than something inherited from the RSP literature.
## Diffusion-style guarantee for autoregressive LLMs and the KV-cache issue

The diffusion argument in Appendix~\ref{app:input-divergence} has two logically separate parts. First, fresh Gaussian noise is injected at every call to the denoising network, which gives a direct anti-concentration bound on the input to that call. Second, an explicit assumption is needed to move from distinct inputs to a lower bound on non-reusable internal computation. The first part can be transferred to an autoregressive transformer only if we define the incremental model input correctly.

Let $C_t(r)$ denote the complete layerwise KV cache after the prompt and the first $t$ generated tokens have been processed. An incremental decoder call has the form

\[
(\ell_{t+1}(r), C_{t+1}(r))
  = F_\theta\bigl(\widetilde e_t(r), C_t(r)\bigr),
\qquad
y_{t+1}(r)=\operatorname{Decode}(\ell_{t+1}(r)),
\]

where $\widetilde e_t(r)$ is the embedding supplied for the newest token. The model input at generation step $t+1$ is therefore not just the concatenated token string. In an implementation using KV caching it is the pair $(\widetilde e_t(r),C_t(r))$.

This distinction is also reflected in standard autoregressive cost models: prompt encoding builds the cache once, while each later token requires a forward pass for the new token and attention against the stored keys and values; see [Autoregressive Inference of Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/d1a14493e5f84d6c6129414f0cd1a7c6-Paper-Conference.pdf).

### A one-time noisy prompt gives only a cache-persistence argument

Suppose the prompt is perturbed once, either by a random soft prefix or by additive Gaussian noise on its embeddings. In the abstract full-sequence definition, the input remains different at every later generation step because the initial perturbed embeddings are still part of the sequence. In a cached implementation, however, those embeddings are represented only through $C_t(r)$. The step-level claim now requires

\[
\Pr\!\left[\|C_t(r)-C_t(r')\|_\infty < \delta_q/2\right]
\]

to be small. Initial input noise does not by itself prove this. Attention, normalization, quantization, or cancellation can attenuate the perturbation, and a cache can contain many entries whose values are unaffected or whose effect on the next output is negligible.

It is therefore valid to say that a one-time perturbation makes every *raw concatenated sequence* different, even when the generated tokens are identical. It is not valid to replace this with an unqualified claim that every optimized incremental call is different. The latter is true only after defining the call input as $(\widetilde e_t,C_t)$ and establishing that the cache representation retains a detectable challenge-dependent component. For a random pseudo-token, the first-layer $K/V$ projection will generally differ almost surely in real arithmetic when the projection is nonconstant, but a quantized circuit needs a rank- and scale-dependent anti-concentration bound. Layer normalization and low-dimensional grouped-query projections can reduce the effective dimension of that bound.

There is a useful middle ground. A random prefix token's first-layer $K/V$ entries are computed during prefill and then persist unchanged in every later cache $C_t$. If those entries are separated by more than $\delta_q/2$, the *complete cache tensor* is separated at every generation step, even if the influence on later query states is small. This is enough to reproduce the diffusion paper's two-stage framing: prove persistent cache-input divergence first, then state and test an LLM-specific analogue of Assumption~\ref{def:ftheta-assumption} for non-reuse inside the transformer. It is not enough to claim that a large fraction of the cache or computation is challenge-dependent; the attacker may still reuse all clean-token entries and recompute only the random block.

A random prefix is consequently weaker than the diffusion construction. With a clean prefix and clean generated token embeddings, the first-layer query/key/value projections of an unchanged token can often be reused because they depend on that token's embedding and position before the token has attended to the random prefix. The first-layer attention output, later-layer states, and later-layer KV entries may depend on the prefix, but that propagation must be measured or assumed. A suffix is weaker still: all prompt states before the suffix, including the user-query prefill, are exactly unchanged under the causal mask.

Dense noise on every registered prompt embedding is stronger for the prefill. It makes the first-layer projections of those prompt tokens challenge-dependent (subject to the rank and normalization of the projection), and the resulting cache is carried into every later step. Nevertheless, clean generated tokens can still have reusable first-layer projections when the generated token is the same. More generally, a cache can be compressed or partially evicted while preserving nearly the same output; [CriticalKV](https://arxiv.org/abs/2502.03805) formalizes this output-perturbation perspective and shows that attention weight alone does not determine which cache entries matter. Thus cache inequality is not equivalent to a full fresh-work lower bound.

### The closest exact analogue: fresh noise on every autoregressive input

To reproduce the structure of the diffusion proof, add an independent challenge-derived noise vector at every incremental call. For example, after token $y_t(r)$ has been selected, supply

\[
\widetilde e_t(r) = E(y_t(r)) + \sigma_t z_t(r),
\qquad z_t(r)\sim\mathcal N(0,I_{d_e}),
\]

to the next transformer call, and use a separately derived noisy prompt embedding block for the prefill. The generated token may be the same or different in the two executions; conditional on both histories, the deterministic embedding difference is just a shift. For two independent seeds,

\[
\widetilde e_t(r)-\widetilde e_t(r')
 = a_t + \sigma_t\bigl(z_t(r)-z_t(r')\bigr),
\]

for a history-dependent but conditionally fixed vector $a_t$. The same Gaussian anti-concentration argument as in Appendix~\ref{app:input-divergence} then gives

\[
\Pr\!\left[
  \|\widetilde e_t(r)-\widetilde e_t(r')\|_\infty < \delta_q/2
  \mid\text{histories}
\right]
 \leq
 \left(\frac{\delta_q}{2\sigma_t\sqrt{\pi}}\right)^{d_e}.
\]

With a fixed maximum output length $T$, a union bound gives

\[
\Pr\!\left[\exists t\in\{0,\ldots,T\}:
  \|\widetilde e_t(r)-\widetilde e_t(r')\|_\infty < \delta_q/2\right]
 \leq
 (T+1)\left(\frac{\delta_q}{2\sigma_{\min}\sqrt{\pi}}\right)^{d_e},
\]

up to the separate prompt-block dimension factor. This is the direct LLM analogue of the diffusion theorem: the *input embedding to each model call* is different with overwhelming probability, even if the decoded token is identical.

The conditioning step requires a schedule assumption. The blocks $z_t$ must be independently sampled (or computationally indistinguishable from independent samples) conditional on the earlier history. If all per-step vectors are exposed as correlated slices of one small challenge, observing earlier logits or tokens can reveal information about later slices and the statistical anti-concentration proof no longer follows verbatim. In PoML, the natural implementation is to domain-separate a PRG/VRF expansion by the query identifier, step index, and role (prompt noise versus generated-token noise), and then state the result as a computational hybrid from independent discretized-Gaussian blocks.

The sampling coins must be separated as well. If the same VRF output both perturbs the next token embedding and selects the current token, then conditioning on the observed token can leak information about the perturbation and invalidate the simple shifted-Gaussian argument. Use independent domain-separated streams for (i) output-token sampling and (ii) the noise that is applied only after that token has been selected for the next transformer call.

This theorem does not by itself prove that the whole transformer computation is non-reusable. It supports the same two-stage structure already used for diffusion:

1. prove step-level input divergence from fresh noise; and
2. assume or empirically establish an LLM analogue of Assumption~\ref{def:ftheta-assumption}, now for $F_\theta(\widetilde e_t,C_t)$, stating that a $\delta_q$-separated incremental input forces a non-negligible fraction of layer activations and cache entries to be recomputed.

Fresh noise on the newest token also prevents the easiest exact-cache shortcut: even if the previous tokens and their outputs coincide, the first-layer key/value projections of the new token are different. A dense random vector requires a dense projection such as $W_K z_t$ and $W_V z_t$; precomputing $W_K E(y_t)$ does not precompute these challenge-dependent terms. This is a computational plausibility argument, not a lower-bound proof, because an attacker may still seek low-rank, approximate, or output-insensitive corrections.

### The cost and semantics trade-off

The diffusion-equivalent construction is considerably more invasive than RSP. It perturbs the representation of every generated token before that token is fed back into the model, so the perturbation recursively changes future logits. Existing embedding-noise literature does not establish that one such noisy autoregressive rollout preserves a desired answer. Training with noise, as in [NEFTune](https://arxiv.org/abs/2310.05914), is evidence that a model can be made tolerant, but it is not evidence for an off-the-shelf model at inference. RESTA-style work likewise relies on multiple perturbed rollouts and aggregation.

There is therefore a three-way trade-off:

- **One-time random prefix or prompt noise:** best semantic compatibility and closest to current RSP/RESTA, but only a cache-persistence argument; no fresh randomness at later calls.
- **Fresh noise on each new token embedding:** closest to the diffusion theorem and strongest protection against exact KV reuse, but requires testing or training for stability and changes the generation process.
- **Fresh noise directly in every layer's K/V cache entry:** strongest cache-level statement by construction, but it is a new stochastic transformer architecture with no established LLM utility evidence and a substantially larger DeepProve relation.

For the current PoML framework, the most defensible progression is to prototype the second option experimentally while retaining the first as the utility baseline. If per-token noise is unacceptable, the paper should not claim the diffusion theorem transfers unchanged; it should state a conditional cache-divergence assumption and measure the optimal cached-trace cost.

### Quantization caveat

The diffusion appendix writes the argument for real-valued Gaussian variables and then chooses $\delta_q$ according to the ZKP quantization resolution. The LLM construction should make this explicit as well. In an actual finite-field or fixed-point circuit, the noise is a discretized Gaussian or a PRG-derived finite distribution. The relevant bound is then the maximum probability mass of a quantization bin (or the collision probability of two independent challenge blocks), not literal equality of real Gaussian values. The noise scale must be large relative to the quantization step while remaining within the model's empirically acceptable perturbation range.

For a cleaner cache-divergence bound, it may be preferable to inject the challenge after the model's input RMSNorm/LayerNorm and before the first $Q/K/V$ projections, or to include the normalization in the proved randomization map and bound its output distribution directly. Noise added before normalization is still a valid experiment, but its effective covariance is nonlinear and may lose radial information; the diffusion anti-concentration calculation cannot simply be copied with the raw embedding variance.

The recommended Section~5 wording is consequently conditional and precise: a per-step noisy embedding schedule can establish that every incremental transformer call receives a different input with high probability; a separate model-specific assumption or experiment is still required to convert that fact into a percentage of non-reusable inference or ZKP computation. This is the same logical division already present in the diffusion analysis, with KV-cache divergence replacing the simpler explicit state recurrence.

## Evidence for embedding perturbation versus recursive per-token noise

The literature supports embedding perturbation as a legitimate operation on language models, but it supports several importantly different claims. It supports (i) adding noise to an input or a first generated-token embedding for robustness, exploration, or privacy; (ii) training a model to be locally smooth or noise-tolerant; and (iii) measuring the sensitivity of each next-token probability to perturbations of its preceding embeddings. It does **not** yet establish that an ordinary, off-the-shelf decoder-only LLM can be run with an independent Gaussian perturbation added recursively to every generated token while preserving answer quality. That last construction should therefore be presented as a new PoML mechanism, not as an established LLM practice.

### What the closest papers actually do

| Work | Perturbation used | What it establishes for PoML | What it does not establish |
|---|---|---|---|
| [R3F](https://arxiv.org/abs/2008.03156) | Gaussian or uniform noise is added to representations during fine-tuning, with a consistency objective. | Noise injection is an established way to control representation drift; experiments include generation and summarisation tasks. | It is training-time regularisation, not noisy inference on a frozen model. |
| [NEFTune](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4bdeeaeb380b35302bbda1823d328c22-Abstract-Conference.html) | Gaussian noise is added to input embeddings during instruction fine-tuning. | A model can be trained to tolerate embedding noise; the reported AlpacaEval score for LLaMA-2-7B rose from 29.79\% to 64.69\% with noisy-embedding fine-tuning. | The authors explicitly do not add noise at generation time; this is not evidence that an unmodified model tolerates per-token perturbations. |
| [Soft Reasoning](https://proceedings.mlr.press/v267/zhu25ae.html), ICML 2025 | Gaussian exploration is applied to the embedding of the first generated token; the subsequent rollout is greedy and deterministic for that candidate. | This is peer-reviewed evidence that a small latent perturbation can deliberately redirect an LLM generation trajectory while retaining useful reasoning quality. | It is a one-time perturbation and uses search/selection over several candidate rollouts, not one recursively noisy rollout. |
| [RESTA](https://arxiv.org/html/2501.16497) | Independent Gaussian/directional/orthogonal noise is applied to user-content embeddings; tentative next tokens are generated in $k$ branches and majority-voted. | This is the closest security-oriented inference precedent. With $k=10$ and prefix length $l=20$, it reports large jailbreak-success reductions while retaining substantially more utility than character perturbation. Isotropic/orthogonal noise used scales around $\sigma=0.01$--$0.04$; directional variants require different scales. | Generated token embeddings are explicitly **not** perturbed. After the first $l$ tokens, the method returns to a single clean rollout. Thus it cannot support a claim about recursive per-token noise. |
| [Embedding Perturbation may Better Reflect Intermediate-Step Uncertainty](https://arxiv.org/html/2602.02427) | For a fixed, already-generated response, i.i.d. Gaussian noise is added to all prompt and response embeddings and the change in each next-token probability is measured. | This is direct evidence that preceding-token embedding perturbations reveal token-level instability. The paper uses $\sigma=0.001$ and $l=20$ perturbation samples; random perturbation gives higher error-step detection than NLL/entropy baselines on MATH and BBH. | It is teacher-forced sensitivity analysis: the response is held fixed while probabilities are recomputed. It is not free-running decoding in which a perturbed token is fed back and changes all later tokens. |
| [Noiser](https://arxiv.org/abs/2504.02911) | Bounded noise is added to each input embedding and a binary search finds the largest scale that preserves the original next-token prediction. | It gives a practical model-specific calibration procedure and shows bounded embedding perturbations can be used on six LLMs and three generation tasks. | It optimises an attribution diagnostic, not a PoML protocol; its scale is selected per model/input and is not a universal constant. |
| [How Stable is the Next Token?](https://iclr.cc/virtual/2026/poster/10008781), ICLR 2026 | Defines a local hidden-state radius, the Token Constraint Bound $\delta_{\mathrm{TCB}}$, before the next-token distribution changes by a chosen tolerance. | It supplies a principled way to measure the local safety margin of a candidate noise level rather than assuming that probability or perplexity is a sufficient stability metric. | It is a local first-order bound on next-token output, not a sequence-level guarantee or a cache-recomputation theorem. |
| [Are Reasoning LLMs Robust to Interventions on their Chain-of-Thought?](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html), ICLR 2026 | Perturbs the generated chain of thought at controlled timesteps using several textual interventions. | It is a useful warning about autoregressive compounding: robustness is worse for early interventions, and neutral/adversarial interventions increased chain length by more than 200\% in some settings. | The interventions are textual, not Gaussian embedding noise, so the exact numbers cannot be transferred to PoML. |

The strongest positive evidence is therefore narrower than “LLMs naturally support noisy decoding.” Soft Reasoning shows that a single embedding-space nudge can guide a useful generation, RESTA shows that input-embedding smoothing can be deployed as a safety defence, and R3F/NEFTune show how to train models to be tolerant of such perturbations. The strongest evidence about *per-token* effects is the 2026 perturbation-UQ paper, but its use of all response embeddings is a diagnostic under teacher forcing. Its implementation deliberately reuses one sampled perturbation over the fixed response rather than recursively changing the response. This distinction must be stated explicitly in the paper.

A related [ICLR 2026 study of noise stability in Transformer models](https://iclr.cc/virtual/2026/poster/10009111) analyses correlated noise on all input coordinates and trains a noise-stability regulariser on next-token-prediction tasks. It reports faster training, not a runtime noisy-decoding guarantee. It is useful support for treating stability to representation noise as a measurable model property, but it reinforces the need to distinguish training-time robustness from the PoML inference mechanism.

### Why the effect is token- and model-dependent

There is no model-independent percentage by which a given embedding noise level changes the output. Under a local linearisation, let $m_{t,b}$ be the logit margin between the selected token and a rival token $b$, and let $g_{t,b}$ be the gradient of that margin with respect to the preceding embeddings. For a Gaussian perturbation of scale $\sigma$,

\[
\Delta m_{t,b}\approx \sigma g_{t,b}^{\mathsf T}z,
\qquad
\Pr[\text{rival }b\text{ overtakes}]
\approx
\Phi\!\left(-\frac{m_{t,b}}{\sigma\lVert g_{t,b}\rVert_2}\right).
\]

This is the same phenomenon measured by the perturbation-UQ paper: sensitivity is approximately $\sigma^2\lVert\nabla_H\log P(x_t\mid H)\rVert_2^2$ and is largest near a decision boundary. A high-margin instruction such as “Repeat after me: The sky is blue” will usually be less affected than a low-margin reasoning transition, but “usually” is not a protocol guarantee. Formatting tokens, punctuation, stop tokens, and the first token after an instruction can have small margins even when the semantic task is trivial.

For recursive noise, the relevant state difference is not just the current embedding perturbation. If $h_t$ denotes the hidden/cache state, a local recurrence has the form

\[
\Delta h_{t+1}\approx A_t\Delta h_t+B_t\sigma_t z_t.
\]

The matrices $A_t$ can contract or amplify differences. Transformers are not known to be globally contractive, so a bound such as “the output changes by at most $c\sigma$” or “only $p$ percent of tokens change” cannot be inferred from the one-step perturbation norm. Once one token flips, all later conditional distributions are evaluated on a different history; sequence divergence can then cascade. Conversely, if the model remains in a high-margin basin, many independent perturbations may leave the discrete output unchanged. This explains why a one-time perturbation can preserve semantics while a per-token rollout can eventually become much less stable even at the same per-token scale.

The right empirical quantity is consequently a curve, not a single noise number:

\[
\sigma\longmapsto
\bigl(\text{token-flip rate},\text{exact-sequence agreement},
\text{semantic/task accuracy},\text{format/stop-token failure},
\text{cache divergence},\text{attacker-reusable work}\bigr).
\]

The $\sigma=0.001$ value in the perturbation-UQ paper is a useful initial scale for a *sensitivity probe*, not a universal PoML setting. RESTA demonstrates that useful security/utility operating points can be much larger, but its values are on Vicuna-13B/Llama-2-7B, use multiple branches, and perturb only the input prefix. Scales must therefore be normalised to the deployed model (for example by the RMS of the post-embedding representation and the circuit's quantisation step) and selected on a held-out calibration set.

### Implications for the KV cache and precomputation

For a one-time dense prompt perturbation, the prefill $K/V$ entries for the perturbed positions are challenge-dependent and persist in later caches. An old clean cache cannot be reused *exactly* for those positions unless the perturbation is erased by normalisation or quantisation. However, unchanged generated-token embeddings still permit reuse of their first-layer projections, and approximate cache methods can retain a subset of entries while keeping the output nearly unchanged; [CriticalKV](https://arxiv.org/abs/2502.03805) is a reminder that cache inequality is not itself a lower bound on fresh FLOPs.

With fresh noise on every generated-token embedding, even if the token ID is identical across two executions, its first-layer projections contain the fresh terms $W_K(\sigma_tz_t)$ and $W_V(\sigma_tz_t)$. This removes the simplest exact reuse of the newest token's $K/V$ computation and gives the cleanest diffusion-style input-divergence statement. It still does not force the attacker to recompute every old cache entry or every layer: an attacker can reuse clean portions, exploit low-rank structure, or compute challenge-dependent corrections more cheaply than a full forward pass. The theorem should therefore claim “no exact reuse of the challenged input block, subject to a quantised anti-concentration/rank assumption,” followed by the separate LLM non-reuse assumption already required in Section~5.

An input that is guaranteed not to repeat as a token sequence is not enough. A transformer can map many distinct embeddings to the same quantised representation or to states with the same output, and the attacker can reuse internal subcomputations even when the raw input differs. The PoML object that must be shown distinct is the actual incremental input $(\widetilde e_t,C_t)$, including the cache representation, not merely the concatenated text.

### Recommended PoML position

For the main construction, use one-time dense perturbation of the registered prompt embeddings (preferably at a precisely specified point after any input normalisation and before the first $Q/K/V$ projections). This has the clearest inference-time precedent and the lowest semantic risk. Retain the Section~5 statement as a conditional cache-divergence assumption and measure how much clean prefill/cache work remains reusable.

Treat recursive per-token noise as an optional, stronger anti-reuse variant. It is the closest analogue of the diffusion proof and is the best candidate if the paper needs a theorem that every incremental call receives a fresh challenge-dependent input. But it should not be presented as literature-backed inference practice until the following ablation is run on the exact model and quantised DeepProve representation:

1. Clean decoding; one-time prompt noise; RESTA-style prefix noise; teacher-forced all-token sensitivity; and free-running per-token noise.
2. A logarithmic grid of normalised noise scales around the literature starting points (including $10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}$), with the grid explicitly labelled as an experiment rather than a universal recommendation.
3. Exact-match and semantic agreement with the clean answer, first-token and per-token flip rates, NLL/KL changes, pass@1/task accuracy, length and stop-token failures, layerwise activation/cache distances after quantisation, and an adversarial accounting of reusable FLOPs/ZKP constraints.
4. Separate challenge streams for token sampling and embedding noise. The noise schedule must remain unpredictable until the parent block/VRF is fixed and must be domain-separated by query, role, and step.

If the per-token curve shows unacceptable semantic drift, a short prefix-only schedule can be used as a utility compromise, but the paper must then state plainly that later calls can still reuse clean computation and that the full diffusion-style per-step claim no longer applies. The current literature justifies making the experiment; it does not justify assuming a favourable result.

## What one-time dense prompt perturbation rules out

Fix a token sequence $x=(x_1,\ldots,x_n)$ and let the public, unpredictable challenge $r$ determine a dense perturbation $\eta_r=(\eta_{r,1},\ldots,\eta_{r,n})$. Write

\[
\widetilde H_r=Q_E\bigl(E(x)+\eta_r\bigr),
\qquad
C_r=\operatorname{Prefill}_\theta(\widetilde H_r),
\]

where $Q_E$ is the circuit's embedding/activation quantizer and $C_r$ is the complete layerwise prompt $K/V$ cache. For two distinct seeds $r,r'$, a large challenge space and a suitably scaled discretized-Gaussian sampler can make

\[
\Pr[\widetilde H_r=\widetilde H_{r'}]
\]

negligible. The relevant condition is separation between the **two perturbations after quantisation**, not merely that each perturbation has a norm larger than one quantisation step. For a uniform scalar quantizer with bin width $\delta_E$, equality of two quantized vectors implies their unquantized coordinatewise difference is smaller than approximately $\delta_E$ in every coordinate. Thus an $\ell_\infty$ anti-concentration bound is the natural input-collision statement.

Input separation alone does not logically imply cache separation. Let $N$ denote the normalisation preceding the first attention projections and let $A=[W_K;W_V]$. The first-layer cache is of the form

\[
C^{(1)}_r=Q_{KV}\bigl(A N(E(x)+\eta_r)\bigr).
\]

Two distinct quantized input tensors can collide after $N$, can differ only in a null direction of $A$, or can produce a projected difference smaller than the $K/V$ quantisation step. The theorem should therefore bound

\[
\Pr[C^{(1)}_r=C^{(1)}_{r'}]
\]

directly. For independent dense Gaussian blocks, a non-degenerate effective rank of the projected noise, adequate singular values, and noise scale larger than the $K/V$ quantisation resolution give a small-ball bound of the qualitative form

\[
\Pr[C^{(1)}_r=C^{(1)}_{r'}]
\leq
\left(
  c\,\frac{\delta_{KV}}{\sigma s_{\mathrm{eff}}}
\right)^{d_{\mathrm{eff}}n},
\]

where $d_{\mathrm{eff}}$ and $s_{\mathrm{eff}}$ capture the rank and conditioning of the normalised $K/V$ randomisation map. The exact constants and exponent depend on the architecture, grouped-query projections, normalisation, finite-field encoding, and the discretised noise sampler. If the protocol injects the perturbation after the relevant normalisation and immediately before $W_K,W_V$, the proof is cleaner because the projected perturbation is linear; perturbing the raw embedding instead requires analysing the normaliser.

Subject to this projected anti-concentration condition, at least one first-layer prompt $K/V$ entry differs except with negligible probability. Consequently, the complete old cache $C_{r'}$ cannot be used verbatim as the cache for challenge $r$: this rules out an **exact full-cache hit** and forces some challenge-dependent online prefill work. It remains true even if the generated output tokens coincide, because the relation concerns the challenge-conditioned internal computation rather than only the final text.

The same conclusion applies to proof replay only if the DeepProve relation binds the challenge. The cleanest statement makes $(x,r)$ public and derives $\eta_r$ inside the verified circuit (or verifies a binding commitment to the derived noisy tensor) before proving inference. An old proof or old full witness for $r'$ is then invalid for $r$, except through a soundness failure or a challenge/noise/cache collision. If only the final output is public and the circuit does not enforce the derivation of $\eta_r$, prompt perturbation provides no such protection.

This does **not** justify the unrestricted sentence “precomputation cannot be reused.” The attacker may still reuse:

- the model, proving key, circuit structure, FFT tables, and other witness-independent prover preprocessing;
- clean token embeddings and challenge-independent components of first-layer projections;
- a clean or previous prefill as a base from which to compute challenge-dependent corrections;
- any circuit columns, cache entries, or layer operations that remain identical;
- partial, low-rank, or otherwise structured computations that permit faster exact updates.

For example, if the perturbation is injected immediately before a linear projection, an attacker can precompute $A N(E(x))$ and compute only the online correction $A\eta_r$ (under the corresponding placement convention). Later attention and MLP activations will generally depend on that correction, but proving that their exact update costs almost a full forward pass is the separate non-reuse assumption required by Section~5.

The safe paper claim is therefore:

> Because the beacon/VRF challenge is unavailable before the parent block and determines a high-entropy dense prompt perturbation, a miner cannot precompute or replay the exact challenge-specific prompt cache, inference witness, or proof. Under a projected anti-concentration assumption for the quantised first-layer $K/V$ map, every challenge forces nonzero online prefill work except with negligible probability. Challenge-independent preprocessing and partial computation may remain reusable; the fraction of fresh work is addressed separately by the model-specific computational-independence assumption and measurements.

This is stronger than merely saying that the raw input is different and weaker than saying that all precomputation is prevented. It is sufficient to rule out the particular attack in which a miner stores one complete prefill/cache/proof for $x$ and reuses it unchanged after learning a new challenge. A percentage lower bound on fresh inference or ZKP work still requires an explicit cost decomposition and either a computational lower-bound assumption or an adversarial implementation benchmark.

## Skewed output distributions do not require output-token divergence

The one-time perturbation construction does not need to make the generated token sequence different. For a prompt such as “Repeat after me: The sky is blue,” the perturbed and unperturbed executions may have essentially the same top token at every step because the logit margins are large. Requiring a different output token would be both unnecessary for proof security and counterproductive for utility: forcing an argmax crossing would require a prompt-dependent, potentially large perturbation and could destroy the requested semantics.

The correct cryptographic object is the statement being proved, not the final token string. Define the public statement to include the model commitment, the original prompt $x$, the beacon/VRF challenge $r$ (and the per-token sampling randomness or its commitment), and the claimed output $y$. The DeepProve circuit must derive

\[
r\ \longmapsto\ \eta_r\ \longmapsto\ \widetilde H_r
\ \longmapsto\ \operatorname{Inference}_\theta(\widetilde H_r),
\]

and check that the resulting output is $y$. If $r\neq r'$, then the public statements are different even when $y(r)=y(r')$. A proof generated for $r'$ is not a proof for the statement with $r$; the verifier rejects it. This rules out **proof replay** without requiring any output-token difference. An old proof could only verify for the new challenge through a soundness failure, a collision in the challenge-to-perturbation/cache representation, or an omitted binding to $r$.

Likewise, if the perturbation is a circuit-derived witness, the witness contains challenge-dependent embedding values and (under the cache-separation condition above) challenge-dependent first-layer $K/V$ values. The old complete witness/trace is therefore not valid for the new statement. A fresh proof must be produced, although witness-independent proving-key preprocessing and portions of the circuit computation may still be reused. “The whole ZKP must be different” should be interpreted as “the old proof cannot be accepted for the new public statement,” not as a lower bound saying that every internal prover operation or every proof group element is guaranteed to change.

The VRF sampling used for later tokens should be bound in the same way. If token sampling randomness is not included in, or deterministically derived inside, the proved relation, an adversary might replay a proof of the same output while changing the claimed sampling process. The usual construction is to domain-separate (i) the prompt perturbation stream and (ii) the per-token sampling stream, derive both from the block-bound challenge, and verify both derivations inside the circuit.

There is no general theorem that a perturbed input must change any particular output token. Under a local margin model, a token changes only when the perturbation crosses a decision boundary; high-margin tokens can remain unchanged with high probability. What can be claimed is instead:

1. the challenge-conditioned input and, under the projected anti-concentration assumption, the prompt cache are different;
2. the public ZKP statement and valid proof are challenge-specific, so an old proof cannot be replayed; and
3. the amount of fresh inference/prover work beyond this minimum is a separate computational-independence question.

This distinction is especially important for Section~5: output uniqueness is not the right invariant. The relevant invariant is uniqueness of the challenge-bound statement and of the quantised cache/witness representation used by the proof.

## A cache-certifiable challenge-conditioned perturbation

The preceding discussion leaves a choice between two goals that should not be conflated:

* **literature fidelity and minimal model changes**, for which the natural construction is RESTA-style noise added to the token embeddings; and
* **a uniform cache-collision bound**, for which the noise should be inserted at a point where the map to a cached tensor is linear and has a measurable rank.

For the operational PoML construction, the preferred choice is the first option: RESTA-style noise added directly to the registered raw token embeddings. This is the closest match to existing LLM practice and avoids changing the transformer block. The resulting cache claim must be model-specific and experimentally audited because normalization and projection can attenuate or collapse a perturbation. The post-normalization intervention described below remains a useful proof-friendly fallback if the raw-embedding experiment cannot produce a satisfactory security/utility operating point; it is not required for the main proposal.

### Exact construction

Assume a pre-norm decoder block, as in the Llama-family architectures used by many ZKML implementations. Let $h_i$ be the residual-stream vector at prompt position $i$ before the first attention block and let

\[
u_i=N(h_i)
\]

be the output of the first RMSNorm/LayerNorm. The protocol derives an independent noise block

\[
z_{r,i}\leftarrow \mathcal N(0,I_d)
\]

from the block-bound VRF challenge $r$ using a domain-separated expansion, and replaces the first attention input by

\[
\widetilde u_{r,i}=u_i+\sigma z_{r,i}.
\]

The first-layer projections are then computed as usual,

\[
q_{r,i}=W_Q\widetilde u_{r,i},\qquad
k_{r,i}=W_K\widetilde u_{r,i},\qquad
v_{r,i}=W_V\widetilde u_{r,i}.
\]

The residual branch remains $h_i$; only the input to this first attention operation is randomized. No perturbation is applied to generated-token embeddings in the one-time prompt version. If the deployed architecture uses a different normalization ordering, the insertion point should be defined directly in the circuit as “the last representation before the first cached $K/V$ projections.”

The circuit derives the noise itself, rather than accepting it as a prover-supplied witness. In practice, the ideal Gaussian is replaced by a fixed, public discretized/truncated Gaussian sampler. A typical domain-separated derivation is

\[
z_{r,i}=\operatorname{GaussExpand}
  \bigl(\operatorname{PRG}(r\Vert H(x)\Vert\texttt{``prefill-noise''}\Vert i)\bigr).
\]

The token-sampling stream must be separate:

\[
\omega_{r,t}=\operatorname{SampleExpand}
  \bigl(\operatorname{PRG}(r\Vert H(x)\Vert\texttt{``token-sampling''}\Vert t)\bigr).
\]

Both derivations and their use in the inference relation should be checked by DeepProve. This prevents a prover from omitting the noise or from substituting a different sampling randomness after seeing the desired output.

This construction is grounded in, but not identical to, existing work. RESTA explicitly adds statistically independent Gaussian (and directional or orthogonal) noise to each user-content embedding and evaluates the resulting generation/safety trade-off; it deliberately leaves newly generated-token embeddings clean ([RESTA](https://arxiv.org/html/2501.16497), Sections 3.1--3.3). A peer-reviewed ICLR 2026 study gives a separate inference-time precedent for adding random perturbations to hidden activations while sampling, and reports comparable behaviour for bounded Gaussian and uniform noise; it also evaluates perturbing attention activations ([Liu et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d032263772946dd5026e7f3cd22bce5b-Abstract-Conference.html); [full paper](https://arxiv.org/html/2502.03799)). R3F and NEFTune support noise tolerance as a representation-level training principle, but do not by themselves justify noisy inference on an unmodified model. I did not find a paper that proves a quantized KV-cache anti-collision theorem for an LLM; that part is a PoML contribution and must be stated as such.

The intervention should be called “post-normalization Gaussian prompt randomization” in the paper, rather than dropout or temperature sampling. It is dense across prompt positions, does not rely on a changed token ID, and does not require changing the model's vocabulary or retraining. It does change the first attention computation, so utility must be measured rather than assumed.

### A prompt-uniform cache anti-collision bound

The useful feature of injecting after $N$ is that the deterministic prompt representation cancels when comparing two challenges. It gives a bound that is uniform over the attacker's choice of prompt.

Let $W_V\in\mathbb R^{m_V\times d}$ be the first-layer value projection. Select $q$ actual output coordinates by a public row-selection matrix $S\in\{0,1\}^{q\times m_V}$ such that

\[
B:=S W_V
\]

has full row rank. The selection can be found once using rank-revealing QR (or another public conditioning procedure) on the committed model weights. Let $s_0=s_{\min}(B)>0$. We only use these $q$ coordinates for the proof; if they differ, the complete $K/V$ cache differs. It is enough to use $V$, so the argument does not depend on the details of RoPE applied to $K$.

For one prompt position and two independent ideal Gaussian blocks $z,z'$, the selected real-valued cache difference is

\[
D= S(v_r-v_{r'})
  =\sigma B(z-z')
  \sim \mathcal N\!\left(0,2\sigma^2BB^{\mathsf T}\right).
\]

Let $\Delta_V$ be the scalar cache-quantization bin width. If the two selected quantized values are equal, every coordinate of $D$ lies in an interval of width at most $2\Delta_V$ around zero (up to the convention used at bin boundaries). Bounding the Gaussian density by its maximum and multiplying by the box volume gives

\[
\Pr[Q_V(Sv_r)=Q_V(Sv_{r'})]
\;\leq\;
\min\!\left\{1,
\frac{\left(\Delta_V/(\sigma\sqrt\pi)\right)^q}
     {\sqrt{\det(BB^{\mathsf T})}}
\right\}
\;\leq\;
\min\!\left\{1,
\left(\frac{\Delta_V}{\sigma s_0\sqrt\pi}\right)^q
\right\}.
\tag{cache-small-ball}
\]

The deterministic term $W_Vu_i$ does not appear in this expression. Thus the bound is the same for a benign prompt, a highly skewed “repeat after me” prompt, or an adversarially selected prompt. With independent noise blocks at $n$ registered prompt positions,

\[
\Pr[C^{(1)}_{V,r}=C^{(1)}_{V,r'}]
\;\leq\;
\prod_{i=1}^{n}
\frac{\left(\Delta_V/(\sigma\sqrt\pi)\right)^q}
     {\sqrt{\det(BB^{\mathsf T})}}
\;\leq\;
\left(\frac{\Delta_V}{\sigma s_0\sqrt\pi}\right)^{qn},
\tag{prompt-cache-bound}
\]

whenever the displayed ratio is below one. If the protocol allows an empty user prompt, $n$ should include at least the registered BOS/template position; otherwise an attacker could intentionally remove the prompt-length exponent. The implementation should also state which system/template positions are registered and perturbed. Perturbing only attacker-visible user tokens does not give a claim about unperturbed system tokens, although one differing registered position is sufficient to rule out an exact full-cache hit.

The proof is elementary: the difference of two independent Gaussian blocks is $\sqrt 2$ times a standard Gaussian, its $q$-dimensional density has maximum $(2\pi)^{-q/2}\det(2\sigma^2BB^{\mathsf T})^{-1/2}$, and the equal-quantization event is contained in a box of volume $(2\Delta_V)^q$. The second inequality follows from $\det(BB^{\mathsf T})^{1/2}\geq s_0^q$.

For the actual DeepProve relation, the theorem should be written with implementation slack. Let $\tau$ be a certified bound on the pairwise projection/rounding error between the ideal real-valued difference and the fixed-point difference, and let $\varepsilon_{\rm samp}$ be the statistical distance between the implemented finite sampler and the ideal Gaussian block. Replacing $\Delta_V$ by $\Delta_{\rm eff}=\Delta_V+\tau$ gives the conservative statement

\[
\varepsilon_{\rm cache}
\;\leq\;
\operatorname{Adv}_{\rm PRG}
 +2n\varepsilon_{\rm samp}
 +\left(\frac{\Delta_{\rm eff}}
              {\sigma s_0\sqrt\pi}\right)^{qn}.
\tag{implemented-cache-bound}
\]

The exact coefficient in front of $\varepsilon_{\rm samp}$ can be tightened once the sampler is specified; the important point is that finite-field quantization and discretization must not be silently replaced by a real-valued Gaussian argument. The rounding margin $\tau$ should be obtained from interval/error propagation through the fixed-point $W_V$ multiplication and output quantizer, not guessed from the norm of the input perturbation.

For a target $\lambda$-bit cache-collision bound, a sufficient design inequality is

\[
qn\log_2\!\left(\frac{\sigma s_0\sqrt\pi}
                         {\Delta_{\rm eff}}\right)
\;\geq\;\lambda,
\tag{design-rule}
\]

in addition to the PRG and sampler error terms. This equation is the quantity that should be reported after measuring $s_0$ and $\Delta_{\rm eff}$ for the committed model and DeepProve quantization. It is substantially stronger than saying that the raw embedding perturbation norm exceeds the input quantization step.

This establishes **no exact reuse of the challenged first-layer value-cache block**, except with the stated probability. It does not establish that the attacker must redo the entire transformer or the entire SNARK prover. The attacker can still precompute $W_Vu_i$, the model-dependent proving key, FFTs, and other challenge-independent work, and can compute the online correction $W_V(\sigma z_{r,i})$. The percentage of fresh FLOPs or proving work therefore remains a separate cost-accounting assumption/measurement for Section 5.

### Why raw embedding noise is less suitable for the main theorem

The most literal RESTA-style alternative is

\[
\widetilde e_{r,i}=E(x_i)+\sigma z_{r,i},
\qquad
C^{(1)}_{r}=Q_{KV}\bigl(A N(\widetilde e_{r,i})\bigr),
\]

with $A=[W_K;W_V]$. This is probably the best first utility baseline because it follows the published embedding-smoothing operation exactly. It does not, however, yield the preceding prompt-uniform theorem without an additional model-specific assumption. RMSNorm/LayerNorm is nonlinear and can remove a radial component; $A$ may have a nullspace (especially with grouped-query attention); and the density of $A N(E(x)+\eta)$ depends on the particular prompt and on the local Jacobian of $N$. A valid theorem would need a certified lower bound on the smallest singular value of

\[
D\!\left(S A N(E(x)+\eta)\right)
\]

over the entire perturbation region, together with a multiplicity or inverse-Lipschitz bound. That can be tested for a fixed model and bounded prompt set, but it is not a generic consequence of “Gaussian noise was added to the embedding.” The paper should therefore either (i) use post-normalization noise for the formal construction, or (ii) keep raw embedding noise and state the cache anti-concentration condition as an experimentally audited assumption.

### Experiment that can support the claim

The experiment should use the exact model weights, tokenizer, first-layer normalization, DeepProve fixed-point arithmetic, cache quantizer, VRF-to-noise derivation, and proof relation intended for deployment. Floating-point PyTorch results alone cannot validate a finite-field cache claim.

1. **Calibrate the linear map.** Compute $W_V$ in the committed arithmetic, choose $S$ by rank-revealing QR, and report $q$, $s_0$, $\det(BB^{\mathsf T})$, the output bin width $\Delta_V$, and the certified rounding margin $\tau$. Repeat this for each attention group if the implementation has grouped-query projections. If no sufficiently conditioned rows exist, lower $q$ or use a public orthogonal output transform inside the cache relation and account for it explicitly.

2. **Use an adversarial prompt suite.** Include copy/repetition prompts, one-token and empty prompts (with the mandatory BOS/template token), highly peaked factual prompts, long repeated-token prompts, code/JSON formatting prompts, ordinary instruction-following prompts, and reasoning/math prompts. The prompt should be chosen before the VRF challenge, as it is in the protocol, but the evaluation should include prompts selected after inspecting the model and quantizer.

3. **Sweep a normalized noise grid.** Because RESTA's raw embedding scale is not numerically transferable to a post-RMSNorm intervention, report $\sigma$ relative to the RMSNorm output scale (which is approximately one) and include a logarithmic grid such as $10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2},3\times10^{-2}$. Select the final value on a held-out calibration set using both the design rule above and utility constraints. The grid is an experimental protocol, not a claim that one value is universal.

4. **Draw independent challenge pairs.** For every prompt and noise scale, generate many independent pairs $(r,r')$ and record (a) exact collisions of the quantized input, first-layer $K$, first-layer $V$, and complete prompt cache; (b) the fraction of cache coordinates and token blocks that differ; (c) $\ell_\infty$ and $\ell_2$ distances before and after quantization; and (d) the first layer/coordinate at which the DeepProve witness differs. Run the same tests with the raw embedding baseline.

5. **Measure utility and sequence sensitivity.** Record next-token logit margins, KL divergence from the clean next-token distribution, first-token and per-token flip rates, exact-sequence agreement, semantic/task accuracy, format and stop-token failures, length, and (where applicable) pass@1. Include the “repeat after me” family explicitly: output-token agreement is expected to be high and is not a failure of the cache-security claim.

6. **Measure the attacker's actual reuse opportunity.** Implement the strongest obvious attacker: precompute clean embeddings, $W_Vu_i$, all challenge-independent circuit/prover preprocessing, and any reusable cache columns; after $r$ is known, compute only the challenge correction and update the proof. Compare full recomputation with this optimized update in wall-clock time, field multiplications, witness columns, FFTs, and memory traffic. This is the experiment that can provide the percentage relevant to Section 5; cache inequality by itself cannot.

No observed collision in a finite experiment proves a cryptographic negligible probability. With $M$ independent challenge pairs and zero observed collisions, the approximate one-sided 95% upper bound is $3/M$ (the rule of three), which is useful for engineering but nowhere near $2^{-128}$ unless $M$ is astronomically large. The paper should therefore combine the analytic bound (cache-small-ball through implemented-cache-bound) with the experiment: the experiment validates the measured rank, conditioning, rounding slack, and utility; it does not replace the anti-concentration argument.

### Recommended claim for the paper

The cleanest Section~5-compatible claim is:

> The VRF challenge is unavailable before the parent block and is used to derive independent dense Gaussian blocks at every registered prompt position. We inject these blocks after the first normalization and before the first $Q/K/V$ projections. For the committed model, if the selected first-layer value map has rank $q$ and smallest singular value $s_0$, then the quantized prompt cache collides across two challenges with probability bounded by Eq.~(implemented-cache-bound). Thus an old complete prompt cache, witness, or proof cannot be replayed for a new challenge except with the stated cache-collision, sampler/PRG, or proof-soundness error. This rules out exact full-cache/proof reuse, not all challenge-independent preprocessing. The fraction of fresh inference and prover work is measured separately.

Raw RESTA-style embedding perturbation should be reported as the closest literature-faithful baseline. If it has equal or better utility, it may be used operationally, but its security statement should retain the explicit Jacobian/cache anti-concentration assumption rather than silently inheriting the post-normalization theorem. If changing the first attention input is unacceptable architecturally, the paper should make that trade-off explicit instead of claiming that any nonzero embedding perturbation automatically makes the entire KV cache or ZKP non-reusable.

### Bridge back to $\varepsilon_i$ in Section 5

The cache theorem above is necessary for the current reduction, but it is not by itself a proof of the $\varepsilon_i$-computational-independence definition in `5_reduction.tex`. A miner can precompute the clean term $W_Vu_i$ and, after the beacon is revealed, compute $W_V(\sigma z_{r,i})$. The fact that the resulting cache is different does not say whether that correction, and the downstream transformer/prover update, costs $1\%$ or $99\%$ of a fresh query.

The precise way to preserve the existing reduction is to introduce an explicit *conditional update-cost* parameter. Let $\rho_{\rm inf}$ be a bound on the maximum fraction of a fresh inference depth that can be saved by an adversary that has arbitrary prior traces but is conditioned on a non-colliding challenged cache. State the LLM analogue of the existing diffusion assumption as

\[
\Pr\!\left[
  \mathsf{Cost}_{\mathcal A_2}(M_\theta(x,r)\mid\text{prior traces})
  < (1-\rho_{\rm inf})\,\mathsf{Depth}(M_\theta)
  \;\middle|\; C_r\neq C_{r'}
\right]
\leq \varepsilon_{\rm update}.
\tag{LLM-update-CIA}
\]

Then the unconditional failure probability of the lower-cost event is at most

\[
\varepsilon_{\rm cache}+\varepsilon_{\rm update},
\]

up to the PRG, sampler, and proof-soundness terms. In the notation of Definition~\ref{def:delta-reuse}, the parameter to use is $\varepsilon_i=\rho_{\rm inf}$, not $\varepsilon_{\rm cache}$. The latter is the probability that the adversary gets an exact cache collision; the former is the maximum *fractional computation saving* after a cache collision has been ruled out. This distinction prevents the Section~5 theorem from treating “one cache entry changes” as if it implied “almost the whole inference is recomputed.”

The same decomposition should be made for the prover. If $\rho_{\rm pf}$ is the maximum reusable fraction of the DeepProve proving depth on a new statement whose challenge-dependent witness/cache differs, and $\varepsilon_{\rm pf}$ is the residual failure probability, then set

\[
\varepsilon_p=\rho_{\rm pf},
\qquad
\varepsilon_{\rm total}
\leq
\varepsilon_{\rm cache}+\varepsilon_{\rm update}+\varepsilon_{\rm pf}
\]

for the corresponding inference--proof pair. The hypothesis needed by Theorem~\ref{thm:amort-resist} remains

\[
\max(\rho_{\rm inf},\rho_{\rm pf})
<\frac{1}{q_{\mathsf{ML}}+1},
\]

with the negligible/cryptographic error terms added to the theorem statement. If the measurements only support, for example, a $40\%$ reusable fraction, then the paper must use $\rho_{\rm inf}=0.40$ (and the corresponding prover value), rather than claiming negligible reuse. If that value violates the displayed Section~5 inequality, the protocol parameters or the construction must change; the cache-collision theorem cannot repair an insufficient cost bound.

The proposed benchmark in the previous subsection should therefore report two separate outputs: (i) an analytic $\varepsilon_{\rm cache}$ from the small-ball bound, and (ii) an empirical upper confidence bound on $\rho_{\rm inf}$ and $\rho_{\rm pf}$ for a clearly specified attacker class. A universal lower bound against every circuit-level adversary remains a computational assumption, just as `5_reduction.tex` already assumes for the SNARK. This is the honest way to keep the diffusion-style logic relevant without claiming more than the perturbation actually proves.

## Raw embedding perturbation as the operational choice

Raw embedding perturbation is a reasonable choice for PoML, provided that the paper makes the security claim empirical/model-specific rather than presenting it as a theorem that follows from Gaussian noise alone. It is closer to published LLM practice than changing the first attention block, and it leaves the architecture and tokenizer unchanged.

The exact insertion point should be stated as the model's *prefill input representation immediately before its first normalization*. For architectures with additive positional embeddings, $E(x_i)$ in the equations below should mean the token-plus-position representation; for RoPE-style models it is normally just the token embedding, since RoPE is applied later to $Q/K$. This avoids ambiguity about whether the positional component is part of the perturbed or unperturbed input.

### Proposed noise law

Let $E\in\mathbb R^{|\mathcal V|\times d}$ be the committed input-embedding table and let

\[
s_E:=\operatorname{std}\{E_{v,j}:v\in\mathcal V,\ j\in[d]\}
\]

be its entrywise standard deviation, computed once from the public model. For every registered prefill position $i$ (including a protocol-owned BOS/template position if the user prompt can be empty), derive an independent standard-normal block and set

\[
\widetilde e_{r,i}
   =E(x_i)+\eta_{r,i},
\qquad
\eta_{r,i}=\alpha s_E z_{r,i},
\qquad
z_{r,i}\sim\mathcal N(0,I_d).
\]

Here $\alpha$ is dimensionless and $\sigma_{\rm abs}=\alpha s_E$ is the actual per-coordinate noise standard deviation. The zero-mean isotropic law is preferable to hard/soft directional noise for PoML: it has full support in every embedding direction, is independent of the particular token direction, and gives the strongest generic anti-concentration intuition. Directional noise has lower effective rank and its scale is not comparable to isotropic noise. Orthogonal noise may preserve embedding direction better, but the projection itself is token-dependent and can interact badly with normalization; it should be an ablation, not the default.

The parameter $\alpha$ has an interpretable relative scale. For one position, $\mathbb E\|\eta_{r,i}\|_2^2=d\alpha^2s_E^2$. If typical token embeddings satisfy $\|E(x_i)\|_2\approx s_E\sqrt d$, then $\alpha$ is approximately the RMS perturbation divided by the RMS embedding magnitude. This makes reporting both $(\alpha,s_E)$ and the absolute $\sigma_{\rm abs}$ more informative than reporting an unnormalised number alone.

The VRF-to-noise map should be deterministic inside the circuit, for example

\[
z_{r,i}=\operatorname{GaussExpand}
  \bigl(\operatorname{PRG}(r\Vert H(x)\Vert\texttt{``embed-noise''}\Vert i)\bigr),
\]

where `GaussExpand` is a public discretized/truncated Gaussian sampler. The token-sampling stream uses a different domain separator. The finite sampler precision must be finer than the cache quantization step; its statistical distance from the intended Gaussian law becomes part of the stated error term. A shared noise vector for all positions should be avoided: independent blocks give the strongest measured cache separation and prevent the experiment from overstating the effective dimension.

At the raw embedding tensor itself, ideal independent Gaussian blocks do give the same elementary input-collision calculation as in the diffusion appendix. If the embedding quantizer has bin width $\Delta_E$, then for one position

\[
\Pr\bigl[Q_E(E(x_i)+\eta_{r,i})
          =Q_E(E(x_i)+\eta_{r',i})\bigr]
\leq
\left(\frac{\Delta_E}{\sigma_{\rm abs}\sqrt\pi}\right)^d,
\]

and independent prompt positions multiply the bound. This is useful evidence that the challenge-conditioned *input* is not reusable. It still cannot be substituted for the cache condition: $N$ and $A=[W_K;W_V]$ occur after this point and may attenuate or collapse the difference. The experiment below is precisely what tests whether the input-level separation survives that pipeline.

This law follows the direct RESTA baseline $e+\mathcal N(0,\sigma^2I)$, which applies independent noise to each user-content embedding. In its Vicuna experiments, RESTA reports useful isotropic/orthogonal operating points around absolute $\sigma=0.01$--$0.04$ (with model- and attack-dependent utility/robustness trade-offs), while directional variants require different scales ([RESTA](https://arxiv.org/html/2501.16497), Sections 3.1 and 4). RSP provides an additional scale reference by fitting a Gaussian to the embedding-table mean and variance, but it is a preprint and appends random vectors rather than adding a small perturbation to each existing token ([RSP](https://arxiv.org/html/2605.11936)). Noiser supplies the most useful calibration idea for our purpose: it uses Gaussian embedding perturbations and binary-searches the largest scale that preserves a model's original next-token prediction across several LLM families ([Noiser](https://arxiv.org/html/2504.02911)). An ICLR 2026 study also supports inference-time perturbation of hidden activations, including attention activations, and reports comparable behaviour for bounded Gaussian and uniform noise ([paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d032263772946dd5026e7f3cd22bce5b-Abstract-Conference.html); [full text](https://arxiv.org/html/2502.03799)). These sources justify the operation and the calibration strategy, but none supplies a PoML cache-security guarantee.

There is also a relevant caution. The peer-reviewed NAACL 2024 Self-Denoised Smoothing study explains that directly feeding randomized inputs can reduce an LLM's performance on noisy examples, and improves the trade-off by adding a denoising stage ([Self-Denoised Smoothing](https://aclanthology.org/2024.naacl-short.23.pdf)). Our PoML construction cannot freely add such a denoising pass because it would increase the proved computation and could erase the challenge-dependent difference. This is why the utility ceiling must be measured on the exact frozen model rather than inferred from RESTA's reported scales.

### How to choose $\alpha$ without guessing

There should not be one universal noise value copied from RESTA. The selected value has to lie in an interval with both a utility ceiling and a cache-separation floor:

\[
\alpha_{\rm cache}\ \leq\ \alpha\ \leq\ \alpha_{\rm utility}.
\]

If this interval is empty on the committed model, raw embedding perturbation is not a viable PoML construction at the desired utility level; reducing the claim or changing the insertion point is then necessary.

**Step 1: define the candidate grid from the literature.** Evaluate absolute scales

\[
\sigma_{\rm abs}\in\{0.005,0.01,0.02,0.04,0.08\},
\qquad
\alpha=\sigma_{\rm abs}/s_E,
\]

and include the table-matched stress point $\sigma_{\rm abs}=s_E$ suggested by the RSP parameterization. The values $0.01$--$0.04$ are a starting range taken from RESTA's isotropic experiments, not a claim that they transfer numerically to another embedding table or model family. If the model's $s_E$ differs substantially, report $\alpha$ as the primary quantity and $\sigma_{\rm abs}$ as a reproducibility detail.

**Step 2: estimate a utility ceiling.** For each held-out calibration prompt $x_j$, draw fixed Gaussian directions $z_{j,k}$ and evaluate the clean and perturbed next-token distributions. Following Noiser, binary-search the largest $\alpha_{j,k}^{\star}$ for which the clean top-1 next token is unchanged. Use a conservative lower quantile, for example

\[
\alpha_{\rm local}=Q_{0.05}\!\left(\{\alpha_{j,k}^{\star}\}_{j,k}\right),
\]

as a *starting* global ceiling: approximately 95\% of the tested prompt-direction pairs then preserve the first prediction. This is only a local/teacher-forced criterion. The final $\alpha_{\rm utility}$ must also satisfy free-running sequence tests: choose the largest candidate whose held-out task accuracy (or exact-match rate), semantic agreement, format compliance, and stop-token reliability remain within predeclared tolerances of the clean baseline. For example, one may require the lower 95\% confidence bound of task accuracy to be no more than $\delta_U$ below clean accuracy and the format-failure rate to increase by no more than $\delta_F$. The tolerances are application choices and should be reported, not presented as literature constants.

The local threshold is informative for why a prompt such as “repeat after me” may tolerate a larger perturbation: its top-token margins are often large. It is not a security criterion. A low-margin punctuation, stop, or reasoning token can be the first place where the same $\alpha$ changes the free-running trajectory.

**Step 3: estimate the cache-separation floor in the exact circuit.** For every candidate $\alpha$, and for every adversarial prompt in the evaluation suite, draw independent challenge pairs $(r,r')$. Compute the exact fixed-point pipeline

\[
E(x)+\eta_r
\;\longrightarrow\;
N(\cdot)
\;\longrightarrow\;
W_K,W_V
\;\longrightarrow\;
Q_{KV}(\cdot).
\]

Record the empirical probability of equality of (i) the normalized prompt representation, (ii) first-layer $K$, (iii) first-layer $V$, and (iv) the complete prompt cache. Also report the distribution of

\[
d_{KV}(x,r,r')
 :=\frac{\left\|Q_{KV}(A N(E(x)+\eta_r))
              -Q_{KV}(A N(E(x)+\eta_{r'}))\right\|_0}
             {\#\text{cache coordinates}},
\]

and the minimum nonzero coordinate distance in units of the cache bin width. Define $\alpha_{\rm cache}$ as the smallest candidate for which the desired empirical condition holds, for example no complete-cache collisions in the test set and a specified lower quantile of $d_{KV}$ above a target fraction. The target fraction should be linked to the adversarial reuse benchmark, not chosen only because the cache “looks different.”

For raw embedding noise, this remains a model-specific assumption:

\[
\sup_{x\in\mathcal X_{\rm eval}}
\Pr_{r,r'}[C_r(x)=C_{r'}(x)]
\leq \widehat\varepsilon_{\rm cache},
\]

where $\mathcal X_{\rm eval}$ is the documented adversarial prompt class. It is not a theorem for all possible prompts. The experiment should use a one-sided Clopper--Pearson upper confidence bound (or the rule-of-three approximation when zero collisions are observed). Even $10^6$ collision-free pairs only gives an engineering upper bound of approximately $3\times10^{-6}$, not a cryptographic $2^{-128}$ statement. The paper should therefore state the measured condition as an assumption/validation result and keep the public challenge binding and DeepProve soundness arguments separate.

**Step 4: measure the actual reusable work.** At each feasible candidate, implement the strongest obvious attacker: precompute the clean embedding lookup, the clean normalized/projection terms, all challenge-independent circuit columns, proving-key preprocessing, FFT tables, and any unchanged generated-token cache entries. Once $r$ is available, compute the perturbation correction and the minimum exact update needed to produce a valid proof. Let

\[
\rho_{\rm inf}(\alpha),\qquad \rho_{\rm pf}(\alpha)
\]

be the measured maximum fractions of fresh inference and proving depth saved by this attacker. Select $\alpha$ only if the measured values meet the Section~5 requirement

\[
\max\{\rho_{\rm inf}(\alpha),\rho_{\rm pf}(\alpha)\}
<\frac{1}{q_{\mathsf{ML}}+1}.
\]

This is the step that turns “different cache” into a bound on the adversary's query rate. The cache experiment alone cannot provide that percentage.

### Recommended decision rule

The operational procedure should be:

\[
\boxed{
\alpha^{\star}
 =\min\left\{\alpha\in\mathcal G:
 \begin{array}{l}
 \text{utility constraints hold on held-out prompts},\\
 \text{cache-separation condition holds on adversarial prompts},\\
 \max(\rho_{\rm inf}(\alpha),\rho_{\rm pf}(\alpha))
       <1/(q_{\mathsf{ML}}+1)
 \end{array}
 \right\},
}
\]

where $\mathcal G$ is the literature-anchored grid above. Choosing the smallest feasible value is preferable because the security benefit is monotone only at the level of the perturbation's anti-concentration, whereas semantic disruption can grow nonlinearly. If no grid point is feasible, expand the grid only after checking the exact quantized cache and utility curves; do not silently increase $\sigma$ until the output changes.

The resulting paper claim should be deliberately narrow: raw RESTA-style Gaussian perturbation is an inference-time, literature-grounded mechanism that empirically produces challenge-dependent quantized KV caches on the committed model. The measured cache condition rules out exact replay of a prior prompt cache and invalidates the corresponding old witness/proof when the challenge is bound in the statement. The Section~5 reduction additionally requires the measured/assumed bounds on $\rho_{\rm inf}$ and $\rho_{\rm pf}$; output-token diversity is neither required nor expected for high-margin prompts.

## DeepProve integration: what the proof must actually bind

The answer to the central question is qualified but positive. A full DeepProve proof does not merely assert that the final token string is syntactically valid. For the supported transformer graphs, the final argmax relation takes the final logit matrix as its input, and the preceding operator proofs link that matrix back through the unembedding, transformer blocks, attention, normalization, and embedding layers. Thus the logits are constrained as private witness values, even though a zero-knowledge verifier normally does not see the logit vector itself. Public DeepProve materials describe this as end-to-end coverage from embeddings through next-token argmax and as proofs of all transformer operators ([DeepProve paper](https://eprint.iacr.org/2026/1112), [open-source README](https://github.com/Lagrange-Labs/deep-prove/blob/master/zkml/README.md)).

There are three qualifications that matter for PoML.

1. **The verifier checks the quantized circuit, not ideal floating-point logits.** DeepProve uses post-training static quantization, finite-field encodings, clamping, and lookup-based approximations for operations such as normalization and softmax. ``Exact logit'' must therefore mean the exact integer/finite-field value required by the committed DeepProve graph and its lookup tables. It does not mean that the verifier learns, or even checks, an FP32 logit vector. If a quantized logit vector $L_t$ is an internal wire feeding argmax, then a false $L_t$ cannot be substituted without violating the preceding layer constraints; nevertheless, many distinct real-valued logits can map to the same quantized $L_t$, and many distinct $L_t$ vectors can have the same argmax.

2. **The current PoML relation is too abstract unless $M_\theta$ is defined as that exact graph.** The notation $M_\theta(x,\mathcal R)=y$ in `4_protocols.tex` hides whether the embedding perturbation, all intermediate quantizers, the complete logit rows, and token selection are in the relation. The paper should define a quantized, challenge-conditioned map rather than rely on the informal phrase ``correct output.''

3. **DeepProve's published implementation is centered on deterministic argmax.** The paper explains that randomized decoding can be derandomized by supplying a seed, but also says that the public implementation does not include a mechanism proving that the initial seed was sampled from a public source. PoML's VRF therefore has to be integrated into the relation and the final decoding proof; checking VRF transcripts outside the ZKP is not enough to prove that the selected token was sampled from the proved logits.

### A PoML-specific relation

For a prompt $x=(x_1,\ldots,x_s)$, query digest $h_u$, challenge seed $r$, and verified VRF outputs $z_{\rm emb},z_1,\ldots,z_T$, define a domain-separated deterministic expansion

$$
\eta_{r,i}=\alpha s_E\,\operatorname{GaussExpand}
\bigl(\operatorname{PRG}(z_{\rm emb}\Vert h_u\Vert\mathsf{qid}\Vert i\Vert\texttt{embed})\bigr),
\qquad
\widetilde E_{r,i}=Q_E\!\left(E(x_i)+\eta_{r,i}\right).
$$

The noise sampler, truncation rule, finite-field encoding, and $Q_E$ must be public and deterministic. In particular, the prover must not be allowed to supply arbitrary noisy embeddings as a witness. A clean formulation is

$$
\begin{aligned}
L_t &= Q_{\rm logit}\!\left(F_{\theta,Q}\bigl(\widetilde E_r(x),E(y_{<t})\bigr)\right),\\
u_t &= \operatorname{CoinExpand}(z_t,\mathsf{qid},t,\texttt{sample}),\\
y_t &= \operatorname{Sample}_{Q}(L_t,u_t),
\end{aligned}
$$

where $F_{\theta,Q}$ is the exact DeepProve integer graph with causal masking and the prompt perturbation inserted before the first normalization. The NP relation should assert the existence of $x$, the perturbation-expansion values, all operator wires (including $L_t$), the selected tokens $y$, and encryption randomness such that

$$
H_{\rm zk}(x\Vert\mathsf{qid})=h_u,\quad
\text{all DeepProve constraints for }F_{\theta,Q}\text{ hold},\quad
y_t=\operatorname{Sample}_{Q}(L_t,u_t),\quad
\mathsf{ct}=\operatorname{Enc}(\mathsf{pk}_u,y\Vert\mathsf{qid};r_{\rm enc}).
$$

For deterministic argmax, the sampling line reduces to the existing argmax relation (with an explicit tie-break rule). For VRF sampling, it is a new relation: the circuit must check the cumulative-probability interval or other precisely specified sampler condition. The token-selection proof is essential. Otherwise a miner could prove a correct logit computation but choose a convenient output token, or could present a valid VRF value that was not actually used by decoding.

The high-level witness currently written as $(x_u,y,r_{\rm enc})$ should be understood as shorthand for a larger auxiliary witness containing $\eta$, all intermediate tensors, all final logit rows, and sampler variables. It is acceptable for those values to remain hidden, but the circuit must constrain them. If the implementation exposes only a partial graph (for example, a final-token argmax gadget), the claim must be narrowed to that partial graph; the paper cannot infer full-inference or full-logit correctness from the output token alone.

### Changes required beyond adding an outer VRF check

The following are protocol requirements, not optional implementation details.

* **Put the perturbation in the committed graph.** Add an embedding-noise operator with fixed $\alpha$, $s_E$, PRG/`GaussExpand` specification, position convention, truncation, and quantization. Include its code/version and parameters in the model/circuit hash. Recalibrate DeepProve's activation scales with the perturbed distribution; otherwise clamping can silently map many challenge values to the same integer and defeat the cache experiment.

* **Bind the noise to the challenge and query.** The statement must include $r$ (or the verified $z_{\rm emb}$), $h_u$, `qid`, model hash, and the position/domain separator. A convenient layout is $\mathcal R=(z_{\rm emb},z_1,\ldots,z_T)$ with a VRF proof for the embedding stream plus the existing per-token proofs. Reusing a token-sampling VRF output as embedding noise is possible only with a domain-separated expansion and an explicit statement that the two uses are linked.

* **Keep the private-input binding.** DeepProve's embedding PIOP is described for a token-index vector available to the verifier, whereas PoML intentionally keeps $x_u$ private and publishes only $h_u$. The PoML integration therefore needs a ZK-compatible private embedding lookup (or a commitment/opening variant) and the in-circuit hash check $H_{\rm zk}(x_u\Vert\mathsf{qid})=h_u$. Simply passing a public token sequence to an unmodified DeepProve verifier would change the privacy model.

* **Prove the actual decoding rule.** If PoML keeps argmax, fix the tie-break and prove $L_{t,y_t}\ge L_{t,v}$ for every competing vocabulary entry in the committed quantized semantics. If PoML uses VRF sampling, specify the quantized softmax/top-$k$ distribution, cumulative-sum precision, comparison convention, and rejection/edge cases, then add the corresponding lookup/range/comparison constraints. Outer verification of $\mathsf{VRF.Vfy}$ does not establish this relation by itself.

* **Bind the sequence-level certification input.** DeepProve obtains efficiency by concatenating the prompt and claimed response and running one causally masked pass. The perturbed prompt must be the input to this pass, and the claimed output sequence must be bound to the same $r$ and sampler stream. A proof for the same token string under a different $r$ is a different statement and must fail unless the complete quantized witness happens to collide.

* **Commit to the exact model and quantization configuration.** The model weights, operator graph, activation scales, lookup tables, outlier-smoothing transforms, bit width, clipping rules, and noise operator all belong in the committed model identifier $\theta$. Otherwise the prover may choose a cheaper or more permissive graph for which the measured perturbation experiment is irrelevant.

* **Separate validity from work.** DeepProve soundness says that an accepted proof has a satisfying witness for the specified relation. It does not give a lower bound on the prover's wall-clock work or forbid reuse of static proving-key preprocessing. The existing Section~5 parameter must therefore continue to use measured/assumed update-cost bounds $\rho_{\rm inf}$ and $\rho_{\rm pf}$; the cache-collision probability is an additional error term, not a substitute for those bounds.

### What one-time perturbation does and does not establish

With the relation above, the intuitive argument is sound in the following limited sense. If two challenges produce different quantized prompt embeddings (or first-layer $K/V$ rows), then the old exact prompt trace and the old proof are not valid for the new public statement. This remains true even when a high-margin prompt such as ``repeat after me'' produces exactly the same output token sequence: token equality is not the relation being replayed; the proof must satisfy the challenge-conditioned embedding, downstream operators, logits, and sampler constraints.

It is not sound to conclude that different prompt $K/V$ rows force every prover operation to be recomputed. DeepProve's proof path has no token-by-token KV-cache memory to replay: its one-pass certification uses the concatenated sequence, and it commits only selected witness information rather than every intermediate tensor. An attacker can still reuse model commitments, proving keys, lookup tables, clean portions of a trace, and possibly algebraic preprocessing. Moreover, a perturbation can be attenuated by normalization or lost in later quantization, and two distinct logit vectors can have identical quantized values or identical sampled tokens. Therefore the formal claim should be “old exact witness/proof replay is rejected except with the measured cache/witness-collision and proof-soundness errors,” while the fraction of fresh inference and proving work remains the separately measured $\rho_{\rm inf}$ and $\rho_{\rm pf}$ assumption.

### Minimal integration experiment

The decisive test should run the unmodified and modified DeepProve graphs on the same frozen model and exact fixed-point configuration. For each prompt and independent pair of VRF challenges, record equality rates for the perturbed embedding tensor, normalized tensor, first-layer $Q/K/V$, every re-quantized activation, final quantized logits, selected tokens, and the complete witness commitments. Then run a reuse benchmark in which the adversary precomputes the clean model/proving context and updates only challenge-dependent columns. Report inference and proving time, field multiplications, committed witness columns, FFTs, and memory traffic. The experiment should include high-margin copy prompts, because identical output sequences with different internal logits are the expected stress case. A zero token-flip rate is compatible with security; a nonzero witness/commitment difference is the relevant observation.

The paper should not claim that the output logits are “different enough” merely because the input perturbation is nonzero. The auditable condition is instead

$$
\Pr_{r,r'}\!\left[W_r=W_{r'}\right]\leq\varepsilon_{\rm wit},
$$

for the exact quantized witness/commitment object $W$ used by the DeepProve integration, together with an empirical upper bound on the reusable inference/prover fractions. If only the cache object is measured, call it $\varepsilon_{\rm cache}$ and do not silently promote it to a bound on the entire ZKP transcript.

**Editorial correction to carry into the main paper.** The sentence in `security_note.tex` claiming that “distinct inputs produce distinct execution traces” is too strong as a circuit statement. A circuit can map distinct inputs to equal values in some wires, and quantization can create further collisions. It should instead say that the proof enforces the unique trace *for the supplied input witness*, while the probability of a challenge-induced collision of the measured cache/witness object is an empirical or model-specific assumption.
