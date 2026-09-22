# Relative Gaussian embedding experiments

`experiments/gpt2_embedding_alpha.py` reruns both Appendix C experiments from
saved tokenized inputs. Both paths use **the same law**:

`eta = alpha * s_E * z`, with `z ~ N(0, I)` and
`s_E = std(E_theta, correction=0)` over every embedding-table entry.

The population standard deviation is computed in FP64 from the frozen
checkpoint. For GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`,
`s_E = 0.14369554758888243`. Defaults are relative scales
`alpha = 0.05, 0.10, 0.20, 0.40`; no zero-noise trace results are published.

## Noise and evaluation

The original VRF/domain-separated SHA-256 and Box–Muller expansion supplies
the Gaussian samples. The alpha campaign disables both the old coordinate
clipping at three standard deviations and the additional `1e-6` noise grid.
Only the final cast to the model's FP32 arithmetic remains. As with any
finite implementation, the expansion approximates a continuous Gaussian;
its deterministic seed derivation and finite sampler are part of the artifact.

This is a new measurement campaign, not a relabelling of old absolute-sigma
results. The old utility/collision driver's defaults remain available for
historical reproduction. Its sampler now accepts `clip=None, quantum=0`,
which **both** new paths use, and its utility checkpoint key includes those
noise settings to prevent reuse of incompatible observations.

| Experiment | Inputs | Protocol |
| --- | --- | --- |
| Utility | The same 100 examples each from WikiText-2, HellaSwag, PIQA, ARC-Easy | Three perturbation seeds, reference/candidate likelihood scoring; clean baseline retained |
| Trace | The same 100 prompts each from WikiText-2, LAMBADA, HellaSwag, PIQA, ARC-Easy, Resisting Correction | Four challenge pairs, six sampled prefix lengths, shared clean greedy continuation |

The six-dataset trace set matches the shortened manuscript; the 16 old
BIG-bench repeat-copy prompts are omitted. There is no change to the chosen
examples within the retained datasets.

## Layer meaning and aggregation

The paper plot and exact table list, in execution order:

1. Embedding sum after perturbation and position addition.
2. Projected attention output and projected feed-forward output for each of
   blocks 1–12 (before their residual additions).
3. Final normalization and the next-token logits.

For each non-logit boundary, compare the **whole context tensor** at prefix
lengths `t = 0, 1, 4, 8, 16, 32` (number of appended reference tokens).
For logits, compare the next-token vector at each of those positions.
Each point is the minimum, over 600 prompts, four pairs, and six prefix
lengths, of a tensor's maximum absolute coordinate difference. It is not a
minimum over scalar differences. The figure labels `iA` and `iF` identify
block i's attention and feed-forward output; there is no averaging over layers.

Additional records include complete block outputs after residual addition,
both normalizations per block, first-layer Q/K/V, and prompt-only/active-row
diagnostics. Generated-token embeddings and their first position-wise
projections can be identical even when the context tensor is different.

Both members of a pair use a clean greedy 32-token continuation, so token
changes cannot inflate the measured logit differences. EOS is retained as a
reference token. Causal teacher forcing evaluates the selected prefixes;
this is a fixed-context sensitivity experiment, not a timed incremental-cache
run or a test of every possible free-running continuation.

The output retains raw FP32 activation differences subtracted in FP64 and
integer distances under `Q(x) = clip(round(x / 0.001), ±(2**31-1))`.
With no clipping, a raw difference exceeding one bin width excludes equality
under this uniform rounding quantizer. The paper's proximity threshold
`delta_q/2 = 0.0005` is distinct from the bin width `Delta = 0.001`.
This comparison quantizer is applied after inference; it does not emulate
DeepProve's internal quantized graph.

## Run and report

```bash
HF_HUB_OFFLINE=1 .venv/bin/python experiments/gpt2_embedding_alpha.py utility \
  --alphas 0.05,0.1,0.2,0.4 --device cuda:0
HF_HUB_OFFLINE=1 .venv/bin/python experiments/gpt2_embedding_alpha.py trace \
  --alphas 0.05,0.1,0.2,0.4 --device cuda:1
.venv/bin/python experiments/gpt2_embedding_alpha.py report
```

The default output directory is `experiments/results/gpt2_embedding_alpha`.
Offline mode requires cached weights. `--limit N --output-dir PATH` supports
a separate pilot. Matching checkpoints resume; changed model, source,
manifest, noise, or count metadata requires another directory.

- `utility/utility_results.csv` and `utility/utility_summary.csv` retain
  example/seed scores and paired summaries, including alpha and absolute sigma.
- `trace/distances.jsonl` retains every pair's distance at each measured
  boundary, scope, and prefix length, plus prompt and reference tokens.
- `trace/layer_minima.csv` and `trace/layer_table.{md,tex}` give exact table values.
- `trace/layer_distances.{pdf,png}` provide the paper plot and preview.
- `trace/layer_plot.tex` provides the same coordinates as a portable PGFPlots figure.
- `trace/boundary_summary.csv` retains the additional boundaries and diagnostic scopes.
- Each mode's `metadata.json` records the same scale/noise definition, model
  and manifest hashes, software versions, and measured-source hashes.

The executed sources are preserved in `source/`: later import cleanup and a
native LaTeX plot exporter change file hashes without changing the measurements.

## Output boundary and conditional claim

The manuscript now defines the LLM computation's output as the complete
sequence of pre-decoding logit vectors, in a fixed numerical representation.
Sampled tokens are derived from those vectors and may agree across challenges.
The proved relation must constrain the logit sequence and sampling; checking
only the delivered text would not enforce the revised output boundary.

Activation separation rules out direct substitution of the measured complete
tensors. It does not rule out a cheap update or partial reuse, even after
moving the output boundary to logits. The manuscript therefore keeps a brief
separate cost clause in its assumption. The conditional theorem invokes that
clause instead of deriving a depth lower bound from numerical differences.
Likewise, GPT-2's fixed width is not a cryptographic security parameter; the
finite experiment does not prove the asymptotic `negl(d)` separation premise.

## Sources

Dataset citations are attached to individual names in the manuscript:
[WikiText-2](https://openreview.net/forum?id=Byj72udxe),
[LAMBADA](https://aclanthology.org/P16-1144/),
[HellaSwag](https://aclanthology.org/P19-1472/),
[PIQA](https://doi.org/10.1609/AAAI.V34I05.6239),
[ARC-Easy](https://arxiv.org/abs/1803.05457), and
[Resisting Correction / Inverse Scaling](https://openreview.net/forum?id=DwgRm72GQF).
The [Inverse Scaling release](https://github.com/inverse-scaling/prize/tree/main/data-release)
provides the structural dataset. See the [measured report](../gpt2_embedding_alpha_report.md)
for completed results and the exact layer table.
