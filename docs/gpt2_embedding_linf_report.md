# GPT-2 embedding perturbation: Appendix C measurements

Historical absolute-sigma campaign. The current appendix uses the
[relative Gaussian rerun](gpt2_embedding_alpha_report.md), which updates both
utility and trace measurements and reports individual layer outputs.

Date: 2026-09-10. Completed: 616 prompts, four independent challenge pairs
per prompt, five noise settings (including the zero control).

## Result

The smallest measured active block-output distance at `sigma_abs = 0.005`
is **0.0029105842113494873**. It exceeds the `0.001` comparison-bin width
and the appendix proximity tolerance `delta_q/2 = 0.0005`. This minimum
ranges over all 616 prompts, four pairs, 12 blocks, and six sampled prefix
lengths: 177,408 block-vector comparisons at each noise scale.

| Noise sigma | Prompt embeddings | Prompt block tensors | Active block rows | Next-token logits |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 |
| 0.005 | 0.020954996 | 0.085951328 | 0.0029105842 | 0.0090026855 |
| 0.01 | 0.041065007 | 0.22006989 | 0.005012244 | 0.026092529 |
| 0.02 | 0.078324005 | 0.52189863 | 0.011317253 | 0.039535522 |
| 0.04 | 0.16635 | 0.90233588 | 0.024347901 | 0.10009766 |

All cells are **minima of raw L-infinity distances**, not mean differences
or collision counts. Prompt-block minima also range over all 12 blocks.
Active-block and logit minima additionally range over prefix lengths
`t = 0, 1, 4, 8, 16, 32`, where t counts appended reference tokens.

## Quantized distances

The following are minima in integer grid steps, using bin width `0.001`.
Multiply by `0.001` for represented activation units. No measured value was
clipped. Every zero-noise raw and quantized distance was exactly zero.

| Noise sigma | Prompt embeddings | Prompt block tensors | Active block rows | Logits |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | 0 | 0 | 0 |
| 0.005 | 21 | 86 | 3 | 9 |
| 0.01 | 41 | 220 | 5 | 26 |
| 0.02 | 79 | 522 | 12 | 40 |
| 0.04 | 167 | 903 | 24 | 100 |

At the lowest noise scale, prompt-side K/V raw minima were
`0.07495570182800293` and `0.01745268702507019`, respectively. The minimum
active-block distances by stratum were:

| Stratum | Prompts | Pairs per scale | Minimum active-block distance at sigma 0.005 |
| --- | ---: | ---: | ---: |
| Standard benchmarks | 500 | 2,000 | 0.0029105842113494873 |
| Repeat-copy | 16 | 64 | 0.010743200778961182 |
| Resisting correction | 100 | 400 | 0.007508918642997742 |

## Design and provenance

- Reused the original tokenized manifest: 100 prompts each from WikiText-2,
  LAMBADA, HellaSwag, PIQA, and ARC-Easy, plus 16 repeat-copy and 100
  resisting-correction prompts. Prompt lengths are 1–160 tokens. The saved
  manifest contains no retained target-token-suite prompts.
- Reused GPT-2 weights, verified by the same SHA-256 hash as the original
  campaign: `8d16b883bc2f3902959192944c897bbc394ca3b91b27b0b4d2ff8f270b7c0fb4`.
- GPT-2 revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`; 12 blocks,
  hidden width 768, vocabulary 50,257. FP32 inference, FP64 differences,
  NVIDIA RTX A6000 (`cuda:1`), PyTorch 2.11.0+cu128, Transformers 5.16.1.
- Original noise expansion: VRF and domain-separated SHA-256, Box–Muller,
  coordinate clipping at three standard deviations, noise quantum `1e-6`.
  Seed 42; pair seeds match the original common-token-stream condition.
- Both challenges use the same clean greedy 32-token reference continuation.
  EOS is retained as a token in this fixed-length diagnostic. Causal teacher
  forcing evaluates six prefix lengths in one pass; this is not an
  incremental-cache timing benchmark or an exhaustive all-step evaluation.
- Actual block outputs and final normalization are recorded separately.
  The old hidden-state list included final normalization in its block count.
- The old checkpoint contains no raw tensors or distance magnitudes, so a
  measurement rerun was necessary. The utility table reuses saved results.
- Completed run: 16 minutes 25 seconds of prompt-loop wall time, including
  reference generation and recording. This is an execution log, not a model
  throughput benchmark; unrelated integration tests ran concurrently.

## Interpretation and manuscript changes

The revised `.scratch/PoML_paper_draft/appendix.tex` defines the repeated
embedded-context-to-logits map `f_theta` and its decoding recurrence. It
anchors `d` to the embedding/residual width and distinguishes the initial
`s*d` noise coordinates. Assumption 2 now uses the named block outputs, with
a separate explicit cost assumption. Its conditional theorem uses a reuse
bound `rho` and negligible failure in the security parameter `kappa`.

The distances support separation of the tested block tensors at the stated
precision, even with identical continuation tokens. They do not establish
negligible collision probability for arbitrary prompts, or a lower bound on
online inference/proving cost. Generated-token embeddings, first normalization,
and first-layer Q/K/V have **zero** distance at every measured generated
position; attention subsequently reads the perturbed prompt state. The
assumption therefore concerns block outputs, not every suboperation.

The quantizer is applied after floating-point inference and is not an
integrated DeepProve computation. A deployment needs measurements under its
actual fixed-point execution. The original utility prose was also corrected
to describe likelihood scoring rather than free-running generations.

## Reproduce and inspect

See [the measurement guide](features/gpt2_embedding_linf.md) for commands and
output schemas. Artifacts are in `experiments/results/gpt2_embedding_linf/`:

- `linf_checkpoint.jsonl` retains all 1,626,240 individual boundary/pair
  distances, with prompt tokens, reference tokens, and metadata.
- `linf_appendix.csv` and `linf_appendix.tex` regenerate the paper table.
- `linf_summary.csv` retains all strata, boundaries, sampled steps, and
  quantiles. `metadata.json` records hashes and numerical settings.
- `measurement_source.py` preserves the executed source before formatting
  and adding the final compact table exporter. Its hash matches metadata.

Validation: **122 tests passed, zero failed**, including EZKL integration;
Ruff lint/format and Python compilation passed. The output audit verified
all 616 unique prompts, all pair/scale/boundary entries, finite distances,
zero clipping, the zero-noise controls, and the executed-source hash.

Appendix C compiles independently. The full local paper has pre-existing
build problems: an `xcolor` option clash and a libpng error in
`plots/PoML.drawio.png`. The standalone preview supplies the package option
and omits the rest of the paper; cross-references outside Appendix C remain
unresolved in that preview.
