# GPT-2 relative Gaussian perturbation: Appendix C results

Completed 2026-09-11 (Hong Kong time). Both experiments were rerun using
`eta = alpha * s_E * z`, where `s_E = 0.14369554758888242` is the population
standard deviation of the entire frozen embedding table and `z` is standard
Gaussian. The relative scales are `0.05, 0.10, 0.20, 0.40`. Neither experiment
clips Gaussian coordinates or applies the old additional noise-rounding grid.

These measurements supersede the [absolute-sigma report](gpt2_embedding_linf_report.md)
for the current Appendix C. They do not relabel the previous results. See the
[method and commands](features/gpt2_embedding_alpha.md) for reproducibility,
noise expansion, dataset sources, and all recorded boundaries.

## Utility

Each task uses the same 100 saved examples as the earlier utility campaign.
Perturbed scores aggregate three seeds per example. Accuracy is the mean
correctness; perplexity is the exponential of mean target-token NLL across
examples and seeds. The retained WikiText-2 examples each have one target token.

| Task / metric | Clean | alpha=0.05 | alpha=0.10 | alpha=0.20 | alpha=0.40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiText-2 / perplexity | 44.80 | 44.79 | 44.77 | 45.39 | 44.14 |
| HellaSwag / accuracy (%) | 25.00 | 25.00 | 24.67 | 24.67 | 24.33 |
| PIQA / accuracy (%) | 62.00 | 62.00 | 63.00 | 61.33 | 62.33 |
| ARC-Easy / accuracy (%) | 36.00 | 36.67 | 36.67 | 36.33 | 36.33 |

The largest perplexity increase is 1.3109%; the largest accuracy decrease is
0.6667 percentage points. At alpha=0.05, perplexity decreases by 0.0113%,
HellaSwag and PIQA are unchanged, and ARC-Easy increases by 0.6667 points.
These are descriptive point estimates on small subsets, not evidence that
noise improves task performance. The paper reports no significance claim.

## Trace distances and quantization

The run contains 100 saved prompts each from WikiText-2, LAMBADA, HellaSwag,
PIQA, ARC-Easy, and Resisting Correction. It excludes the previous 16
repeat-copy prompts and the zero-noise trace setting. Both members of each
of four challenge pairs receive the same clean greedy 32-token continuation.
Distances are sampled at 0, 1, 4, 8, 16, and 32 appended reference tokens.

Each paper point is the minimum of 14,400 tensor distances: whole-context
L-infinity distances at non-logit boundaries, and next-token-vector distances
at the logit boundary. The prefixes share prompts and noise and are not
independent statistical samples. Since a complete context contains its prompt,
its separation does not imply separation of every individual token row.

| Relative alpha | Absolute sigma | Smallest plotted distance | Layer | Minimum logit distance | Minimum quantized logit distance (grid units) |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0.05 | 0.007184777379 | 0.002029299736 | 6: attention | 0.01602172852 | 16 |
| 0.1 | 0.01436955476 | 0.003881677985 | 5: attention | 0.03082275391 | 30 |
| 0.2 | 0.02873910952 | 0.00587439537 | 4: attention | 0.06625366211 | 67 |
| 0.4 | 0.05747821904 | 0.01133602858 | 4: attention | 0.1420593262 | 142 |

All 108 plotted values (27 boundaries, four scales) exceed the comparison
bin width Delta=0.001. The smallest, 0.0020292997360229492, occurs at block 6's
attention output at alpha=0.05. At this scale the embedding minimum is
0.029750486835837364 and the logit minimum is 0.016021728515625. Directly
measured integer distances are at least 2 grid units across the plotted
boundaries and 16 grid units for logits. No comparison-quantizer clipping
occurred anywhere in the 8,294,400 recorded per-pair distances.

A raw infinity distance greater than one uniform rounding-bin width means
at least one coordinate must have a different quantized value, provided the
quantizer does not clip. This excludes exact substitution of the measured
complete tensor at that precision. The proximity tolerance delta_q/2=0.0005
is a separate, smaller threshold. The comparison quantizer is applied after
FP32 inference; these are not measurements of DeepProve's internal integer
activations.

### Exact layer table

Rows list the embedding sum, the projected attention and feed-forward outputs
before residual addition for every block, final normalization, and logits.
Values below are rounded to five significant digits; the CSV retains full
precision. The paper uses a line graph to fit these 27 boundaries compactly.

| Layer output | alpha=0.05 | alpha=0.1 | alpha=0.2 | alpha=0.4 |
| --- | ---: | ---: | ---: | ---: |
| Embedding | 0.02975 | 0.059044 | 0.12465 | 0.24236 |
| 1: attention | 0.1315 | 0.29981 | 0.61164 | 1.3744 |
| 1: feed-forward | 0.17222 | 0.27888 | 0.61937 | 1.2086 |
| 2: attention | 0.068264 | 0.083471 | 0.16948 | 0.26503 |
| 2: feed-forward | 0.082002 | 0.18527 | 0.4277 | 0.96033 |
| 3: attention | 0.0088581 | 0.010183 | 0.024909 | 0.070974 |
| 3: feed-forward | 0.069938 | 0.16909 | 0.12158 | 0.18921 |
| 4: attention | 0.0021369 | 0.005184 | 0.0058744 | 0.011336 |
| 4: feed-forward | 0.062607 | 0.17347 | 0.38105 | 0.49135 |
| 5: attention | 0.0023125 | 0.0038817 | 0.0078122 | 0.015107 |
| 5: feed-forward | 0.016586 | 0.10942 | 0.19115 | 0.16533 |
| 6: attention | 0.0020293 | 0.0085546 | 0.0089186 | 0.018694 |
| 6: feed-forward | 0.016113 | 0.085007 | 0.13095 | 0.11122 |
| 7: attention | 0.0020885 | 0.0099106 | 0.011276 | 0.031779 |
| 7: feed-forward | 0.0094604 | 0.052677 | 0.073128 | 0.10917 |
| 8: attention | 0.0029969 | 0.0059177 | 0.016881 | 0.024541 |
| 8: feed-forward | 0.0069389 | 0.033939 | 0.053741 | 0.026039 |
| 9: attention | 0.0026105 | 0.0099854 | 0.01252 | 0.023286 |
| 9: feed-forward | 0.0062199 | 0.016689 | 0.030973 | 0.033714 |
| 10: attention | 0.0049092 | 0.011325 | 0.021289 | 0.04424 |
| 10: feed-forward | 0.0053349 | 0.013309 | 0.036801 | 0.045256 |
| 11: attention | 0.0054735 | 0.018527 | 0.021471 | 0.06023 |
| 11: feed-forward | 0.006959 | 0.023739 | 0.049181 | 0.098324 |
| 12: attention | 0.047852 | 0.54712 | 0.96631 | 0.87207 |
| 12: feed-forward | 0.097579 | 0.26493 | 0.34109 | 1.1857 |
| Final normalization | 0.11088 | 0.15176 | 0.39626 | 0.76826 |
| Logits | 0.016022 | 0.030823 | 0.066254 | 0.14206 |


### What this establishes

The observations support finite-sample numerical separation, including when
the compared token continuations are identical. They do not prove a
cryptographic collision probability, exclude partial tensor reuse, or establish
a depth lower bound against cheaper update algorithms. Some generated-token
embedding rows and early position-wise projections are identical by construction;
the detailed diagnostic scopes retain those values.

The paper now takes the inference output to be the full sequence of
pre-decoding logit vectors in a fixed numerical representation. Delivered
tokens may agree across challenges. The inference relation binds logits and
their decoding, and the conditional independence theorem retains a short
separate cost assumption. Main-text model/protocol references use the same
output convention. This change is a paper definition; the campaign measures
fixed-context sensitivity and does not implement that complete proof relation.

## Provenance and validation

- GPT-2 revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Weight SHA-256: `8d16b883bc2f3902959192944c897bbc394ca3b91b27b0b4d2ff8f270b7c0fb4`.
- Float32 inference, float64 differences; NVIDIA RTX A6000, PyTorch 2.11.0+cu128,
  Transformers 5.16.1; TF32 disabled; master seed 42.
- Both metadata files were checked to use identical model weights, table std,
  alpha/sigma values, and noise settings. Executed source hashes match the
  preserved `source/` copies.
- Completed 400 utility examples and 600 unique trace prompts. The trace log
  records 19m37s for its prompt loop, including recording; this is not a
  throughput benchmark. Utility records contain 4,800 perturbed example/seed/scale
  scores. Dataset and prompt counts were checked directly from the records.
- `.venv/bin/pytest -q`: **169 passed**, including EZKL integration. The pre-existing
  heatmap image tests are slow on this host (full suite: 16m14s).
- Ruff checks passed for the modified experiment modules and tests; formatting
  passed for the new driver, distance driver, and their test modules.
- Standalone Appendix C compiles with the manuscript preamble and bibliography,
  with no missing citations/references or overfull boxes. Two main-text reference
  numbers are imported from the existing `main.aux` for the standalone preview.
  A full-document build remains blocked by the existing `plots/PoML.drawio.png`
  libpng error, outside these edits.

Artifacts are local and gitignored under `experiments/results/gpt2_embedding_alpha/`:

- `utility/utility_results.csv`, `utility/utility_summary.csv`, and checkpoints.
- `trace/distances.jsonl`: every raw/quantized pair distance, prompt, and reference tokens.
- `trace/layer_minima.csv`, `trace/layer_table.md`, and `trace/layer_table.tex`.
- `trace/layer_distances.pdf`, `trace/layer_distances.png`, `trace/layer_plot.tex`.
- `trace/boundary_summary.csv`: additional normalizations, residual block outputs,
  first-layer Q/K/V, and prompt-only/active-row scopes.
- Both `metadata.json` files, executed `source/` snapshots, test/build logs,
  and the standalone `paper/appendix_c.pdf`.
