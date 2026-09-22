# GPT-2 embedding activation distances

For the revised Appendix C campaign using `sigma_abs = alpha * s_E` in both
utility and trace measurements, see [the relative-noise driver](gpt2_embedding_alpha.md).
This guide documents the earlier absolute-scale campaign.

`experiments/gpt2_embedding_linf.py` measures the magnitude of activation
differences for the LLM appendix. It complements the original utility and
collision campaign with raw and quantized L-infinity distances.

## Why a measurement rerun is necessary

The original `gpt2_embedding/separation` CSVs and JSONL checkpoints retain
collision indicators and changed-coordinate fractions, but neither activation
tensors nor difference magnitudes. Those statistics cannot recover an
L-infinity distance. The new campaign reuses the exact saved tokenized prompts
and noise expansion. The existing utility results do not require rerunning.

## Reproduce

From the repository root, with the existing optional LLM dependencies:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python experiments/gpt2_embedding_linf.py \
  --manifest experiments/results/gpt2_embedding/separation/prompt_manifest.jsonl \
  --device cuda:1 --pairs 4 --sigmas 0,0.005,0.01,0.02,0.04 \
  --steps 0,1,4,8,16,32 \
  --output-dir experiments/results/gpt2_embedding_linf
```

Offline mode requires cached GPT-2 weights. Omit `HF_HUB_OFFLINE=1` to allow
the model download. `--limit N` runs a labelled pilot on the first N manifest
prompts; use a separate output directory. Output settings and model identity
must match to resume. Source changes also invalidate the checkpoint fingerprint;
the completed campaign includes `measurement_source.py` matching its recorded
source hash. All generated artifacts remain under the ignored results directory.

Regenerate tables from the saved distances, with no model inference:

```bash
.venv/bin/python experiments/gpt2_embedding_linf.py \
  --output-dir experiments/results/gpt2_embedding_linf --report-only
```

## Measurement definition

Each prompt receives one clean, greedy 32-token reference continuation. Both
members of every challenge pair use this exact continuation. EOS is treated
as a reference token instead of stopping this fixed-length diagnostic. This
holds the discrete context constant; common sampling uniforms alone would
not guarantee identical token sequences.

A batched causal forward pass evaluates these shared prefixes. In exact
arithmetic its row at position `s+t-1` is the active row of the Transformer
pass on the prompt followed by `t` reference tokens. This evaluates the
mathematical recurrence; it does not benchmark incremental KV-cache execution.
The noise seeds match the original campaign's `common_token_stream` seeds.
Pairs and scales are pooled only for explicitly labelled minima; they are not
treated as independent observations for confidence intervals.

The measured boundaries are the actual embedding sum, first normalization,
first-layer Q/K/V, all 12 actual block outputs, final normalization separately,
and next-token logits. Hooks avoid labelling GPT-2's final normalized hidden
state as the output of the last block. Prompt rows are compared as whole
`s × d` tensors; active rows have `d` coordinates, or vocabulary-size
coordinates for logits. Logits are recorded only at the active positions.

For paired tensors `A,B`, the two recorded quantities are:

- `raw_linf = max(abs(A-B))`, subtracting stored FP32 activations in FP64;
- `quantized_linf_steps = max(abs(Q(A)-Q(B)))`, where
  `Q(x) = clip(round(x / 0.001), -(2**31-1), 2**31-1)`.

Multiplying the second value by `0.001` gives the represented activation
distance. A raw distance of at least one bin width rules out equality for
this rounding quantizer when clipping does not occur; crossing a bin edge can
also change a quantized value at a much smaller raw distance. Clipped
coordinate counts are retained so saturation cannot silently invalidate that
interpretation.

## Artifacts

| Artifact | Contents |
| --- | --- |
| `metadata.json` | Model hash/revision, manifest hash, software, device, noise, steps, quantizer, source hash |
| `linf_checkpoint.jsonl` | Every prompt, reference continuation, and every pair's exact distances at every recorded boundary |
| `linf_summary.csv` | Per stratum/scale/boundary/step minima, fifth percentiles, medians, means, maxima, counts, and clipping |
| `linf_table.md` | Detailed human-readable minima |
| `linf_appendix.csv` | Compact minima pooled over all prompts, pairs, blocks, and selected steps |
| `linf_appendix.tex` | The compact raw-distance table for the paper |

The compact columns compare prompt embeddings, prompt block tensors, active
block rows, and active logit vectors. Each is a minimum of per-pair infinity
norms, never a minimum of individual coordinate differences. The active-block
minimum also ranges over all 12 blocks and six measured steps.

## Interpretation

In the paper, `f_theta` denotes the same Transformer plus language-model head
repeatedly applied to embedded contexts. Its width is `d = d_model = 768` for
GPT-2; an `s`-token perturbed prompt has `s*d` random coordinates. The model's
vocabulary size is 50,257. These dimensions follow the
[GPT-2 model interface](https://huggingface.co/docs/transformers/model_doc/gpt2).

One-time prompt noise does not change a shared generated token's embedding,
first normalization, or first-layer Q/K/V projections. These are useful zero
controls, not block-output failures: the subsequent attention can read the
changed prompt keys and values. The actual block outputs and final logits
are the relevant downstream measurements.

Finite nonzero distances support separation of the measured tensors. They do
not prove a negligible probability for adversarial prompts or a lower bound
on online inference/proving work. The comparison quantizer runs after floating
point inference and is not an integrated DeepProve execution. The revised
appendix retains the computational reuse bound as an explicit additional
assumption and indexes cryptographic negligibility by the security parameter,
not by GPT-2's fixed width.

See [the measured report](../gpt2_embedding_linf_report.md) for the completed
campaign and the table values used in Appendix C.
