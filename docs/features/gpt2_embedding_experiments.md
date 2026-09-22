# GPT-2 embedding-perturbation driver

`experiments/gpt2_embedding_experiments.py` is the implementation companion to
[`gpt2_new_plan.md`](../gpt2_new_plan.md). It is intentionally separate from
the original `gpt2_experiments.py` driver.

For the appendix's actual activation-distance tables, use the
[L-infinity measurement driver](gpt2_embedding_linf.md). The original
collision checkpoints described below do not retain difference magnitudes
and cannot reconstruct those tables.

## Experiments

- `utility` scores WikiText-2, LAMBADA, HellaSwag, PIQA, and ARC-Easy examples
  with original and one-time challenge-conditioned prefix embeddings. It writes
  paired per-example observations, bootstrap summaries, task-level normalized
  utility graph, and a metric-change plot. The utility graph uses an
  original-result-normalized scale (`1.0 = original, unperturbed result`); it
  overlays all benchmarks with separate colors and line styles. The change plot
  reports perplexity percentage changes and accuracy percentage-point changes
  separately.

  The files are `utility_curve.png` and `utility_changes.png` in the utility
  output directory.

### Utility metric definition

Perplexity is the standard autoregressive language-model score derived from
next-token log probabilities. For a target-token sequence after a
context `x`, the usual definition is

`perplexity = exp(-(1/T) * sum_t log p(y_t | x, y_<t))`.

Lower perplexity means the model assigns higher probability to the observed
continuation. The utility graph reports `original perplexity / perturbed
perplexity`, so `1.0` means the embedding perturbation left the result
unchanged. Classification-style tasks use `perturbed accuracy / original
accuracy` instead.

- `separation` evaluates no-perturbation, common-token-stream, and full-
  protocol pairs. It records complete quantized embedding/activation/logit
  tensors only as compact collision and changed-coordinate statistics, plus
  prefix-token collision curves and a challenge-bound trace-commitment replay
  check. `rho_pf` is explicitly unavailable until a DeepProve prover is wired
  in; logits are never used as a proxy for proving time.

Both commands checkpoint each prompt in JSONL and show live `tqdm` progress
bars, including resumed items. Prompt units are distributed round-robin across
one model instance per device. If settings change, an incompatible checkpoint
is archived as `.stale-*` and a fresh checkpoint starts automatically; matching
settings continue to resume.
The practical `run-all` defaults use 50 utility examples per task, 100
standard separation examples per task, and cap each structural stratum at 100
prompts. Sigma values, conditions, and pairs remain at their plan defaults.
Increase `--examples-per-task` and `--max-structural-prompts` for the full
release-scale campaign.

Separation work is the Cartesian product of prompts and `--sigmas`. For a
fast pilot, use fewer sigmas/pairs/conditions as described by the CLI; the
default command already applies the 100-prompt structural cap.

The main separation artifacts are `trace_separation.csv` and its pooled
`trace_separation_summary.csv`, `prefix_collisions.csv`, `token_agreement.csv`,
and focused full-protocol PNG summaries (`trace_separation.png`,
`prefix_collisions.png`, and `token_agreement.png`). The replay counts remain
available in `replay_test.json` (no replay bar chart is generated), and
`reuse_work.csv`/`reuse_work.png` when `--measure-reuse` is selected.

The trace PNG is an annotated collision table. A collision means every integer
coordinate matches after `QΔ(x) = np.rint(x / Δ)` with `Δ = 0.001`; values are
the same only when they land in the same quantization bin, and one mismatched
coordinate is enough to count the pair as different (green table cells mean no
collision; red means at least one). The prefix heatmaps use a green-to-red
scale from low to high collision probability and use
exact token-ID equality for the first `ℓ` generated tokens, and the sequence
table uses exact equality of the complete generated sequence (up to 32 tokens
or EOS; green means no matching pair and amber means at least one). The
underlying CSVs retain all three experimental conditions for
audit, while these figures focus on nonzero-noise `full_protocol` results.

## Perturbation and quantization

Noise is generated from a domain-separated Ed25519-VRF value and a SHA-256
counter expansion, transformed with Box-Muller Gaussian samples, clipped at
three standard deviations, and quantized to a fixed `1e-6` noise quantum.
Only prompt embeddings are changed; candidate continuation embeddings remain
clean. Trace comparisons use the configured integer quantization scale and
clip, so whole-tensor equality is the security-facing collision predicate.

Public BIG-bench and Resisting Correction prompts may be supplied as
newline-delimited files. `--smoke` uses clearly labelled local structural
fallbacks and synthetic utility rows, so it does not require benchmark dataset
downloads. Results belong below
`experiments/results/` and are gitignored. Each campaign also emits a
newline-delimited prompt manifest so tokenized inputs can be audited without
dumping floating-point tensors.

The PIQA utility task uses the parquet-backed `regisss/piqa` mirror, which is
compatible with current `datasets` releases. If a benchmark loader still
rejects a legacy dataset script, the driver falls back to the public
Hugging Face Dataset Viewer rows API rather than executing repository code.
The same path supplies script-free `tasksource/bigbench` repeat-copy prompts
and the parquet `pminervini/inverse-scaling` resisting-correction release for
the separation strata.

## tmux

Run both experiments in a persistent session with:

```bash
scripts/run_gpt2_embedding_tmux.sh --device cuda --devices cuda:0,cuda:1
```

The launcher returns immediately after creating the session; it never invokes
an attach operation. Open or switch to the tmux session yourself.

Use `--mode utility` or `--mode separation` to run one experiment, and
`--session NAME` when multiple campaigns need independent sessions.
