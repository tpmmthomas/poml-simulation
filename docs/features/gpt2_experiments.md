# GPT-2 experiment driver

`experiments/gpt2_experiments.py` is the executable implementation of
`gpt2_plan.md`. It deliberately separates the protocol sampler from model and
dataset loading so the deterministic pieces can be unit-tested without a
checkpoint download.

## Experiment semantics

- The default model is Hugging Face `gpt2`; pass `--revision` to pin a commit.
- Prompts are one deduplicated, fixed-length native GPT-2 BPE window per source
  row. The default source is raw WikiText-2 test; LAMBADA is optional.
- A trace derives an Ed25519 private key from its experiment seed, evaluates the
  repository sign-then-hash VRF once per generated token, and maps the first
  eight output bytes to an inverse-CDF uniform value. VRF messages include a
  domain separator, the tokenized-prompt digest, and a one-based step number.
- Sampling defaults to full categorical sampling (`top_p=1`, no `top_k`, no
  repetition penalty), with temperature applied before softmax. EOS ends a
  trace; no tokens are invented after EOS.
- Prefix comparisons are EOS-censored: a pair contributes to `C_ell` only when
  both traces contain at least `ell` generated tokens. A zero-collision tail
  includes a rule-of-three 95% upper bound rather than a misleading zero claim.
- Bootstrap intervals resample prompts, since prompts—not individual tokens—are
  the unit to which a benchmark-distribution claim generalizes.

## Outputs

`collision` writes `collision_curves.csv`, `prompt_curves.csv`,
`first_divergence.csv`, `collision_aggregates.csv` (including expected
common-prefix length), `lp_estimates.csv` (sequence-probability estimator),
`collision_curves.png`, and `first_divergence_cdf.png`. `concentration` writes
`token_concentration.csv`, `concentration_summary.csv` (position-wise
10/50/90th percentiles), and `top_token_probability.png`. `cache` writes
`cache_timings.csv`, `cache_timings.png`, `cache_workload.csv`, and
`cache_workload.png`; the latter is a bounded-LRU workload model covering
independent, repeated, shared-prefix, and near-match prompts. Timings cover model forward/KV work only and do not assert reusable
ZK-prover witnesses. `perplexity` writes `perplexity.json`.
The collision figure intentionally zooms to generated prefix lengths 1--8,
uses percentage axes, and includes a companion expected-shared-prefix bar
chart; the full 32-token curve remains in `collision_curves.csv`.
Every command writes `metadata.json`, including model revision/configuration,
tokenizer information, software version, VRF construction, and EOS policy.
The default campaign uses 500 prompts, four collision pairs, 32 generated
tokens, and all five temperatures (`0.7,1.0,1.3,1.5,2.0`). With `--device
cuda`, one worker model is created per visible GPU; use `--devices
cuda:0,cuda:1,cuda:2` to pin a three-GPU run. Collision and concentration
workers process prompt/temperature units round-robin and keep the output order
deterministic. The cache timing diagnostic remains a single-temperature
measurement (`--temperature 1.0` by default), since it measures state reuse
rather than the temperature-dependent collision distribution.
Collision and concentration runs also maintain append-only
`*_checkpoint.jsonl` files with matching `.meta.json` fingerprints. They print
temperature, prompt counts, elapsed time, and ETA; rerunning with the same
settings resumes completed prompt/temperature units. Use `--progress-every 1`
for per-prompt output, or `--no-resume` to deliberately restart.
Perplexity prints the same prompt progress (but is recomputed if interrupted).
Pass `--record-transcripts` to `collision` or `run-all` when the full per-step
VRF input/output/proof audit is needed; this can be a large JSONL file. Without
that flag, the seed schedule and derivation rule in `metadata.json` make every
transcript reproducible without duplicating it in the result directory.

The model and dataset are downloaded lazily by Hugging Face. For offline runs,
provide `--prompt-file` with one newline-delimited source row and use a locally
cached `--model` path.
