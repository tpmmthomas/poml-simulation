# Experiments

The scripts in this directory implement the experiments named in the paper.
Start with the [experiment guide](../docs/experiments.md), which explains the
order in which a reader encounters the concepts and the meaning of each metric.

| Paper experiment | Entry point | Purpose |
| --- | --- | --- |
| Fresh inference–proof measurements | `measure_pairs.py` | Build the verified `(T,C)` bank used by the replay experiments |
| Useful-work efficiency | `work_efficiency.py` | Time fresh inference, proof, verification, and protocol overhead |
| Useful-work shard merge | `merge_work_efficiency.py` | Combine isolated worker outputs and recompute ratios |
| PoML Liveness and Block Generation Stability | `liveness.py` | Compare PoML block intervals with a PoW hash lottery or a Poisson control |
| Wasted Work Analysis | `wasted_work.py` | Measure duplicate completed complexity under a fixed query pool |
| DDPM Compatibility: Formal Statements and Experiments | `diffusion_compatibility.py` | Compare Stable Diffusion latent trajectories under perturbation |
| LLM Compatibility: Formal Statements and Experiments | `gpt2_compatibility.py` | Measure GPT-2 utility and layer-wise activation separation |
| GPT-2 generated-prefix collisions | `gpt2_collision.py` | Measure shared output prefixes and first divergence under independent sampling |
| A Reference Complexity Function for GPT-2 and DeepProve | `complexity_counts.py` | Count instrumented operations and validate the public schedule |
| A Reference Complexity Function for GPT-2 and DeepProve | `fit_complexity.py` | Fit nonnegative runtime weights with grouped validation |


## Quick checks

```bash
python experiments/liveness.py --smoke --blocks 3 --pow-mode poisson \
  --output experiments/results/liveness-smoke
python experiments/wasted_work.py --smoke --miners 2,4 --queries 4,20 \
  --targets 300 --repeats 3 --output experiments/results/waste-smoke
```

These commands use synthetic timing fixtures. They check wiring and output
schemas; they are not model measurements.

## Useful-work efficiency

This campaign measures fresh GPT-2/DeepProve responses, rather than replaying a
measurement bank. Five warm-up responses are discarded, followed by 50
responses with prompt lengths 8, 16, 24, and 32 tokens. The reproducible
campaign command below fixes the output cap at one token so the proof run
finishes in a practical amount of time. The output directory
contains the exact prompts, a hardware and source manifest, per-response
`measurements.jsonl`, and `summary.json`.

The useful-work numerator includes model inference, proof generation, and one
proof verification. All remaining online elapsed time is auxiliary work,
including VRF evaluation, hashing, complexity calculation, encryption,
signatures, serialization, artifact I/O, and protocol validation. Setup and
network time are excluded. The campaign reports the ratio of sums
`alpha_cert = total_elapsed / total_useful`, its reciprocal useful fraction,
and a prompt-cluster bootstrap interval within each prompt-length stratum.

```bash
python experiments/work_efficiency.py --device cuda:0 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency \
  --queries 50 --warmup 5 --max-output 1 --bootstrap 10000 --seed 20260925
```

To regenerate only the summary from a completed campaign:

```bash
python experiments/work_efficiency.py --summarize \
  experiments/results/work-efficiency --bootstrap 10000 --seed 20260925
```

The same campaign can be sharded across isolated GPUs. Run, for example, two
warm-ups plus 17, 17, and 16 measured responses on GPUs 0, 1, and 2, then merge
the completed directories:

```bash
python experiments/work_efficiency.py --device cuda:0 --queries 17 --warmup 2 \
  --max-output 1 --seed 20260925 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency-shard0
python experiments/work_efficiency.py --device cuda:1 --queries 17 --warmup 2 \
  --max-output 1 --seed 20260925 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency-shard1
python experiments/work_efficiency.py --device cuda:2 --queries 16 --warmup 2 \
  --max-output 1 --seed 20260925 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency-shard2
python experiments/merge_work_efficiency.py \
  experiments/results/work-efficiency-shard0 \
  experiments/results/work-efficiency-shard1 \
  experiments/results/work-efficiency-shard2 \
  --output experiments/results/work-efficiency
```

The repository's current backend uses a genuine model proof and verification,
but the full private PoML relation and the formally unbiasable VRF are
explicitly represented by trusted-host and sign-then-hash experimental
substitutions. The manifest records these boundaries; the timing result should
be interpreted as an implementation measurement under those substitutions. GPU
devices are isolated across shards; use disjoint `taskset -c` ranges as well if
strict host-CPU isolation is required.
