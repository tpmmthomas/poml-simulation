# Add the GPT-2 prefix-collision experiment

Date: 2026-09-25

## What changed

Added `experiments/gpt2_collision.py` and the reusable implementation in
`src/poml_sim/gpt2_collision.py`. The campaign samples independent GPT-2
traces from fixed WikiText-2 prompts, estimates the probability that two traces
share each generated prefix, and records the first divergence position.

## Reproduction

```bash
python experiments/gpt2_collision.py --device cuda:0 \
  --output experiments/results/gpt2/full/collision
```

The default campaign uses 500 deduplicated 32-token prompts, four pairs per
prompt, temperatures `0.7, 1.0, 1.3, 1.5, 2.0`, full categorical sampling,
and a 32-token cap. It writes resumable checkpoints, CSV summaries, and the
PNG/PDF collision plot. EOS-shortened pairs are excluded from later-prefix
denominators.

## Interpretation

The output measures observable token-level agreement under independent
sampling. It complements the activation-distance trace experiment and does
not prove hidden-state non-reuse.
