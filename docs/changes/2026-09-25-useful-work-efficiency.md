# Useful-work efficiency campaign

Date: 2026-09-25

## What changed

The repository now measures fresh GPT-2/DeepProve response production with
exclusive component timers, classifies inference, proof generation, and one
proof verification as useful work, and reports aggregate useful-work ratios.
Three GPU-isolated shards can be merged with
`experiments/merge_work_efficiency.py`.

## Why

The PoML paper needs an implementation-level estimate of the inefficiency
ratio in the wasted-work model. The measurement includes online protocol
overhead while keeping setup, network, and duplicate-query work outside this
per-response estimate.

## Verified campaign

The 50-response campaign used 17, 17, and 16 measured responses on three
separately allocated GPUs, with two warm-ups per shard, one generated token,
and prompt lengths of 8, 16, 24, and 32. The workers inherited the shared host
CPU affinity; strict CPU isolation can be added with disjoint `taskset -c`
ranges. All 50 records contain fresh inference and proof flags and passed
model-proof verification. Aggregate results were 4,104.290 s elapsed,
4,073.955 s useful, `alpha_cert = 1.007446`, useful fraction 99.261%, and
auxiliary fraction 0.739% (10,000 prompt-cluster bootstrap resamples).

## Boundary

The model backend supplies a genuine model proof and verification. The full
private PoML relation is represented by a trusted-host receipt, and the
repository's sign-then-hash construction stands in for a formally unbiasable
VRF. Both substitutions are recorded in each campaign manifest and must be
kept visible when interpreting the number.

## Reproduction

See [the experiment guide](../experiments.md#useful-work-efficiency) for the
single-worker command, the three-shard commands, and the merge procedure.
