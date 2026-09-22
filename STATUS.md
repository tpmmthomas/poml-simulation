# Project Status
Updated: 2026-09-17
**Health:** green — 36,000 uniform-fee races audited; focused checks pass and the updated paper compiles.
**Now:** Query-pool section follows the original uniform-fee design with LLM complexity and a complete grid.

## Components

| Area | State | Notes |
| --- | --- | --- |
| Paper evaluation | updated | Original query-pool wording/analysis restored; three full-grid mean ± SD heatmaps |
| Live GPT-2 miner | qualified | Genuine bank stage; replayed inference/proof service times for virtual miners; WikiText prompts, Gaussian noise, EOS/cap |
| DeepProve decoder | qualified | Composite inverse-CDF verification over public logits; context 64 and EOS checked |
| Complexity/lottery | calibrated | Nonnegative runtime regression on 88 existing pairs; actual N,K, frozen integer weights, literal SHA-256 |
| Experiments 1–2 | campaign complete | Varied-K replay and separate 600-block fitted-weight comparison complete; no new proof measurements |
| Query-pool scaling | campaign complete | 36,000 homogeneous LLM races; first-completion credit; 35,829 adopted, 171 exhausted |
| Historical experiments | retained | Original diffusion/EZKL and previous trace-bank simulations |

## Recent changes

| Date | Change | Ref |
| --- | --- | --- |
| 2026-09-17 | Restored uniform-fee query-pool design; reran all M/Q cells with raw complexity and original heatmap style | [uniform-fee guide](docs/features/llm_uniform_fee_scaling.md) |
| 2026-09-16 | Updated active evaluation and surrounding experimental claims; rebuilt figures from archived results only | [paper results](docs/features/llm_paper_results.md) |
| 2026-09-16 | Completed runtime-weighted selection replay; held-out MAPE 10.12% → 3.95%, long-policy ticket-rate excess 8.02% → 1.49% | [weighting guide](docs/features/runtime_weighted_selection.md) |
| 2026-09-15 | Added profiled replay after genuine perturbed inference/proof bank construction; public-logit proofs and scaled hashes | [live guide](docs/features/llm_live_protocol.md) |
| 2026-09-15 | Moved bulk artifacts and environment to NAS with verified workspace links | [storage map](docs/features/nas_artifact_storage.md) |
| 2026-09-15 | Audited historical protocol coverage and corrected full-prefix response accounting | [historical audit](docs/features/llm_protocol_simulation_audit.md) |
| Earlier | Benchmark profiling, relative-noise utility, variable-shape/CUDA, complexity and embedding studies | [docs](docs/README.md) |

## Todo / next

| Priority | Item | Notes |
| --- | --- | --- |
| medium | Assess selection uncertainty before stronger claims | Three seeds/policy; short-policy pooled block yield still 1.179× uniform |

## Known issues

| Issue | Impact | Workaround |
| --- | --- | --- |
| Existing manuscript references | Four unresolved references and one duplicated label outside query-pool section | Repair liveness/appendix references separately; current subsection resolves |
| Replay proofs are source-bound | Each virtual attempt points to a genuine archived proof but does not re-prove its new challenge | Report the bank/replay distinction; lottery and current auxiliary hashes remain fresh |
| Modified graph uses the approved reference C | Counts are not newly audited exact costs of noise/sampling graph | Log raw/scaled C and examine measured C/time |
| Host executes independent virtual miners serially | Canceled jobs still cost physical time/storage | Separate physical and logical accounting |
| Operation weights are underidentified | 47 features have rank six across 17 measured N,K shapes | Treat as an empirical schedule; report held-out prediction error |
