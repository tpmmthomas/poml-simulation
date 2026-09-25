# Project Status

Updated: 2026-09-25
**Health:** green — protocol tests and both real model-proof backends pass.
**Now:** paper-aligned simulator and experiment documentation are in place.

## Components

| Area | State | Notes |
| --- | --- | --- |
| Protocol | implemented | Query admission, three-key miner registration, proof chains, complexity-weighted lottery, fees, forks, and deferred responses |
| GPT-2 / DeepProve | verified | Fresh proofs, measured inference–proof pairs, operation counts, and optional fitted weights |
| Tiny U-Net / EZKL | verified | Fresh setup and verified single-pass proofs |
| Experiments | runnable | PoML Liveness and Block Generation Stability, useful-work efficiency, Wasted Work Analysis, DDPM Compatibility, LLM Compatibility, GPT-2 prefix collisions, and the GPT-2/DeepProve complexity function |
| Packaging | verified | MIT original code, optional pinned dependencies, package-data schedule |

## Recent changes

| Date | Change | Ref |
| --- | --- | --- |
| 2026-09-25 | Restore resumable GPT-2 generated-prefix collision and first-divergence experiment | [experiment guide](docs/experiments.md) |
| 2026-09-25 | Add fresh useful-work efficiency campaign with component timings and bootstrap accounting | [experiment guide](docs/experiments.md) |
| 2026-09-22 | Rewrite Markdown around the paper's terminology and the main-branch comparison-table style | [documentation note](docs/changes/2026-09-22-paper-aligned-docs.md) |
| 2026-09-22 | Replace historical workspace with the current paper implementation and experiments | [verification](docs/verification.md) |
| 2026-09-22 | Validate real backends, fresh counts, dataset loaders, and reduced experiment commands | [verification](docs/verification.md) |
| Earlier | Historical measurements and paper drafts retained locally as ignored artifacts | `.scratch/` |

## Todo / next

| Priority | Item | Notes |
| --- | --- | --- |
| optional | Run full paper-scale campaigns on a dedicated host | Use the defaults in [the experiment guide](docs/experiments.md); reduced runs are not paper results |

## Known issues

| Issue | Impact | Workaround |
| --- | --- | --- |
| Full private PoML NP relation is abstracted | Trusted-host receipts; no independent distributed verification or privacy claim | Keep the boundary explicit in reports |
| Serial host execution and zero network delay | Virtual timing differs from distributed elapsed time | Report physical work from canceled attempts separately |
| Modified GPT-2 graph includes extra noise and sampling work | The shipped reference schedule is an approximation for those additions | Treat the schedule as a calibrated public complexity function |
| Runtime weights can be underidentified | Individual coefficients need not represent unique physical costs | Report feature rank and grouped held-out error |
