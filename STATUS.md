# Project Status
Updated: 2026-09-22
**Health:** green — publication protocol tests and both real model-proof backends pass.
**Now:** `fc27` packages an independent paper-aligned simulator and the requested experiments.

## Components

| Area | State | Notes |
| --- | --- | --- |
| Protocol | implemented | Single-hash threshold, query/seed bindings, deferred responses, fees and forks |
| GPT-2 / DeepProve | verified | Fresh proofs and eight-pair measurement bank; optional fitted weights |
| Tiny U-Net / EZKL | verified | Setup from scratch and fresh verified single-pass proofs |
| Main experiments | runnable | Liveness fresh/replay modes, actual/Poisson PoW, fixed-pool completed waste |
| Appendix | runnable | DDPM, GPT-2 utility/traces, operation counter and nonnegative runtime fit |
| Packaging | verified | MIT original code, optional pinned dependencies, package-data schedule |

## Recent changes

| Date | Change | Ref |
| --- | --- | --- |
| 2026-09-22 | Replace historical workspace with publication implementation and retain only paper experiments | [audit](docs/features/publication.md) |
| 2026-09-22 | Validate real backends, fresh counts, dataset loaders and reduced experiment commands | [verification](docs/verification.md) |
| Earlier | Historical measurements and paper drafts retained locally as ignored artifacts | [scope](docs/experiments.md) |

## Todo / next

| Priority | Item | Notes |
| --- | --- | --- |
| optional | Run full publication campaigns on a dedicated host | [commands](docs/experiments.md); reduced runs are not paper results |

## Known issues

| Issue | Impact | Workaround |
| --- | --- | --- |
| Full private PoML NP relation is abstracted | Trusted-host receipts; no independent distributed verification/privacy claim | Explicit audit and real model proofs |
| Serial host execution and zero network delay | Virtual timing differs from real distributed elapsed time | Report canceled physical jobs separately |
| Modified GPT-2 graph uses reference C | Extra perturbation/sampling operations are not newly counted exactly | Frozen reference schedule; document approximation |
| Runtime weights can be underidentified | Coefficients need not represent unique physical costs | Report rank and grouped held-out errors |
