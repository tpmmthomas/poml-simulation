# Paper-aligned documentation

Date: 2026-09-22 · Status: done

## Goal

Make the repository documentation read as one authorial set with the current
PoML paper, while retaining the clearer section flow and comparison table used
by the main branch.

## Changes

- `README.md` — reorganized installation, simulator backends, experiments, and
  tests; replaced the implementation audit with a paper-versus-implementation
  table.
- `docs/experiments.md` — reordered commands in the paper's reading order and
  introduced terminology before using it.
- `experiments/README.md` — mapped each paper experiment to its entry point and
  treated additional wasted-work heatmaps as results of the same experiment.
- `docs/verification.md` — separated completed checks from paper-scale runs not
  rerun during verification.
- `docs/README.md` and `STATUS.md` — repaired navigation and recorded the new
  documentation baseline.

## Decisions

The paper's names are canonical: PoML Liveness and Block Generation Stability,
Wasted Work Analysis, DDPM Compatibility, LLM Compatibility, and the Reference
Complexity Function for GPT-2 and DeepProve. Internal script names remain only
inside commands and entry-point tables.

## Verification

- `rg --files -g '*.md'` — all tracked Markdown reviewed and aligned.
- `git diff --check` — no whitespace errors.
- Existing repository test and style commands remain documented in the README
  and verification record.
