# Workspace artifact storage

Large workspace data lives under `/mnt/nas/thomas_work/poml-sim`. The repository
keeps symlinks at the original locations, so existing commands and absolute
paths inside saved experiment records continue to find their files.

| Workspace path | NAS path relative to `/mnt/nas/thomas_work/poml-sim` |
| --- | --- |
| `experiments/results` | `experiments/results` — historical and future campaign output |
| `models` | `models` — Stable Diffusion downloads |
| `model_cache` | `model_cache` — current DeepProve model downloads |
| `.scratch` | `.scratch` — DeepProve/crypto checkouts, Cargo builds, model caches, proof setups and paper working checkout |
| `.venv` | `.venv` — installed Python environment |
| Generated `model/*.key`, `*.srs`, `*.ezkl`, `*.onnx`, `*.data`, `*.json` | Corresponding `model/` files — historical EZKL artifacts |

Tracked Python files in `model/` stay in the repository. The Hugging Face cache
outside the repository is unchanged; this migration concerns this workspace.
Symlinks themselves and their generated contents are ignored by the root Git
repository. The nested paper and DeepProve repositories retain their Git data.

Run the current campaign directly on NAS:

```bash
.venv/bin/python experiments/run_live_llm_experiments.py all \
  --device cuda:1 \
  --output /mnt/nas/thomas_work/poml-sim/experiments/results/llm_poml/live-protocol
```

The previous `--output experiments/results/llm_poml/live-protocol` spelling
resolves to the same place. Append `--resume` for an interrupted campaign started
with the current implementation/configuration. Historical manifests are retained
byte-for-byte; source or canonical-path changes can still trigger their strict
resume checks. Do not rewrite old provenance to bypass these checks.

The NAS mount must be available to run the environment, build DeepProve, or read
results. New files under the linked directories are automatically stored on NAS.
Use the NAS output tree for additional large artifacts; arbitrary output paths
elsewhere are not silently redirected. Storage latency can affect setup and
artifact I/O, so recalibrate before collecting the new campaign's measurements.
The GPU performs inference/proving on the same host as before.

## Migration and recovery

`scripts/relocate_artifacts.py --destination /mnt/nas/thomas_work/poml-sim` copies
each group with rsync, preserving modes, timestamps, symlinks and hardlinks.
A metadata comparison and parallel SHA-256 checks verify every regular file before replacing the local
directory/file with a symlink. The local original is removed only after the
verified destination is linked. `migration.json` in the destination records each
group's paths, verification timestamp and completion state.

The script refuses tracked source files and unrelated existing destinations.
Stop builds, experiments and package installation before running it. If a copy
or cutover is interrupted, repeat the same command using the system `python`;
the journal and adjacent recovery copy allow continuation even if `.venv` is
temporarily being switched. Recovery tests cover corrupted copies and interrupted
local cleanup as well as ordinary moves.
