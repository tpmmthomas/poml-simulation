#!/usr/bin/env bash
# Orchestrate the full experiment pipeline:
#   1. PoW hashrate calibration  (~10 s)
#   2. Experiment 1 — block time stability (PoML + PoW, ~8 h at default target)
#   3. Experiment 2 — wasted work sweeps (~5 h)
#
# Usage:
#   ./experiments/run_all.sh                # full run, default target 300s
#   ./experiments/run_all.sh --smoke        # tiny sanity run, ~5 min
#   ./experiments/run_all.sh --skip-exp2    # run only calibration + Exp 1
#   ./experiments/run_all.sh --target 200   # use 200s expected block time
#
# Should be invoked from the project root; relocates to it if not already.

set -euo pipefail

# ---- locate project root (script lives in experiments/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- defaults ----
MODE="full"
TARGET_SECONDS=300
NUM_MINERS=4
BLOCKS_EXP1=50
BLOCKS_EXP2=10
BENCH_SECONDS=5
SKIP_CAL=false
SKIP_EXP1=false
SKIP_EXP2=false

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --smoke                 Quick end-to-end sanity run (~5 min total)
  --target SECONDS        Target expected block time (default: 300)
  --miners N              Number of miners for calibration/Exp 1 (default: 4)
  --blocks-exp1 N         Blocks for Exp 1 each mechanism (default: 50)
  --blocks-exp2 N         Blocks per config for Exp 2 (default: 10)
  --bench-seconds S       Length of the PoW hashrate benchmark (default: 5)
  --skip-calibration      Reuse existing results/pow_calibration.json
  --skip-exp1             Do not run Experiment 1
  --skip-exp2             Do not run Experiment 2
  -h, --help              This message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)            MODE="smoke"; shift ;;
    --target)           TARGET_SECONDS="$2"; shift 2 ;;
    --miners)           NUM_MINERS="$2"; shift 2 ;;
    --blocks-exp1)      BLOCKS_EXP1="$2"; shift 2 ;;
    --blocks-exp2)      BLOCKS_EXP2="$2"; shift 2 ;;
    --bench-seconds)    BENCH_SECONDS="$2"; shift 2 ;;
    --skip-calibration) SKIP_CAL=true; shift ;;
    --skip-exp1)        SKIP_EXP1=true; shift ;;
    --skip-exp2)        SKIP_EXP2=true; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ---- activate venv if present ----
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
  echo "[run_all] using virtualenv: ${PROJECT_ROOT}/.venv"
else
  echo "[run_all] WARNING: no .venv found at ${PROJECT_ROOT}/.venv — using system python"
fi

PY="$(command -v python)"
echo "[run_all] python: ${PY}"

# ---- sanity: required reference log exists ----
REF_LOG="${PROJECT_ROOT}/logs/run_20260423_050539.log"
if [[ ! -f "${REF_LOG}" ]]; then
  echo "[run_all] ERROR: PoML calibration log not found: ${REF_LOG}" >&2
  exit 2
fi

mkdir -p "${PROJECT_ROOT}/experiments/results/logs"

# ---- overall log (mirrored console output) ----
MASTER_LOG="${PROJECT_ROOT}/experiments/results/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
echo "[run_all] ======================================================"
echo "[run_all] mode:           ${MODE}"
echo "[run_all] target seconds: ${TARGET_SECONDS}"
echo "[run_all] miners:         ${NUM_MINERS}"
echo "[run_all] exp1 blocks:    ${BLOCKS_EXP1}"
echo "[run_all] exp2 blocks:    ${BLOCKS_EXP2}"
echo "[run_all] master log:     ${MASTER_LOG}"
echo "[run_all]"
echo "[run_all] Progress is streamed to stdout (block/proof events)."
echo "[run_all] Per-run detailed logs: ${PROJECT_ROOT}/experiments/results/logs/"
echo "[run_all] Tail latest run log: tail -F \$(ls -t ${PROJECT_ROOT}/experiments/results/logs/*.log | head -1)"
echo "[run_all] ======================================================"

start_ts=$(date +%s)

# ---- 1. PoW calibration ----
if [[ "${SKIP_CAL}" == true ]]; then
  echo "[run_all] [1/3] SKIPPING calibration (--skip-calibration)"
else
  echo "[run_all] [1/3] PoW calibration"
  if [[ "${MODE}" == "smoke" ]]; then
    python experiments/pow_calibrate.py \
      --target "${TARGET_SECONDS}" \
      --miners "${NUM_MINERS}" \
      --benchmark-seconds 1
  else
    python experiments/pow_calibrate.py \
      --target "${TARGET_SECONDS}" \
      --miners "${NUM_MINERS}" \
      --benchmark-seconds "${BENCH_SECONDS}"
  fi
fi

# ---- 2. Experiment 1 ----
if [[ "${SKIP_EXP1}" == true ]]; then
  echo "[run_all] [2/3] SKIPPING Experiment 1 (--skip-exp1)"
else
  echo "[run_all] [2/3] Experiment 1 — block time stability"
  if [[ "${MODE}" == "smoke" ]]; then
    python experiments/exp1_block_time_stability.py --smoke
  else
    python experiments/exp1_block_time_stability.py \
      --blocks "${BLOCKS_EXP1}" \
      --target "${TARGET_SECONDS}"
  fi
fi

# ---- 3. Experiment 2 ----
if [[ "${SKIP_EXP2}" == true ]]; then
  echo "[run_all] [3/3] SKIPPING Experiment 2 (--skip-exp2)"
else
  echo "[run_all] [3/3] Experiment 2 — wasted work"
  if [[ "${MODE}" == "smoke" ]]; then
    python experiments/exp2_wasted_work.py --smoke
  else
    python experiments/exp2_wasted_work.py \
      --blocks "${BLOCKS_EXP2}"
  fi
fi

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
printf "[run_all] ======================================================\n"
printf "[run_all] Done in %dh %dm %ds\n" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
printf "[run_all] Results under: %s/experiments/results/\n" "${PROJECT_ROOT}"
printf "[run_all] Master log:    %s\n" "${MASTER_LOG}"
printf "[run_all] ======================================================\n"
