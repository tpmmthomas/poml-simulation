#!/usr/bin/env bash
# Run the new GPT-2 embedding-perturbation campaign in a persistent tmux
# session.  This launcher always starts the session in the background and
# returns; it never switches the caller into tmux.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

SESSION="gpt2-embedding"
MODE="run-all"
ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_gpt2_embedding_tmux.sh [options] [experiment options]

Starts `experiments/gpt2_embedding_experiments.py` in a persistent tmux
session.  The default runs both plan experiments; use --mode utility or
--mode separation to run one.  All options after -- are passed to the Python
driver (for example --smoke, --device cuda, or --output-dir ...).

Options:
  --session NAME   tmux session name (default: gpt2-embedding)
  --mode MODE      run-all, utility, or separation (default: run-all)
  --detach         compatibility flag; sessions are always started in background
  -h, --help       show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --detach) shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; ARGS+=("$@"); break ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

case "${MODE}" in
  utility|separation|run-all) ;;
  *) echo "invalid --mode: ${MODE}" >&2; exit 2 ;;
esac

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required; install it with your system package manager" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

PYTHON="$(command -v python)"
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # The command runs in a login shell inside tmux, so activate explicitly.
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

COMMAND=("${PYTHON}" "experiments/gpt2_embedding_experiments.py" "${MODE}" "${ARGS[@]}")
printf -v COMMAND_TEXT '%q ' "${COMMAND[@]}"

tmux new-session -d -s "${SESSION}" -n "${MODE}" "cd $(printf '%q' "${PROJECT_ROOT}") && exec ${COMMAND_TEXT}"
echo "Started tmux session: ${SESSION}"
echo "Session name: ${SESSION}"
