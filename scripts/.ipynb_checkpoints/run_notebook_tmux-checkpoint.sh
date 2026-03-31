#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_notebook_tmux.sh <notebook_path> [session_name]

Examples:
  scripts/run_notebook_tmux.sh notebooks/04_train_shared.ipynb train_shared
  scripts/run_notebook_tmux.sh notebooks/03_train_emotion.ipynb baseline_train
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOTEBOOK_INPUT="$1"

if [[ "$NOTEBOOK_INPUT" = /* ]]; then
  NOTEBOOK_PATH="$NOTEBOOK_INPUT"
else
  NOTEBOOK_PATH="${REPO_ROOT}/${NOTEBOOK_INPUT}"
fi

if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "Notebook not found: $NOTEBOOK_INPUT" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if ! command -v jupyter >/dev/null 2>&1; then
  echo "jupyter is not installed or not in PATH." >&2
  exit 1
fi

NOTEBOOK_STEM="$(basename "$NOTEBOOK_PATH" .ipynb)"
SESSION_NAME="${2:-$NOTEBOOK_STEM}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/runs"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_FILE="${NOTEBOOK_STEM}.executed.${TIMESTAMP}.ipynb"
LOG_FILE="${LOG_DIR}/${NOTEBOOK_STEM}.${TIMESTAMP}.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Use a different session name or attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME" "mkdir -p '$RUN_DIR' '$LOG_DIR'" C-m
tmux send-keys -t "$SESSION_NAME" "jupyter nbconvert --to notebook --execute '$NOTEBOOK_PATH' --output-dir '$RUN_DIR' --output '$OUTPUT_FILE' 2>&1 | tee '$LOG_FILE'" C-m

cat <<EOF
tmux session started: $SESSION_NAME
Notebook: $NOTEBOOK_PATH
Executed notebook: $RUN_DIR/$OUTPUT_FILE
Log file: $LOG_FILE

Useful commands:
  tmux attach -t $SESSION_NAME
  tmux ls
  tail -f '$LOG_FILE'
EOF
