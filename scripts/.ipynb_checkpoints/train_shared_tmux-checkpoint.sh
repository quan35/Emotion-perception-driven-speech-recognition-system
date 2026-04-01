#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train_shared_tmux.sh [--session-name NAME] [--log-prefix PREFIX] [--] [train_shared.py args...]

This launcher starts the scripted Whisper shared-model training flow inside tmux.
Any arguments after `--` are passed directly to `scripts/train_shared.py`.

Examples:
  scripts/train_shared_tmux.sh
  scripts/train_shared_tmux.sh --session-name shared_derf -- --norm derf
  scripts/train_shared_tmux.sh --session-name shared_audit -- --audit-only
  scripts/train_shared_tmux.sh -- --profile cpu_preflight --smoke
EOF
}

SESSION_NAME="train_shared"
LOG_PREFIX="train_shared"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --session-name)
      [[ $# -ge 2 ]] || { echo "Missing value for --session-name" >&2; exit 1; }
      SESSION_NAME="$2"
      shift 2
      ;;
    --log-prefix)
      [[ $# -ge 2 ]] || { echo "Missing value for --log-prefix" >&2; exit 1; }
      LOG_PREFIX="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

TRAIN_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${LOG_PREFIX}.${TIMESTAMP}.log"
LATEST_LOG_LINK="${LOG_DIR}/${LOG_PREFIX}.latest.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not installed or not in PATH." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
ln -sfn "$LOG_FILE" "$LATEST_LOG_LINK"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Use a different session name or attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

printf -v TRAIN_CMD '%q ' python -u "$REPO_ROOT/scripts/train_shared.py" "${TRAIN_ARGS[@]}"
TRAIN_CMD+="2>&1 | tee "
TRAIN_CMD+=$(printf '%q' "$LOG_FILE")

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME" "mkdir -p '$LOG_DIR'" C-m
tmux send-keys -t "$SESSION_NAME" "$TRAIN_CMD" C-m

cat <<EOF
tmux session started: $SESSION_NAME
Repo root: $REPO_ROOT
Log file: $LOG_FILE
Latest log link: $LATEST_LOG_LINK

Useful commands:
  tmux attach -t $SESSION_NAME
  tmux ls
  tmux capture-pane -pt $SESSION_NAME | tail -n 50
  tail -f '$LATEST_LOG_LINK'
EOF
