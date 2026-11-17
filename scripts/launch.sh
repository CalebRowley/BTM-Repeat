#!/usr/bin/env bash
set -euo pipefail
HOST="${BTM_HOST:-btm-pod}"
if [ $# -lt 2 ]; then
  echo "usage: launch.sh <run_name> <command...>" >&2
  exit 1
fi
RUN="$1"; shift
CMD="$(printf '%q ' "$@")"
LOG_DIR="/workspace/btm/logs/$RUN"
ssh "$HOST" bash -s <<EOF
set -euo pipefail
if tmux has-session -t '$RUN' 2>/dev/null; then
  echo "tmux session $RUN is already running" >&2
  exit 1
fi
mkdir -p '$LOG_DIR'
echo "== launch \$(date -Is) sha=\$(cat /workspace/btm/code/GIT_SHA 2>/dev/null || echo unknown)" >> '$LOG_DIR/stdout.log'
tmux new-session -d -s '$RUN' "source /workspace/btm/code/scripts/env.sh && cd /workspace/btm/code && $CMD >> $LOG_DIR/stdout.log 2>&1"
echo "launched $RUN"
EOF
