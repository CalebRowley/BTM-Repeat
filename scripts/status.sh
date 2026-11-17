#!/usr/bin/env bash
set -euo pipefail
HOST="${BTM_HOST:-btm-pod}"
RUN="${1:-}"
ssh "$HOST" bash -s <<EOF
tmux ls 2>/dev/null || echo "no tmux sessions"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
df -h /workspace | tail -n 1
if [ -n '$RUN' ]; then
  tail -n 3 '/workspace/btm/logs/$RUN/metrics.jsonl' 2>/dev/null || true
  tail -n 5 '/workspace/btm/logs/$RUN/stdout.log' 2>/dev/null || true
fi
EOF
