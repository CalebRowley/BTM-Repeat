#!/usr/bin/env bash
set -euo pipefail
B=/workspace/btm
while tmux has-session -t phase56 2>/dev/null; do sleep 60; done
if ! grep -q PHASE56_DONE "$B/logs/phase56/stdout.log"; then
  echo "phase56 did not finish cleanly; skipping phase 7"
  echo OVERNIGHT_FAILED > "$B/OVERNIGHT_STATUS"
  exit 1
fi
python -m btm.manifest
chmod -R a-w "$B/ckpt" "$B/data/tok"
echo OVERNIGHT_DONE > "$B/OVERNIGHT_STATUS"
echo PHASE7_DONE
