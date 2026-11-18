#!/usr/bin/env bash
set -u
POD="$1"
DEADLINE=$(( $(date +%s) + ${2:-12600} ))
HOST="${BTM_HOST:-btm-pod}"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  STATUS="$(ssh -n -o ConnectTimeout=20 "$HOST" 'cat /workspace/btm/OVERNIGHT_STATUS 2>/dev/null' 2>/dev/null || true)"
  echo "$(date '+%F %T') status=${STATUS:-pending}"
  [ -n "$STATUS" ] && break
  sleep 300
done
echo "$(date '+%F %T') stopping pod $POD (status=${STATUS:-deadline_reached})"
for i in 1 2 3 4 5; do
  runpodctl pod stop "$POD" && { echo "$(date '+%F %T') stopped"; exit 0; }
  sleep 60
done
echo "$(date '+%F %T') FAILED to stop pod $POD" >&2
exit 1
