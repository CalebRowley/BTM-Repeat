#!/usr/bin/env bash
set -u
POD="$1"
DEADLINE=$(( $(date +%s) + ${2:-12600} ))
STATUS=/workspace/btm/OVERNIGHT_STATUS
while [ ! -f "$STATUS" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
echo "stopping pod $POD at $(date -Is); status=$(cat "$STATUS" 2>/dev/null || echo deadline_reached)"
sleep 30
runpodctl stop pod "$POD"
