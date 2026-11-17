#!/usr/bin/env bash
set -euo pipefail
HOST="${BTM_HOST:-btm-pod}"
cd "$(dirname "$0")/.."
SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: uncommitted changes; commit before launching" >&2
  SHA="$SHA-dirty"
fi
echo "$SHA" > GIT_SHA
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude '.pytest_cache' \
  --exclude 'results/' --exclude 'requirements.lock' --exclude 'CUAD_v1/' \
  ./ "$HOST:/workspace/btm/code/"
if [ -d CUAD_v1/full_contract_txt ]; then
  rsync -az --mkpath CUAD_v1/full_contract_txt/ "$HOST:/workspace/btm/raw/cuad/full_contract_txt/"
fi
echo "synced $SHA to $HOST"
