#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"
case "$MODE" in
  smoke) FLAGS=(--smoke) ;;
  full) FLAGS=() ;;
  *) echo "usage: prep_local.sh [smoke|full]" >&2; exit 1 ;;
esac
LOG_DIR="logs/prep_$MODE"
mkdir -p "$LOG_DIR"
git rev-parse HEAD > GIT_SHA
[ -z "$(git status --porcelain)" ] || sed -i '' 's/$/-dirty/' GIT_SHA
export TOKENIZERS_PARALLELISM=false
COMMON=(
  --set "out_dir=$PWD/data/tok"
  --set "smoke_out_dir=$PWD/data_smoke/tok"
  --set "domains.contracts.root=$PWD/CUAD_v1/full_contract_txt"
)
for spec in python:3 javascript:3 law:3 typescript:1 rust:1 contracts:1 web:1; do
  d="${spec%%:*}"; w="${spec##*:}"
  nohup caffeinate -is .venv/bin/python -m btm.data.prepare "${FLAGS[@]}" --domains "$d" "${COMMON[@]}" --set "num_workers=$w" \
    > "$LOG_DIR/$d.log" 2>&1 &
  echo "$d pid=$! workers=$w log=$LOG_DIR/$d.log"
done
