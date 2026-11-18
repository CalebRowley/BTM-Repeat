#!/usr/bin/env bash
set -euo pipefail
B=/workspace/btm
ELMS=(elm_python elm_javascript elm_law)
DOMAINS=(python javascript law typescript rust contracts web)

python -m btm.train --config configs/elm_law.yaml
python -m btm.eval.score --model "$B/ckpt/elm_law/final" --name elm_law

FINALS=()
for e in "${ELMS[@]}"; do FINALS+=("$B/ckpt/$e/final"); done
[ -d "$B/ckpt/avg_uniform" ] || python -m btm.merge --models "${FINALS[@]}" --out "$B/ckpt/avg_uniform"
python -m btm.eval.score --model "$B/ckpt/avg_uniform" --name avg_uniform

for d in "${DOMAINS[@]}"; do
  [ -f "$B/results/prior_$d.json" ] || python -m btm.posterior \
    --candidates "elm_python=$B/ckpt/elm_python/final" "elm_javascript=$B/ckpt/elm_javascript/final" "elm_law=$B/ckpt/elm_law/final" \
    --domain "$d" --out "$B/results/prior_$d.json"
  [ -d "$B/ckpt/avg_post_$d" ] || python -m btm.merge --weights-json "$B/results/prior_$d.json" --out "$B/ckpt/avg_post_$d"
  python -m btm.eval.score --model "$B/ckpt/avg_post_$d" --name "avg_post_$d" --domains "$d" --splits test
done

python -m btm.eval.report
echo PHASE56_DONE
