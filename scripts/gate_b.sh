#!/usr/bin/env bash
set -euo pipefail
B=/workspace/btm
while tmux has-session -t phase34 2>/dev/null; do sleep 30; done
grep -q PHASE34_DONE "$B/logs/phase34/stdout.log" || { echo "phase34 did not finish cleanly" >&2; exit 1; }
CANDS=(seed="$B/ckpt/seed" elm_python="$B/ckpt/elm_python/final" elm_javascript="$B/ckpt/elm_javascript/final")
python -m btm.posterior --candidates "${CANDS[@]}" --domain law --out "$B/results/law_init_weights.json"
python -m btm.merge --weights-json "$B/results/law_init_weights.json" --out "$B/ckpt/elm_law_init"
python -m btm.merge --models "$B/ckpt/seed" "$B/ckpt/elm_python/final" "$B/ckpt/elm_javascript/final" --out "$B/ckpt/law_init_uniform"
python -m btm.eval.score --model "$B/ckpt/elm_law_init" --name elm_law_init --domains law --splits dev
python -m btm.eval.score --model "$B/ckpt/law_init_uniform" --name law_init_uniform --domains law --splits dev
python - <<'EOF'
import json
import numpy as np
root = "/workspace/btm/results/logprobs"
w = json.load(open("/workspace/btm/results/law_init_weights.json"))
print("weights:", dict(zip(w["candidates"], [round(x, 6) for x in w["weights"]])))
nll = {m: float(-np.load(f"{root}/{m}/law_dev.npy").mean(dtype=np.float64)) for m in ["seed", "law_init_uniform", "elm_law_init"]}
for m, v in nll.items():
    print(f"{m}: law dev NLL {v:.4f} ppl {np.exp(v):.3f}")
print("FLAG: posterior init >5% worse than seed" if nll["elm_law_init"] > 1.05 * nll["seed"] else "posterior init within 5% of seed")
EOF
echo GATE_B_DONE
