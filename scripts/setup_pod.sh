#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/btm/{code,data,data_smoke,ckpt,logs,results,artifacts,hf_cache}
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux rsync)
[ -d /workspace/btm/venv ] || python -m venv --system-site-packages /workspace/btm/venv
source /workspace/btm/venv/bin/activate
echo "torch==$(python -c 'import torch; print(torch.__version__)')" > /workspace/btm/constraints.txt
pip install -c /workspace/btm/constraints.txt -r /workspace/btm/code/requirements.txt
pip freeze > /workspace/btm/code/requirements.lock
