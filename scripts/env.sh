source /workspace/btm/venv/bin/activate
if [ -f /workspace/btm/.env ]; then
  set -a; source /workspace/btm/.env; set +a
fi
export HF_HOME=/workspace/btm/hf_cache
export TOKENIZERS_PARALLELISM=false
