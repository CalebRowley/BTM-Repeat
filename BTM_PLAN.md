# BTM Baseline Replication Plan: Pythia-160M Seed + 3 Expert LMs

**For Claude Code:** read this whole file before doing anything. Start at Phase 0 and confirm the user prerequisites with the user. Work phase by phase, run each phase's acceptance checks, and stop at every **[GATE]** to report back and wait for approval.

---

## 1. Goal and scope

Build a small replication of Branch-Train-Merge (Li et al., 2022, arXiv:2208.03306; reference code: github.com/hadasah/btm). It will serve as a **frozen baseline** for later distillation and merge experiments, so reproducibility and clean artifacts matter more than beating any baseline.

**In scope**

- Pretrained seed: Pythia-160M-deduped.
- Three Expert LMs (ELMs):
  - **python** and **javascript**, both branched directly from the seed (independent of each other).
  - **law**, branched from a posterior-weighted parameter average of {seed, python ELM, javascript ELM}. This is a second BTM iteration.
- Evaluation of single ELMs, output ensembling (cached prior, uniform prior, top-k) and parameter averaging (uniform, posterior-weighted) on in-domain and held-out domains.
- A reusable library for posteriors, ensembling and weighted parameter averaging. It includes a full-vocabulary ensemble distribution, which later distillation work will use as the teacher.
- Frozen, checksummed artifacts.

**Out of scope (do not build now):** distillation runs, a dense multi-domain baseline, a random-split ensemble, multi-GPU training, and hosted experiment tracking (logging is local only).

---

## 2. Background (just enough)

- **ELM:** a full copy of the LM trained on one domain only. After branching, ELMs share no parameters and don't communicate during training.
- **Branch:** initialize a new ELM as `θ_new = Σ_j w_j θ_j` over candidate models. `w_j` is candidate j's domain posterior on the new domain's dev data.
- **Train:** plain next-token cross-entropy on the domain's data.
- **Merge:** add the trained ELM to the set. Nothing else happens.
- **Ensemble inference:** `p(x_t | x_<t) = Σ_j p_j(x_t | x_<t) · P(j | x_<t)`, where `P(j | x_<t) ∝ prior_j · Π_{s<t} p_j(x_s | x_<s)`. The posterior resets at the start of every eval window. The prior is "cached": estimated once per target domain from that domain's dev data.
- **Parameter averaging:** collapse the ELMs into one model with a weighted average of their weights (uniform, or using the cached prior as weights).

---

## 3. Fixed decisions

Change these only if the user says so.

| Item | Decision |
|---|---|
| Seed | `EleutherAI/pythia-160m-deduped`, revision `step143000` (final checkpoint) |
| Tokenizer | Pythia (GPT-NeoX) tokenizer. Vocab < 65,536, so tokens are stored as uint16 |
| ELM domains | python, javascript (near pair, both from the seed); law (far, from the posterior-weighted average of seed + python + javascript) |
| Training tokens | 1.0B per ELM, identical for all three |
| Context length | 2048 for training; 1024-token windows for evaluation and posterior estimation (matches the paper) |
| Hardware | RunPod, 1× A100 SXM 80GB, template PyTorch 2.8.0 / CUDA 12.8.1 / Ubuntu 24.04 |
| Storage | ≥200GB RunPod network volume mounted at `/workspace`. The container disk is ephemeral |
| Claude Code | Runs on the user's local machine and operates the pod over SSH |
| Logging | Local only: JSONL metrics and stdout logs on the network volume |

---

## 4. Operating rules for Claude Code

1. **Never** create, start, stop or terminate pods, or change volumes or billing. Ask the user.
2. Everything persistent goes under `/workspace/btm/`. Never write data or checkpoints to the container disk.
3. Launch every job longer than about 2 minutes in a detached `tmux` session on the pod, logging to `/workspace/btm/logs/<run>/`. Never block an SSH command on a long job; poll the logs instead.
4. Never delete or overwrite anything under `data/` or `ckpt/` without asking. The one exception is the rolling resume checkpoint that each training run manages itself.
5. Never print secrets such as `HF_TOKEN` to logs or the terminal.
6. Stream datasets. Never download a full dataset to disk.
7. Never reinstall or upgrade `torch`. Use the constraints file in §7.
8. Stop and report when:
   - an acceptance check fails;
   - a measured time or disk cost exceeds this plan's estimate by more than 50%;
   - anything here turns out to be wrong (dataset schemas, IDs, flags).
9. The local git repo is the source of truth. Sync it to the pod with `rsync`. Commit before every launch, and record the git SHA in every run's config dump.
10. Write unit tests that run on CPU with a tiny randomly initialized GPT-NeoX config, so they can run locally as well as on the pod.

---

## 5. Phase 0: User prerequisites

Confirm each item with the user before starting.

- [ ] An A100 SXM pod from the PyTorch 2.8.0 template, in the **same datacenter** as a network volume of at least 200GB mounted at `/workspace`.
- [ ] Full SSH over an exposed TCP port is enabled, the user's public key is added, and the user has supplied the host IP and port.
- [ ] The user's Hugging Face account has accepted the terms for `bigcode/the-stack-dedup`. A read token is stored on the pod at `/workspace/btm/.env` as `HF_TOKEN=...`, with `chmod 600`.
- [ ] Optional cost saver (the user decides): Phase 1 is CPU- and network-bound. It can run on a cheap CPU pod attached to the same network volume before switching to the A100 pod.

Local `~/.ssh/config` entry, with the user filling in the placeholders:

```
Host btm-pod
  HostName <pod-ip>
  Port <ssh-port>
  User root
  IdentityFile ~/.ssh/<key>
  ServerAliveInterval 30
```

Verify with `ssh btm-pod "nvidia-smi && df -h /workspace"`.

---

## 6. Repository layout

The repo lives locally and is synced to `/workspace/btm/code/`.

```
btm/
  BTM_PLAN.md                 # this file
  requirements.txt            # no torch here
  configs/
    data.yaml
    base_elm.yaml             # shared training hyperparameters
    elm_python.yaml           # data + init overrides only
    elm_javascript.yaml
    elm_law.yaml
    eval.yaml
  btm/
    data/prepare.py           # stream -> filter -> split -> tokenize -> uint16 shards
    data/loader.py            # memmap windows, deterministic shuffle, resumable
    train.py                  # single-GPU trainer, config-driven, resumable
    merge.py                  # weighted parameter averaging (fp32)
    posterior.py              # sequence log-likelihoods, posteriors, cached prior
    ensemble.py               # offline mixing from cached log-probs + EnsembleLM (full vocab)
    eval/score.py             # per-token log-probs per model per split -> .npy
    eval/report.py            # tables (markdown + CSV) and training-curve PNGs
  scripts/
    setup_pod.sh              # idempotent environment setup
    env.sh                    # env vars, sourced by every job
    sync.sh                   # rsync local -> pod
    launch.sh                 # start a job in tmux with logging
    status.sh                 # tail metrics, nvidia-smi, tmux ls
  tests/
    test_loader.py  test_merge.py  test_posterior.py  test_ensemble.py
```

Directory layout on the pod:

```
/workspace/btm/
  code/  venv/  hf_cache/  .env
  data/tok/<domain>/<split>/shard_00000.bin + meta.json
  data_smoke/...
  ckpt/seed/  ckpt/elm_python/  ckpt/elm_javascript/  ckpt/elm_law_init/  ckpt/elm_law/
  logs/<run>/metrics.jsonl  logs/<run>/stdout.log
  results/logprobs/<model>/<domain>_<split>.npy
  results/REPORT.md  results/*.csv  results/*.png
  artifacts/MANIFEST.json
```

---

## 7. Pod environment (`scripts/setup_pod.sh`)

The script must be idempotent. Pip installs on the container disk are lost on restart, so the venv lives on the volume.

```bash
set -euo pipefail
mkdir -p /workspace/btm/{code,data,data_smoke,ckpt,logs,results,artifacts,hf_cache}
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux rsync)
[ -d /workspace/btm/venv ] || python -m venv --system-site-packages /workspace/btm/venv  # reuses preinstalled torch
source /workspace/btm/venv/bin/activate
echo "torch==$(python -c 'import torch; print(torch.__version__)')" > /workspace/btm/constraints.txt
pip install -c /workspace/btm/constraints.txt -r /workspace/btm/code/requirements.txt
pip freeze > /workspace/btm/code/requirements.lock
```

`requirements.txt`: `transformers`, `datasets`, `huggingface_hub`, `safetensors`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`, `pytest`. Let pip choose versions that are compatible with torch 2.8, then commit `requirements.lock`.

`scripts/env.sh`:

```bash
source /workspace/btm/venv/bin/activate
set -a; source /workspace/btm/.env; set +a
export HF_HOME=/workspace/btm/hf_cache
export TOKENIZERS_PARALLELISM=false
```

Environment check (acceptance):

- `torch.cuda.is_available()`, the device name is A100, and `torch.cuda.is_bf16_supported()`.
- Load the seed with `attn_implementation="sdpa"` and run a bf16 forward pass on 2×2048 tokens.
- Save the seed to `ckpt/seed/`. Then confirm the tokenizer's EOS token and ID; it's expected to be `<|endoftext|>` with ID 0, but verify rather than assume.

`datasets` 4.x no longer supports dataset loading scripts. If a dataset depends on one, read its parquet or jsonl files directly (for example via `hf://datasets/...` paths) instead of downgrading.

---

## 8. Phase 1: Data

### 8.1 Sources

| Domain | Role | Source | Group key (for splitting) |
|---|---|---|---|
| python | ELM training | `bigcode/the-stack-dedup`, `data_dir="data/python"` (gated) | repo name |
| javascript | ELM training | `bigcode/the-stack-dedup`, `data_dir="data/javascript"` | repo name |
| law | ELM training | `free-law/Caselaw_Access_Project` (opinion text) | case ID |
| typescript | Held-out, near javascript | `bigcode/the-stack-dedup`, `data_dir="data/typescript"` | repo name |
| rust | Held-out, code but farther | `bigcode/the-stack-dedup`, `data_dir="data/rust"` | repo name |
| contracts | Held-out, near law | CUAD v1 full contract texts (~510 contracts). Use an HF mirror if it's script-free, otherwise the official release's `full_contract_txt` folder | contract file |
| web | Held-out, general text; also the forgetting monitor | `HuggingFaceFW/fineweb-edu`, config `sample-10BT` | document ID |

- **Verify each schema before writing the pipeline.** Print 3 records per source and check the field names (expected for the-stack-dedup: `content`, `max_stars_repo_name`, `avg_line_length`, `max_line_length`, `alphanum_fraction`, `size`).
- **Fallbacks:**
  - Code: `codeparrot/github-code-clean`, filtered on its language field. Its stream is not split by language, so this may require scanning far more data.
  - Law: ask the user.
- **Contamination note:** the seed was trained on the Pile, which includes GitHub and FreeLaw, so some test documents may already have been seen by the seed. All models share the seed, so this inflates absolute quality but barely affects comparisons between them. Document it in the report and don't try to fix it.

### 8.2 Filtering

- **Code:** keep a file only if `max_line_length <= 1000`, `avg_line_length <= 100`, `alphanum_fraction >= 0.25`, and its size is at most 1MB.
- **Law:** drop documents shorter than 500 characters, and collapse runs of whitespace.
- **All domains:** remove exact duplicates within a domain, using the sha1 of the normalized text.
- Log drop counts by reason.

### 8.3 Splits

| Domain | train | distill (reserved) | dev | test |
|---|---|---|---|---|
| python, javascript, law | 1.00B | 0.30B | 5M | 5M |
| typescript, rust, web | – | – | 1M | 3M |
| contracts | – | – | ~0.5M | the rest |

- **Split by group key so related documents stay together.** Compute `u = int(sha1(group_key)[:8], 16) / 2**32` and map `u` to a split, with probabilities proportional to the token targets. A repo, case or contract therefore lands in exactly one split, which prevents near-duplicate leakage.
- **Keep the assignment deterministic.** When a split has reached its target, drop further documents that hash into it rather than reassigning them. Stop streaming once all targets are met.
- **The `distill` split is tokenized now but must not be read by anything in this project.** It exists to guarantee a disjoint transfer set for future distillation work.

### 8.4 Tokenization and storage

- Append EOS (the ID verified in §7) after every document, then concatenate.
- Write flat `uint16` shards of about 100M tokens each to `data/tok/<domain>/<split>/shard_NNNNN.bin`.
- Write a `meta.json` per split containing: token count, document count, drop counts by reason, source and revision, filter parameters, tokenizer name, git SHA and timestamp.
- Tokenize in a multiprocessing pool of about 14 workers. Streaming download is the likely bottleneck.
- `--smoke` mode writes about 5M training tokens per domain, plus small dev and test splits, to `data_smoke/`. Build and validate this first, then run the full preparation in tmux while you build the trainer.

**Acceptance**

- Token counts are within ±1% of their targets; for contracts, report the actual counts.
- No group key appears in more than one split; check this explicitly.
- Decode 3 random 512-token windows per split, print them to the log, and eyeball them.
- Report disk usage (about 10GB expected).

---

## 9. Phase 2: Training harness and smoke test

`btm/train.py` is a single-GPU trainer. It is driven by a YAML config and resumable.

`configs/base_elm.yaml`:

```yaml
seq_len: 2048
global_batch_seqs: 256          # 524,288 tokens per optimizer step
micro_batch_seqs: 16            # set to the largest that fits (smoke test)
train_tokens: 1_000_000_000     # ≈1,907 optimizer steps
optimizer: adamw                # torch fused AdamW
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: 0.01              # Pythia's value; no decay on biases or norms
lr_peak: 3.0e-4                 # half of Pythia-160M's original 6e-4
lr_min: 3.0e-5
warmup_steps: 50
schedule: cosine
grad_clip: 1.0
precision: bf16_autocast        # fp32 master weights; TF32 matmuls allowed
attn_implementation: sdpa
compile: true                   # keep only if the smoke test shows a speedup and matching loss
data_seed: 1234                 # identical for all ELMs
eval_every_steps: 200           # dev loss: 64 windows of own domain + 64 windows of web
resume_ckpt_every_steps: 200    # one rolling checkpoint, written atomically (tmp file + rename)
snapshot_fractions: [0.25, 0.5, 0.75, 1.0]   # fp32 safetensors, HF format
```

Per-ELM configs override only `model_init`, `train_data` and `run_name`.

**Implementation notes**

- **Fresh optimizer for every ELM.** Pythia doesn't ship its optimizer state. The paper reused it, so this is a documented deviation.
- **Data order:** one pass over the train shards, as 2049-token windows at stride 2048 (input plus shifted target). Shuffle window indices with `data_seed`. The loader position goes into the resume checkpoint.
- **Resume checkpoint contents:** fp32 model, optimizer, scheduler, RNG states and loader position. On start, the trainer resumes automatically if a resume checkpoint exists.
- **Snapshots** at 25/50/75/100% of steps go to `ckpt/<run>/step_<N>/`, with the final one also at `ckpt/<run>/final/`. They are the "divergence dial" for later merge and distillation experiments.
- **Logging to `metrics.jsonl`:**
  - every step: step, tokens seen, learning rate, loss, gradient norm, tokens/s and GPU memory;
  - every eval: dev losses;
  - at run start: the full config, git SHA, and torch and transformers versions.
- Write a `DONE` marker file at the end of the run.

**Smoke test (acceptance)**

- **Short run:** 40 steps on the python smoke data. The loss stays finite and trends down.
- **Resume:** set `resume_ckpt_every_steps: 10`, kill the tmux session around step 25, then relaunch. Training must continue from step 20, and the loss at step 40 must be within 1% of an uninterrupted run.
- **Batch size:** find the largest micro-batch that fits (try 32, then 16, then 8).
- **Throughput:** measure tokens/s with and without `torch.compile`, and project the wall-clock time for 1B tokens.
- **Tests:** all unit tests pass (see §13.4).

**[GATE A]** Report the throughput, projected hours per ELM, data statistics and disk usage. Wait for approval.

---

## 10. Phase 3: Seed reference evaluation and scoring infrastructure

`btm/eval/score.py`:

- For a given model and split, cut the token stream into non-overlapping 1024-token windows.
- For each window, compute the log-prob of every actual next token (positions 1–1023).
- Save the result as float32 `.npy` of shape `[n_windows, 1023]` to `results/logprobs/<model>/<domain>_<split>.npy`, along with a meta file.

All single-model perplexities, ensembles (with any prior) and top-k variants are computed **offline** from these arrays, with no extra GPU passes. Perplexity is `exp(-mean(logprobs))` over every scored token in the split.

Run the scorer for the seed on every dev and test split of all 7 domains.

**Acceptance:** a seed perplexity table for all 7 domains, with finite values.

---

## 11. Phase 4: Train the python and javascript ELMs

- Both start from `ckpt/seed/` and share the same config apart from their data.
- Run them one after the other on the A100. If the user later adds a second pod on the same volume, they can run in parallel.
- After each run finishes, score its `final` checkpoint on every dev and test split.

**Acceptance**

- The `DONE` marker exists and all 4 snapshots are present.
- In-domain test perplexity is below the seed's.
- The change in web perplexity relative to the seed (forgetting) is logged. Some increase is expected.

---

## 12. Phase 5: Branch-from-average initialization and the law ELM

### 12.1 Posterior weights for initialization

- **Candidates:** `seed`, `elm_python/final`, `elm_javascript/final`.
- **Data:** the first N=100 windows (1024 tokens each) of the law dev split. Fix this choice and document it.
- **Compute:**
  - Score each window with each candidate: `LL[i, j] = Σ_t log p_j(x_t | x_<t)`.
  - Per-window posterior with a uniform prior: `post[i, :] = softmax_j(LL[i, :])`.
  - Weights: `w_j = mean_i post[i, j]`.
- **Expected outcome:** per-window posteriors will be close to one-hot, because likelihood ratios over 1024 tokens are huge, and most of the weight will land on the seed. That is expected, not a bug.
- **Note:** the paper's branching formula sums from i=0, which would include the seed, but its experiments only describe averaging existing ELMs. Including the seed is the user's design choice; record it as a deviation.

### 12.2 Build the initialization

- `θ_init = Σ_j w_j θ_j` over every floating-point parameter tensor, computed in fp32.
- Assert that non-float buffers are identical across the candidates, then copy them from the seed.
- Save the result to `ckpt/elm_law_init/` together with `weights.json`.

### 12.3 Initialization check

Compute law dev loss (on the full dev split) for:

- the seed;
- the uniform average of the 3 candidates;
- the posterior-weighted initialization.

**[GATE B]** Report the weights and the three losses. If the posterior-weighted initialization is more than 5% worse than the seed (dev NLL), flag it explicitly. Wait for approval.

### 12.4 Train

- Train the law ELM from `ckpt/elm_law_init/` with identical hyperparameters and the same 1.0B-token budget.
- Score its final checkpoint on every dev and test split, as in Phase 4.

---

## 13. Phase 6: Full evaluation

### 13.1 Inference modes

| ID | Mode | Description |
|---|---|---|
| M0 | Seed | Reference |
| M1–M3 | Single ELM | python, javascript, law; full cross-domain matrix |
| E1 | Ensemble, cached prior | 3 ELMs; prior estimated on the target domain's dev split |
| E2 | Ensemble, uniform prior | Uses no target-domain information (no label leakage) |
| E3 | Ensemble, top-k | k ∈ {1, 2}, chosen by cached prior; prior renormalized over the selected ELMs |
| A1 | Parameter average, uniform | 1/3 weight each |
| A2 | Parameter average, posterior | Weights = the target domain's cached prior; one averaged model per domain |

- **Cached prior** for each eval domain: use the first 100 dev windows of 1024 tokens and set `prior_j = mean_i softmax_j(LL[i, j])` over the 3 ELMs. This is the same procedure as §12.1.
- **Optional EMA prior:** the paper's EMA (λ=0.3) may be added behind a flag. Its printed formula is ambiguous, so document the exact update rule you implement.
- **A1 and A2 need real forward passes:** build the averaged model, then score it with `score.py`. Everything else is computed offline.

### 13.2 Ensemble math

Compute this per window, in log space, offline from the cached log-probs:

```
lp[j, t]     = log p_j(x_t | x_<t)                         t = 1..1023
cum[j, t]    = Σ_{s=1}^{t-1} lp[j, s]                      (cum[j, 1] = 0; resets every window)
logpost[j,t] = log prior_j + cum[j,t] − logsumexp_k(log prior_k + cum[k,t])
lp_ens[t]    = logsumexp_j(logpost[j,t] + lp[j,t])
PPL_ens      = exp(−mean(lp_ens))
```

### 13.3 Library for future distillation

`EnsembleLM` in `btm/ensemble.py` takes `input_ids` and returns full-vocabulary mixed log-distributions of shape `[B, T, V]`, using the same per-position posterior weights. It's the teacher interface for later knowledge-distillation work. Implement and test it now, but don't train anything with it.

### 13.4 Unit tests (CPU, tiny random models)

- **Posterior:** identical models give equal weights; a model with strictly higher likelihood moves toward weight 1.
- **Ensemble:** an ensemble of a model with itself equals that model. Gathering the actual-token entries from `EnsembleLM` reproduces the offline `lp_ens` to within 1e-4.
- **Merge:** averaging a model with itself returns the same model. Weights must sum to 1.
- **Loader:** shuffle order is deterministic, and resuming reproduces the exact sequence of batches.

### 13.5 Additional analysis

- **Mixing:** mean per-token posterior entropy per domain under E1, plus the distribution of final-position posteriors. This shows how much mixing actually happens; expect more on typescript, javascript and python than on law.
- **Domain distance:** the cross-perplexity matrix (M1–M3 × 7 domains) is the measure of distance between domains.

### 13.6 Report (`results/REPORT.md` plus CSVs)

1. Test perplexity table:
   - rows: M0, M1–M3, E1, E2, E3 (k=1, 2), A1, A2;
   - columns: the 7 domains, plus a macro-average over the in-domain sets (python, javascript, law) and a macro-average over the held-out sets.
2. Cached prior weights per domain.
3. The law initialization weights and the Gate B losses.
4. Posterior entropy per domain.
5. Training curves as PNGs: train loss, own-domain dev loss and web dev loss for each ELM.
6. The deviations from the paper (§15) and the contamination note.

**Sanity check:** flag it, without silently fixing anything, if E1 on any in-domain test set is more than about 1% worse than the matching single ELM.

**[GATE C]** Present the report.

---

## 14. Phase 7: Freeze the artifacts

- Write `artifacts/MANIFEST.json` listing:
  - every checkpoint path with its sha256;
  - every data `meta.json`;
  - all configs;
  - the git SHA;
  - `requirements.lock`;
  - the results files.
- Make `ckpt/` and `data/tok/` read-only (`chmod -R a-w`).
- Tag the git commit as `btm-baseline-v1`.
- Pull `results/` and `MANIFEST.json` back into the local repo.

---

## 15. Deviations from the paper (copy into the report)

- **Seed:** a public pretrained seed (Pythia-160M-deduped, about 300B tokens) replaces BTM's own seed phase. There is no seed-fraction knob, and the seed has seen Pile domains, including GitHub and FreeLaw.
- **Optimizer:** each ELM starts with fresh AdamW and its own warmup. The paper continued the seed's optimizer state.
- **Domains and iterations:** 3 domains and 2 BTM iterations. The seed is included as a candidate when averaging for branching.
- **Branching weights:** the mean of per-window posteriors (uniform prior) over 100 dev windows.
- **Eval data:** some evaluation data may overlap with the seed's pretraining data.
- **No compute-matched dense baseline** (out of scope).

---

## 16. Remote workflow cheat sheet

```bash
# sync code (local -> pod)
rsync -az --delete --exclude '.git' --exclude '__pycache__' ./ btm-pod:/workspace/btm/code/

# launch a detached job (create the log dir first)
ssh btm-pod "mkdir -p /workspace/btm/logs/elm_python && tmux new-session -d -s elm_python \
  'source /workspace/btm/code/scripts/env.sh && cd /workspace/btm/code && \
   python -m btm.train --config configs/elm_python.yaml >> /workspace/btm/logs/elm_python/stdout.log 2>&1'"

# monitor
ssh btm-pod "tail -n 3 /workspace/btm/logs/elm_python/metrics.jsonl; tmux ls; \
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"

# pull results (pod -> local)
rsync -az btm-pod:/workspace/btm/results/ ./results/
```

Poll at sensible intervals (every few minutes) rather than in tight loops.

---

## 17. Budget estimate (A100 SXM; refine at Gate A)

| Phase | Estimate |
|---|---|
| 1. Data preparation | 3–6 h wall-clock (CPU- and network-bound; the GPU sits idle) |
| 2. Smoke test | < 1 GPU-h |
| 3. Seed scoring | < 0.5 GPU-h |
| 4. python + javascript ELMs | ~5–8 GPU-h |
| 5. law ELM | ~3–4 GPU-h |
| 6. Evaluation | ~1 GPU-h |
| **Total** | **~12–20 pod-hours** |

Storage: about 8GB of tokenized data, about 15GB of checkpoints, under 1GB of cached log-probs and a few GB of HF cache. That totals about 30GB of the 200GB volume.

---

## 18. Future hooks (not now)

- **Distillation:** use the reserved `distill` splits, with `EnsembleLM` as the teacher. Candidate students are Pythia-70M (same tokenizer) or a 160M model initialized from the seed.
- **Divergence dial:** merge or distill from the 25/50/75% snapshots.
- **Earlier branch point:** branch from an earlier Pythia checkpoint (around step 71000) to get more divergent ELMs.
- **Law-from-seed ablation:** train the law ELM from the seed to measure what branching from the average buys.
- **Controls:** a dense multi-domain baseline and a random-split ensemble.
