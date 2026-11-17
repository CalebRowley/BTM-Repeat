import argparse
import json
from pathlib import Path

import numpy as np
import torch

from btm.utils import git_sha, load_config, write_json


@torch.no_grad()
def token_logprobs(model, windows, batch_size=16, amp_dtype=None):
    device = next(model.parameters()).device
    windows = np.asarray(windows)
    out = np.empty((len(windows), windows.shape[1] - 1), dtype=np.float32)
    for i in range(0, len(windows), batch_size):
        x = torch.as_tensor(windows[i : i + batch_size].astype(np.int64), device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            logits = model(input_ids=x[:, :-1]).logits
        lp = torch.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        out[i : i + batch_size] = lp.cpu().numpy()
    return out


def sequence_loglik(lp):
    return np.asarray(lp, dtype=np.float64).sum(-1)


def window_posteriors(LL, prior=None):
    LL = np.asarray(LL, dtype=np.float64)
    logits = LL if prior is None else LL + np.log(np.asarray(prior, dtype=np.float64))
    logits = logits - logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def cached_prior(LL):
    return window_posteriors(LL).mean(0)


def loglik_from_files(paths, n_windows):
    cols = []
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        if len(arr) < n_windows:
            raise ValueError(f"{p} has {len(arr)} windows, need {n_windows}")
        cols.append(sequence_loglik(arr[:n_windows]))
    return np.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-windows", type=int)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    n = args.n_windows or cfg["prior_windows"]
    names, paths = zip(*(c.split("=", 1) for c in args.candidates))
    root = Path(cfg["logprob_root"])
    LL = loglik_from_files([root / name / f"{args.domain}_{args.split}.npy" for name in names], n)
    post = window_posteriors(LL)
    weights = post.mean(0)

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")
    write_json(out, {
        "candidates": list(names),
        "paths": list(paths),
        "weights": weights.tolist(),
        "domain": args.domain,
        "split": args.split,
        "n_windows": n,
        "window": cfg["window"],
        "prior": "uniform",
        "per_window_posteriors": post.tolist(),
        "git_sha": git_sha(),
    })
    print(json.dumps(dict(zip(names, weights.tolist()))))


if __name__ == "__main__":
    main()
