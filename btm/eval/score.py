import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from btm.data.loader import TokenSplit
from btm.ensemble import perplexity
from btm.posterior import token_logprobs
from btm.utils import git_sha, load_config, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    domains = args.domains or cfg["domains"]
    splits = args.splits or cfg["splits"]
    if "distill" in splits:
        raise SystemExit("the distill split is reserved and must not be scored")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=cfg["attn_implementation"]
    ).to(device).eval()
    amp_dtype = torch.bfloat16 if cfg["dtype"] == "bf16" else None
    out_dir = Path(cfg["logprob_root"]) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        for split in splits:
            out = out_dir / f"{domain}_{split}.npy"
            if out.exists() and not args.overwrite:
                print(f"skip {out} (exists)", flush=True)
                continue
            t0 = time.time()
            windows = TokenSplit(Path(cfg["data_root"]) / domain / split).eval_windows(int(cfg["window"]))
            lp = token_logprobs(model, windows, int(cfg["batch_size"]), amp_dtype)
            tmp = out.with_name(out.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, lp)
            os.replace(tmp, out)
            meta = {
                "model": args.model,
                "name": args.name,
                "domain": domain,
                "split": split,
                "window": int(cfg["window"]),
                "n_windows": int(lp.shape[0]),
                "n_tokens_scored": int(lp.size),
                "ppl": perplexity(lp),
                "dtype": cfg["dtype"],
                "seconds": round(time.time() - t0, 1),
                "git_sha": git_sha(),
            }
            write_json(out.with_suffix(".json"), meta)
            print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
