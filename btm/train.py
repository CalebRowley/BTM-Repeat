import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from btm.data.loader import TokenSplit, TrainLoader
from btm.utils import JsonlLogger, git_sha, load_config


def lr_lambda(cfg, total_steps):
    peak, floor, warmup = float(cfg["lr_peak"]), float(cfg["lr_min"]), int(cfg["warmup_steps"])

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return (floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))) / peak

    return f


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lm_loss(model, batch):
    logits = model(input_ids=batch[:, :-1]).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))


def autocast(device, cfg):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16_autocast")


@torch.no_grad()
def dev_loss(model, windows, batch_size, device, cfg):
    model.eval()
    total = 0.0
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size].astype(np.int64)).to(device)
        with autocast(device, cfg):
            total += lm_loss(model, batch).item() * len(batch)
    model.train()
    return total / len(windows)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_resume(path, model, opt, sched, loader, step):
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "loader": loader.state_dict(),
        "rng": rng_state(),
    }, tmp)
    os.replace(tmp, path)


def save_hf(model, tokenizer_src, out_dir):
    if out_dir.exists():
        return False
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(tokenizer_src).save_pretrained(tmp)
    tmp.rename(out_dir)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    run = cfg["run_name"]
    ckpt_dir = Path(cfg["ckpt_root"]) / run
    log_dir = Path(cfg["log_root"]) / run
    if (ckpt_dir / "DONE").exists():
        print(f"{run} already DONE; nothing to do", flush=True)
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(log_dir / "metrics.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seq_len = int(cfg["seq_len"])
    global_batch = int(cfg["global_batch_seqs"])
    micro_batch = int(cfg["micro_batch_seqs"])
    if global_batch % micro_batch:
        raise ValueError("global_batch_seqs must be divisible by micro_batch_seqs")
    accum = global_batch // micro_batch
    tokens_per_step = global_batch * seq_len
    total_steps = int(cfg["train_tokens"]) // tokens_per_step

    train_data = Path(cfg["train_data"])
    loader = TrainLoader(TokenSplit(train_data), seq_len + 1, seq_len, int(cfg["data_seed"]))
    if loader.n_windows < total_steps * global_batch:
        raise SystemExit(f"train split has {loader.n_windows} windows, need {total_steps * global_batch}")
    eval_len, n_eval = int(cfg["eval_window_len"]), int(cfg["eval_windows"])
    dev_sets = {
        "own": TokenSplit(train_data.parent / "dev").eval_windows(eval_len, n_eval),
        "web": TokenSplit(train_data.parent.parent / "web" / "dev").eval_windows(eval_len, n_eval),
    }

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_init"], dtype=torch.float32, attn_implementation=cfg["attn_implementation"]
    ).to(device)
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(
        param_groups(model, float(cfg["weight_decay"])),
        lr=float(cfg["lr_peak"]),
        betas=tuple(float(b) for b in cfg["betas"]),
        eps=float(cfg["eps"]),
        fused=device.type == "cuda",
    )
    sched = LambdaLR(opt, lr_lambda(cfg, total_steps))

    step = 0
    resume_path = ckpt_dir / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        loader.load_state_dict(state["loader"])
        set_rng_state(state["rng"])
        step = state["step"]
        logger.log({"event": "resume", "step": step, "time": time.time()})
    else:
        random.seed(int(cfg["data_seed"]))
        np.random.seed(int(cfg["data_seed"]))
        torch.manual_seed(int(cfg["data_seed"]))

    logger.log({
        "event": "start",
        "step": step,
        "config": cfg,
        "git_sha": git_sha(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "total_steps": total_steps,
        "accum_steps": accum,
        "train_windows": loader.n_windows,
        "time": time.time(),
    })

    fwd = torch.compile(model) if cfg["compile"] else model
    snap_steps = {max(1, round(f * total_steps)) for f in cfg["snapshot_fractions"]}

    def run_eval():
        rec = {"event": "eval", "step": step}
        for name, windows in dev_sets.items():
            rec[f"dev_loss_{name}"] = dev_loss(model, windows, micro_batch, device, cfg)
        logger.log(rec)
        print(rec, flush=True)

    if step == 0:
        run_eval()

    t_last = time.perf_counter()
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(accum):
            batch = loader.next_batch(micro_batch).to(device)
            with autocast(device, cfg):
                loss = lm_loss(fwd, batch)
            (loss / accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        step += 1

        loss_val = (loss_sum / accum).item()
        now = time.perf_counter()
        rec = {
            "event": "train",
            "step": step,
            "tokens": step * tokens_per_step,
            "lr": lr,
            "loss": loss_val,
            "grad_norm": grad_norm.item(),
            "tokens_per_s": tokens_per_step / (now - t_last),
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        }
        logger.log(rec)
        if step % 10 == 0 or step == 1:
            print(rec, flush=True)
        if not math.isfinite(loss_val):
            raise SystemExit(f"non-finite loss at step {step}")

        if step % int(cfg["eval_every_steps"]) == 0 or step == total_steps:
            run_eval()
        if step in snap_steps:
            saved = save_hf(model, cfg["model_init"], ckpt_dir / f"step_{step}")
            logger.log({"event": "snapshot", "step": step, "saved": saved})
        if step % int(cfg["resume_ckpt_every_steps"]) == 0 and step < total_steps:
            save_resume(resume_path, model, opt, sched, loader, step)
            logger.log({"event": "resume_ckpt", "step": step})
        t_last = time.perf_counter()

    save_hf(model, cfg["model_init"], ckpt_dir / "final")
    (ckpt_dir / "DONE").write_text(f"{git_sha()}\n{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    if resume_path.exists():
        resume_path.unlink()
    logger.log({"event": "done", "step": step, "time": time.time()})
    logger.close()


if __name__ == "__main__":
    main()
