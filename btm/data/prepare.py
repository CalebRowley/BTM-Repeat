import argparse
import bisect
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from btm.data.loader import TokenSplit
from btm.utils import git_sha, load_config, write_json

_TOK = None
_EOS = None
_HSPACE = re.compile(r"[^\S\n]+")
_NEWLINES = re.compile(r"\s*\n\s*")


def _init_worker(name, revision):
    global _TOK, _EOS
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(name, revision=revision)
    _TOK.model_max_length = 10**12
    _EOS = _TOK.eos_token_id


def _tokenize(batch):
    ids = _TOK([text for _, _, text in batch], add_special_tokens=False)["input_ids"]
    return [(split, gid, np.asarray(x + [_EOS], dtype=np.uint16)) for (split, gid, _), x in zip(batch, ids)]


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def collapse_whitespace(text):
    return _NEWLINES.sub("\n", _HSPACE.sub(" ", text)).strip()


def text_hash(text):
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).digest()


def group_u(key):
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def group_id(key):
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:16], 16)


def code_drop_reason(rec, f):
    if rec["max_line_length"] > f["max_line_length"]:
        return "max_line_length"
    if rec["avg_line_length"] > f["avg_line_length"]:
        return "avg_line_length"
    if rec["alphanum_fraction"] < f["alphanum_fraction"]:
        return "alphanum_fraction"
    if rec["size"] > f["max_size_bytes"]:
        return "size"
    return None


def clean(rec, dcfg, filters):
    text = rec.get(dcfg["text_field"]) or ""
    kind = dcfg.get("filter", "none")
    if kind == "code":
        reason = code_drop_reason(rec, filters["code"])
        if reason:
            return None, reason
    elif kind == "law":
        text = collapse_whitespace(text)
        if len(text) < filters["law"]["min_chars"]:
            return None, "min_chars"
    if not text.strip():
        return None, "empty"
    return text, None


class SplitPlan:
    def __init__(self, targets, weights=None):
        self.names = list(targets)
        w = weights or targets
        if any(w[n] is None for n in self.names):
            raise ValueError("splits without a token target need explicit weights")
        total = sum(w[n] for n in self.names)
        self.edges = list(itertools.accumulate(w[n] / total for n in self.names))[:-1]

    def assign(self, u):
        return self.names[bisect.bisect_right(self.edges, u)]


def resolve_targets(dcfg, smoke, smoke_targets):
    targets = dict(dcfg["targets"])
    if smoke:
        targets = {
            s: smoke_targets[s] if t is None else min(t, smoke_targets[s])
            for s, t in targets.items()
            if s in smoke_targets
        }
    weights = dcfg.get("weights")
    if weights:
        weights = {s: weights[s] for s in targets}
    return targets, weights


def resolve_revision(dcfg):
    if dcfg["loader"] == "local_files":
        h = hashlib.sha256()
        root = Path(dcfg["root"])
        for path in sorted(root.glob(dcfg["glob"])):
            h.update(str(path.relative_to(root)).encode("utf-8"))
            h.update(path.read_bytes())
        return "sha256:" + h.hexdigest()
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dcfg["source"]).sha


def iter_records(dcfg, revision):
    if dcfg["loader"] == "hf_stream":
        from datasets import load_dataset

        kwargs = {"split": "train", "streaming": True, "revision": revision}
        if dcfg.get("data_dir"):
            kwargs["data_dir"] = dcfg["data_dir"]
        yield from load_dataset(dcfg["source"], dcfg.get("config"), **kwargs)
    elif dcfg["loader"] == "hf_files":
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        root = f"datasets/{dcfg['source']}@{revision}"
        for path in sorted(fs.glob(f"{root}/{dcfg['glob']}")):
            rel = path.split(f"@{revision}/", 1)[-1]
            yield {"text": fs.read_text(path, encoding="utf-8", errors="replace"), "path": rel}
    elif dcfg["loader"] == "local_files":
        root = Path(dcfg["root"])
        for path in sorted(root.glob(dcfg["glob"])):
            yield {"text": path.read_text(encoding="utf-8", errors="replace"), "path": str(path.relative_to(root))}
    else:
        raise ValueError(f"unknown loader {dcfg['loader']}")


def required_fields(dcfg):
    return list(dict.fromkeys(dcfg.get("expected_fields", []) + [dcfg["text_field"], dcfg["group_field"]]))


class ShardWriter:
    def __init__(self, directory, shard_tokens):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.shards = []
        self.tokens = 0
        self.docs = 0
        self._f = None
        self._cur = 0

    def write(self, ids):
        if self._f is None or self._cur >= self.shard_tokens:
            self._roll()
        self._f.write(ids.tobytes())
        self._cur += len(ids)
        self.tokens += len(ids)
        self.docs += 1

    def _roll(self):
        self._finish()
        name = f"shard_{len(self.shards):05d}.bin"
        self.shards.append({"file": name, "tokens": 0})
        self._f = open(self.dir / name, "wb")
        self._cur = 0

    def _finish(self):
        if self._f is not None:
            self._f.close()
            self.shards[-1]["tokens"] = self._cur
            self._f = None

    def close(self):
        self._finish()


def process_domain(name, cfg, out_root, smoke, tok_meta):
    dcfg = cfg["domains"][name]
    dom_dir = out_root / name
    if dom_dir.exists() and any(dom_dir.iterdir()):
        raise SystemExit(f"{dom_dir} already exists; refusing to overwrite")
    targets, weights = resolve_targets(dcfg, smoke, cfg["smoke_targets"])
    plan = SplitPlan(targets, weights)
    revision = resolve_revision(dcfg)
    filters = cfg["filters"]
    fields = required_fields(dcfg)
    writers = {s: ShardWriter(dom_dir / s, int(cfg["shard_tokens"])) for s in targets}
    groups = {s: set() for s in targets}
    stats = Counter()
    seen = set()

    def full(s):
        return targets[s] is not None and writers[s].tokens >= targets[s]

    def all_full():
        return all(full(s) for s in targets)

    def candidates():
        for i, rec in enumerate(iter_records(dcfg, revision)):
            if i == 0:
                missing = [f for f in fields if f not in rec]
                if missing:
                    raise SystemExit(f"{name}: schema mismatch, missing fields {missing}; got {sorted(rec)}")
            stats["seen"] += 1
            text, reason = clean(rec, dcfg, filters)
            if reason:
                stats[f"drop_{reason}"] += 1
                continue
            h = text_hash(text)
            if h in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(h)
            key = rec.get(dcfg["group_field"])
            if not key:
                stats["missing_group_key"] += 1
                key = "doc:" + h.hex()
            split = plan.assign(group_u(str(key)))
            if full(split):
                stats["drop_split_full"] += 1
                continue
            yield split, group_id(str(key)), text
            if all_full():
                return

    def handle(results):
        for split, gid, ids in results:
            if full(split):
                stats["drop_split_full"] += 1
                continue
            writers[split].write(ids)
            groups[split].add(gid)
            stats["kept"] += 1

    start = last = time.time()
    max_inflight = 4 * int(cfg["num_workers"])
    print(json.dumps({"event": "start", "domain": name, "source": dcfg["source"], "revision": revision, "targets": targets}), flush=True)
    with Pool(int(cfg["num_workers"]), initializer=_init_worker, initargs=(cfg["tokenizer"], cfg["tokenizer_revision"])) as pool:
        pending = deque()
        for batch in batched(candidates(), int(cfg["batch_docs"])):
            pending.append(pool.apply_async(_tokenize, (batch,)))
            while len(pending) >= max_inflight or (pending and pending[0].ready()):
                handle(pending.popleft().get())
            if time.time() - last > cfg["progress_every_s"]:
                last = time.time()
                print(json.dumps({"event": "progress", "domain": name, "elapsed_s": round(last - start),
                                  "tokens": {s: w.tokens for s, w in writers.items()}, "stats": dict(stats)}), flush=True)
            if all_full():
                break
        while pending:
            handle(pending.popleft().get())

    for s, w in writers.items():
        w.close()
        np.save(w.dir / "groups.npy", np.array(sorted(groups[s]), dtype=np.uint64))
        write_json(w.dir / "meta.json", {
            "domain": name,
            "split": s,
            "tokens": w.tokens,
            "documents": w.docs,
            "target": targets[s],
            "target_met": targets[s] is None or w.tokens >= targets[s],
            "shards": w.shards,
            "drops": dict(stats),
            "source": dcfg["source"],
            "revision": revision,
            "data_dir": dcfg.get("data_dir") or dcfg.get("root"),
            "config": dcfg.get("config"),
            "filter": dcfg.get("filter", "none"),
            "filter_params": filters.get(dcfg.get("filter")),
            "split_probs": dict(zip(plan.names, np.diff([0.0, *plan.edges, 1.0]).tolist())),
            **tok_meta,
            "smoke": smoke,
            "git_sha": git_sha(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
    print(json.dumps({"event": "done", "domain": name, "elapsed_s": round(time.time() - start),
                      "tokens": {s: w.tokens for s, w in writers.items()}, "stats": dict(stats)}), flush=True)


def inspect(cfg, domains):
    ok = True
    for name in domains:
        dcfg = cfg["domains"][name]
        revision = resolve_revision(dcfg)
        print(f"== {name}: {dcfg['source']} @ {revision}", flush=True)
        records = list(itertools.islice(iter_records(dcfg, revision), 3))
        for rec in records:
            print({k: (v[:200] + "...") if isinstance(v, str) and len(v) > 200 else v for k, v in rec.items()}, flush=True)
        present = set(records[0]) if records else set()
        missing = [f for f in required_fields(dcfg) if f not in present]
        if missing:
            ok = False
            print(f"!! {name}: missing fields {missing}; got {sorted(present)}", flush=True)
    return ok


def verify(cfg, out_root, smoke, domains, tok):
    ok = True
    rng = np.random.default_rng(0)
    for name in domains:
        targets, _ = resolve_targets(cfg["domains"][name], smoke, cfg["smoke_targets"])
        print(f"== {name}", flush=True)
        for s, t in targets.items():
            meta = json.loads((out_root / name / s / "meta.json").read_text())
            if t is None:
                status = "report"
            else:
                status = "ok" if abs(meta["tokens"] / t - 1) <= 0.01 else "FAIL"
                ok &= status == "ok"
            print(f"{s}: tokens={meta['tokens']:,} docs={meta['documents']:,} target={t} [{status}]", flush=True)
        gids = {s: np.load(out_root / name / s / "groups.npy") for s in targets}
        disjoint = True
        for a, b in itertools.combinations(targets, 2):
            shared = len(np.intersect1d(gids[a], gids[b]))
            if shared:
                disjoint = False
                print(f"FAIL: {shared} group keys shared between {a} and {b}", flush=True)
        ok &= disjoint
        if disjoint:
            print("group keys disjoint across splits", flush=True)
        for s in targets:
            if s == "distill":
                continue
            split = TokenSplit(out_root / name / s)
            for _ in range(3):
                si = int(rng.integers(len(split.shards)))
                off = int(rng.integers(max(1, len(split.shards[si]) - 512)))
                print(f"--- {name}/{s} shard {si} offset {off}\n{tok.decode(split.read(si, off, 512).tolist())}\n", flush=True)
    total = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    print(f"disk usage of {out_root}: {total / 1e9:.2f} GB", flush=True)
    return ok


def hard_exit(code):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    domains = args.domains or list(cfg["domains"])
    out_root = Path(cfg["smoke_out_dir"] if args.smoke else cfg["out_dir"])

    if args.inspect:
        hard_exit(0 if inspect(cfg, domains) else 1)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], revision=cfg["tokenizer_revision"])
    if len(tok) >= 2**16:
        raise SystemExit(f"tokenizer has {len(tok)} tokens; does not fit uint16")

    if args.verify:
        sys.exit(0 if verify(cfg, out_root, args.smoke, domains, tok) else 1)

    if len(domains) > 1:
        for name in domains:
            cmd = [sys.executable, "-m", "btm.data.prepare", "--config", args.config, "--domains", name]
            cmd += ["--smoke"] if args.smoke else []
            for item in args.set:
                cmd += ["--set", item]
            subprocess.run(cmd, check=True)
        return

    tok_meta = {
        "tokenizer": cfg["tokenizer"],
        "tokenizer_revision": cfg["tokenizer_revision"],
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
    }
    print(json.dumps({"event": "tokenizer", **tok_meta, "vocab": len(tok)}), flush=True)
    for name in domains:
        process_domain(name, cfg, out_root, args.smoke, tok_meta)
    hard_exit(0)


if __name__ == "__main__":
    main()
