import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from btm.utils import git_sha, write_json


def check_weights(weights, n):
    weights = [float(w) for w in weights]
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError(f"negative weight in {weights}")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")
    return weights


def average_state_dicts(state_dicts, weights):
    weights = check_weights(weights, len(state_dicts))
    keys = list(state_dicts[0])
    for sd in state_dicts[1:]:
        if set(sd) != set(keys):
            raise ValueError("state dicts have different keys")
    out = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts]
        if tensors[0].is_floating_point():
            acc = torch.zeros_like(tensors[0], dtype=torch.float32)
            for w, t in zip(weights, tensors):
                acc += w * t.float()
            out[k] = acc
        else:
            if not all(torch.equal(tensors[0], t) for t in tensors[1:]):
                raise ValueError(f"non-float tensor {k} differs across models")
            out[k] = tensors[0].clone()
    return out


def merge_checkpoints(paths, weights):
    models = [AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32) for p in paths]
    merged = average_state_dicts([m.state_dict() for m in models], weights)
    base = models[0]
    base.load_state_dict(merged)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--weights-json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    source = None
    if args.weights_json:
        source = json.loads(Path(args.weights_json).read_text())
        models, weights = source["paths"], source["weights"]
    else:
        models = args.models
        weights = args.weights or [1.0 / len(models)] * len(models)
    weights = check_weights(weights, len(models))

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)

    model = merge_checkpoints(models, weights)
    model.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(models[0]).save_pretrained(tmp)
    write_json(tmp / "weights.json", {"paths": models, "weights": weights, "source": source, "git_sha": git_sha()})
    tmp.rename(out)
    print(json.dumps({"out": str(out), "paths": models, "weights": weights}))


if __name__ == "__main__":
    main()
