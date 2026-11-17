import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from btm.utils import git_sha, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--out", default="/workspace/btm/ckpt/seed")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name()
    assert torch.cuda.is_bf16_supported(), "bf16 not supported"

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32, attn_implementation="sdpa"
    ).cuda().eval()
    x = torch.randint(0, len(tok), (2, 2048), device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    assert torch.isfinite(logits).all(), "non-finite logits"
    assert len(tok) < 2**16, f"tokenizer has {len(tok)} tokens; does not fit uint16"

    info = {
        "device": name,
        "torch": torch.__version__,
        "logits_shape": list(logits.shape),
        "model": args.model,
        "revision": args.revision,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "tokenizer_len": len(tok),
        "model_vocab_size": model.config.vocab_size,
        "git_sha": git_sha(),
    }
    print(json.dumps(info, indent=2))
    if tok.eos_token != "<|endoftext|>" or tok.eos_token_id != 0:
        print("WARNING: EOS differs from the expected <|endoftext|> / 0")

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"{out} already exists; not overwriting")
        return
    model.save_pretrained(out)
    tok.save_pretrained(out)
    write_json(out / "seed_meta.json", info)
    print(f"saved seed to {out}")


if __name__ == "__main__":
    main()
