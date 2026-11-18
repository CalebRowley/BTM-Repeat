import argparse
import hashlib
import json
import time
from pathlib import Path

from btm.utils import git_sha, write_json


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def files_with_hashes(root, skip=()):
    root = Path(root)
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in skip
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/btm")
    ap.add_argument("--out", default="/workspace/btm/artifacts/MANIFEST.json")
    args = ap.parse_args()
    root = Path(args.root)
    code = root / "code"
    manifest = {
        "git_sha": git_sha(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoints": files_with_hashes(root / "ckpt", skip={"resume.pt"}),
        "data_meta": {
            str(p.relative_to(root / "data" / "tok")): json.loads(p.read_text())
            for p in sorted((root / "data" / "tok").rglob("meta.json"))
        },
        "configs": {p.name: p.read_text() for p in sorted((code / "configs").glob("*.yaml"))},
        "requirements_lock": (code / "requirements.lock").read_text() if (code / "requirements.lock").exists() else None,
        "results": files_with_hashes(root / "results"),
    }
    write_json(args.out, manifest)
    print(f"wrote {args.out}: {len(manifest['checkpoints'])} checkpoint files, {len(manifest['results'])} result files")


if __name__ == "__main__":
    main()
