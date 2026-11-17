import json
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        cfg = merged
    for item in overrides or ():
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


def git_sha():
    sha_file = REPO_ROOT / "GIT_SHA"
    if sha_file.exists():
        return sha_file.read_text().strip()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def log(self, record):
        self._f.write(json.dumps(record, default=str) + "\n")

    def close(self):
        self._f.close()
