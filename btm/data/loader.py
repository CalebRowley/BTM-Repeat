from pathlib import Path

import numpy as np
import torch


class TokenSplit:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.name == "distill":
            raise PermissionError("the distill split is reserved and must not be read in this project")
        files = sorted(self.path.glob("shard_*.bin"))
        if not files:
            raise FileNotFoundError(f"no shards in {self.path}")
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]

    @property
    def n_tokens(self):
        return sum(len(s) for s in self.shards)

    def window_index(self, length, stride):
        rows = []
        for si, shard in enumerate(self.shards):
            if len(shard) < length:
                continue
            n = (len(shard) - length) // stride + 1
            offsets = np.arange(n, dtype=np.int64) * stride
            rows.append(np.stack([np.full(n, si, dtype=np.int64), offsets], 1))
        if not rows:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(rows)

    def read(self, shard, offset, length):
        return np.asarray(self.shards[shard][offset : offset + length])

    def eval_windows(self, length=1024, limit=None):
        index = self.window_index(length, length)
        if limit is not None:
            index = index[:limit]
        if len(index) == 0:
            return np.zeros((0, length), dtype=np.uint16)
        return np.stack([self.read(s, o, length) for s, o in index])


class TrainLoader:
    def __init__(self, split, window, stride, seed):
        self.split = split
        self.window = window
        self.seed = seed
        self.index = split.window_index(window, stride)
        self.order = np.random.default_rng(seed).permutation(len(self.index))
        self.pos = 0

    @property
    def n_windows(self):
        return len(self.index)

    def next_batch(self, n):
        if self.pos + n > len(self.order):
            raise StopIteration("train split exhausted")
        rows = [self.read_window(i) for i in self.order[self.pos : self.pos + n]]
        self.pos += n
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    def read_window(self, i):
        shard, offset = self.index[i]
        return self.split.read(shard, offset, self.window)

    def state_dict(self):
        return {"pos": self.pos, "seed": self.seed, "n_windows": self.n_windows, "window": self.window}

    def load_state_dict(self, state):
        if state["seed"] != self.seed or state["n_windows"] != self.n_windows or state["window"] != self.window:
            raise ValueError(f"loader state mismatch: {state} vs {self.state_dict()}")
        self.pos = state["pos"]
