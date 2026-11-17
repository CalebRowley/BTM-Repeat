import numpy as np
import pytest

from btm.data.loader import TokenSplit, TrainLoader


@pytest.fixture
def split_dir(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (np.arange(10_000) % 50_000).astype(np.uint16).tofile(d / "shard_00000.bin")
    (np.arange(10_000, 17_000) % 50_000).astype(np.uint16).tofile(d / "shard_00001.bin")
    return d


def batches(loader, n, size=4):
    return [loader.next_batch(size).numpy() for _ in range(n)]


def test_shuffle_is_deterministic(split_dir):
    a = batches(TrainLoader(TokenSplit(split_dir), 33, 32, seed=1), 10)
    b = batches(TrainLoader(TokenSplit(split_dir), 33, 32, seed=1), 10)
    c = batches(TrainLoader(TokenSplit(split_dir), 33, 32, seed=2), 10)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_resume_reproduces_batches(split_dir):
    loader = TrainLoader(TokenSplit(split_dir), 33, 32, seed=1)
    batches(loader, 5)
    state = loader.state_dict()
    expected = batches(loader, 5)
    resumed = TrainLoader(TokenSplit(split_dir), 33, 32, seed=1)
    resumed.load_state_dict(state)
    assert all(np.array_equal(x, y) for x, y in zip(expected, batches(resumed, 5)))


def test_windows_are_contiguous_and_within_shards(split_dir):
    loader = TrainLoader(TokenSplit(split_dir), 33, 32, seed=1)
    assert loader.n_windows == (10_000 - 33) // 32 + 1 + (7_000 - 33) // 32 + 1
    for row in loader.next_batch(64).numpy():
        assert np.array_equal(np.diff(row.astype(np.int64)), np.ones(32))


def test_one_pass_then_exhausted(split_dir):
    loader = TrainLoader(TokenSplit(split_dir), 33, 32, seed=1)
    seen = np.concatenate([loader.next_batch(1).numpy()[:, 0] for _ in range(loader.n_windows)])
    assert len(np.unique(seen)) == loader.n_windows
    with pytest.raises(StopIteration):
        loader.next_batch(1)


def test_eval_windows_and_distill_guard(split_dir, tmp_path):
    w = TokenSplit(split_dir).eval_windows(1024)
    assert w.shape == (10_000 // 1024 + 7_000 // 1024, 1024)
    distill = tmp_path / "distill"
    distill.mkdir()
    np.zeros(10, np.uint16).tofile(distill / "shard_00000.bin")
    with pytest.raises(PermissionError):
        TokenSplit(distill)
