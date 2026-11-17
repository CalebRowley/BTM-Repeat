import pytest
import torch

from btm.merge import average_state_dicts, merge_checkpoints
from conftest import VOCAB


def test_self_average_is_identity(tiny_model):
    sd = tiny_model(0).state_dict()
    avg = average_state_dicts([sd, sd, sd], [0.2, 0.3, 0.5])
    for k in sd:
        assert torch.allclose(avg[k], sd[k].float(), atol=1e-7), k


def test_weighted_average_values(tiny_model):
    a, b = tiny_model(0).state_dict(), tiny_model(1).state_dict()
    avg = average_state_dicts([a, b], [0.25, 0.75])
    for k in a:
        if a[k].is_floating_point():
            assert torch.allclose(avg[k], 0.25 * a[k] + 0.75 * b[k], atol=1e-6), k


def test_weights_must_sum_to_one(tiny_model):
    sd = tiny_model(0).state_dict()
    with pytest.raises(ValueError):
        average_state_dicts([sd, sd], [0.5, 0.6])
    with pytest.raises(ValueError):
        average_state_dicts([sd, sd], [1.5, -0.5])


def test_non_float_tensors_must_match():
    a = {"w": torch.ones(2), "ids": torch.tensor([1, 2])}
    b = {"w": torch.zeros(2), "ids": torch.tensor([1, 3])}
    with pytest.raises(ValueError):
        average_state_dicts([a, b], [0.5, 0.5])


def test_merge_checkpoints_roundtrip(tiny_model, tmp_path):
    m = tiny_model(0)
    m.save_pretrained(tmp_path / "m")
    merged = merge_checkpoints([tmp_path / "m", tmp_path / "m"], [0.5, 0.5]).eval()
    x = torch.randint(0, VOCAB, (2, 16))
    with torch.no_grad():
        assert torch.allclose(merged(input_ids=x).logits, m(input_ids=x).logits, atol=1e-6)
