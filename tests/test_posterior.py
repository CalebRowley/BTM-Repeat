import numpy as np
import torch

from btm.posterior import cached_prior, sequence_loglik, token_logprobs, window_posteriors
from conftest import VOCAB


def test_token_logprobs_match_hf_loss(tiny_model):
    m = tiny_model(0)
    x = torch.randint(0, VOCAB, (3, 17))
    lp = token_logprobs(m, x.numpy(), batch_size=2)
    assert lp.shape == (3, 16)
    with torch.no_grad():
        loss = m(input_ids=x, labels=x).loss.item()
    assert abs(-lp.mean() - loss) < 1e-5


def test_identical_models_get_equal_weights(tiny_model):
    m = tiny_model(0)
    x = torch.randint(0, VOCAB, (5, 17)).numpy()
    ll = sequence_loglik(token_logprobs(m, x))
    prior = cached_prior(np.stack([ll, ll, ll], 1))
    assert np.allclose(prior, 1 / 3)


def test_higher_likelihood_moves_toward_one():
    rng = np.random.default_rng(0)
    base = rng.normal(-100, 5, size=20)
    weights = [cached_prior(np.stack([base + gap, base], 1))[0] for gap in (0.0, 0.1, 1.0, 10.0, 100.0)]
    assert weights[0] == 0.5
    assert all(b > a for a, b in zip(weights, weights[1:]))
    assert weights[-1] > 0.999


def test_posteriors_respect_prior():
    LL = np.zeros((4, 2))
    post = window_posteriors(LL, prior=[0.8, 0.2])
    assert np.allclose(post, [[0.8, 0.2]] * 4)
