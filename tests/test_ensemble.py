import numpy as np
import torch

from btm.ensemble import EnsembleLM, ensemble_from_arrays, log_prior_topk, mixture_logprobs
from btm.posterior import token_logprobs
from conftest import VOCAB


def test_offline_self_ensemble_equals_model():
    lp = np.random.default_rng(0).normal(-3, 1, size=(4, 31))
    lp_ens, _ = ensemble_from_arrays([lp, lp], [0.3, 0.7])
    assert np.allclose(lp_ens, lp, atol=1e-10)


def test_ensemble_lm_self_ensemble_equals_model(tiny_model):
    m = tiny_model(0)
    x = torch.randint(0, VOCAB, (2, 20))
    with torch.no_grad():
        out = EnsembleLM([m, m], [0.5, 0.5])(x)
        ref = torch.log_softmax(m(input_ids=x).logits.float(), -1)
    assert out.shape == (2, 20, VOCAB)
    assert torch.allclose(out, ref, atol=1e-5)


def test_ensemble_lm_matches_offline(tiny_model):
    models = [tiny_model(0), tiny_model(1), tiny_model(2)]
    prior = [0.2, 0.5, 0.3]
    x = torch.randint(0, VOCAB, (3, 33))
    arrays = [token_logprobs(m, x.numpy()) for m in models]
    lp_ens, _ = ensemble_from_arrays(arrays, prior)
    with torch.no_grad():
        full = EnsembleLM(models, prior)(x)
    gathered = full[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1).numpy()
    assert np.abs(gathered - lp_ens).max() < 1e-4
    assert torch.allclose(full.exp().sum(-1), torch.ones(3, 33), atol=1e-4)


def test_posterior_resets_and_starts_at_prior():
    lp = np.random.default_rng(0).normal(-3, 1, size=(2, 5, 10))
    _, logpost = mixture_logprobs(lp, np.log([0.25, 0.75]))
    assert np.allclose(np.exp(logpost[:, :, 0].numpy()), [[0.25] * 5, [0.75] * 5])


def test_topk_prior():
    assert np.allclose(np.exp(log_prior_topk([0.2, 0.5, 0.3], 1)), [0, 1, 0])
    assert np.allclose(np.exp(log_prior_topk([0.2, 0.5, 0.3], 2)), [0, 0.625, 0.375])
    lp = np.random.default_rng(0).normal(-3, 1, size=(4, 12))
    other = lp - 1.0
    top1, _ = ensemble_from_arrays([other, lp, other], [0.2, 0.5, 0.3], k=1)
    assert np.allclose(top1, lp)
