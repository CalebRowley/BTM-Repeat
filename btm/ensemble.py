import numpy as np
import torch
from torch import nn


def log_prior_topk(prior, k=None):
    prior = np.asarray(prior, dtype=np.float64)
    if k is None or k >= len(prior):
        return np.log(prior / prior.sum())
    keep = np.argsort(-prior, kind="stable")[:k]
    p = np.zeros_like(prior)
    p[keep] = prior[keep]
    with np.errstate(divide="ignore"):
        return np.log(p / p.sum())


def mixture_logprobs(lp, log_prior):
    lp = torch.as_tensor(lp, dtype=torch.float64)
    log_prior = torch.as_tensor(log_prior, dtype=torch.float64)
    cum = torch.cat([torch.zeros_like(lp[..., :1]), lp.cumsum(-1)[..., :-1]], -1)
    logpost = log_prior.view(-1, *([1] * (lp.dim() - 1))) + cum
    logpost = logpost - torch.logsumexp(logpost, 0, keepdim=True)
    lp_ens = torch.logsumexp(logpost + lp, 0)
    return lp_ens, logpost


def ensemble_from_arrays(arrays, prior, k=None):
    lp = np.stack([np.asarray(a) for a in arrays])
    lp_ens, logpost = mixture_logprobs(lp, log_prior_topk(prior, k))
    return lp_ens.numpy(), logpost.numpy()


def perplexity(lp):
    return float(np.exp(-np.mean(lp, dtype=np.float64)))


def posterior_entropy(logpost):
    logpost = np.asarray(logpost, dtype=np.float64)
    p = np.exp(logpost)
    return -np.where(p > 0, p * logpost, 0.0).sum(0)


class EnsembleLM(nn.Module):
    def __init__(self, models, prior):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("log_prior", torch.as_tensor(np.log(np.asarray(prior, dtype=np.float64)), dtype=torch.float32))

    def forward(self, input_ids):
        logps = torch.stack([torch.log_softmax(m(input_ids=input_ids).logits.float(), -1) for m in self.models])
        J = logps.size(0)
        targets = input_ids[None, :, 1:, None].expand(J, -1, -1, 1)
        tok = logps[:, :, :-1].gather(-1, targets).squeeze(-1).double()
        cum = torch.cat([torch.zeros_like(tok[..., :1]), tok.cumsum(-1)], -1)
        logpost = self.log_prior.double().view(J, 1, 1) + cum
        logpost = (logpost - torch.logsumexp(logpost, 0, keepdim=True)).float()
        return torch.logsumexp(logpost[..., None] + logps, 0)
