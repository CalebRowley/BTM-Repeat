import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

VOCAB = 97


def make_tiny(seed):
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        rotary_pct=0.25,
    )
    return GPTNeoXForCausalLM(cfg).eval()


@pytest.fixture
def tiny_model():
    return make_tiny
