"""
Vanilla Transformer model implemntation using this AI library

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://www.github.com/srirambandi

MIT License
"""

import math
import numpy as np
from dataclasses import dataclass
from ai.parameter import Parameter
from ai.module import Module
from ai.linear import Linear


class SelfAttention(Module):
    """
    self attention for 1 head in the multi head attention.
    this is a causal self attention, because of the type of masking in forward():
    every token only attends to the tokens before it.
    """
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.d_model = config.d_model
        self.num_head = config.num_head
        self.Q_proj = Linear(self.d_model // self.num_head, self.d_model, bias=config.bias)
        self.K_proj = Linear(self.d_model // self.num_head, self.d_model, bias=config.bias)
        self.V_proj = Linear(self.d_model // self.num_head, self.d_model, bias=config.bias)
        self.A_proj = Linear(self.d_model, self.d_model, bias=config.bias)
        mask = np.ones((1, config.context_length, config.context_length))
        mask = np.tril(mask)
        mask = np.where(mask == 0, float("-inf"), mask)
        self.causal_mask = Parameter(data=mask, requires_grad=False)

    def forward(self, x):
        Q = self.Q_proj(x)  # (B, L, D)
        K = self.K_proj(x)  # (B, L, D)
        V = self.V_proj(x)  # (B, L, D)
        d_k = math.sqrt(K.shape[-1])    # which is d_model(D) generally
        dot_product = Q @ K.transpose(axis0=1, axis1=2) # (B, L, L)
        scaled_dot_prod = dot_product / d_k
        masked_dot_prod = self.graph.multiply(scaled_dot_prod, self.causal_mask)
        attention_probs = self.graph.softmax(masked_dot_prod, axis=-1)   # (B, L, L)
        attention = attention_probs @ V     # (B, L, D)
        output = self.A_proj(attention)

        return output


class MultiHeadAttention(Module):
    def __init__(self, config, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_heads
        self.config = config
        for i in range(config.num_heads):
            setattr(self, f"attn_{i}", SelfAttention(config, bias=bias))
        
    def forward(self, x):
        xs = self.graph.split(x, sections=self.num_heads, axis=-1)
        ys = []
        # TODO: can be made efficient with vectorization or async calls
        for i in range(self.num_heads):
            a = getattr(self, f"attn_{i}")(xs[i])
            ys.append(a)
        output = self.graph.cat(ys, axis=-1)

        return output


class FeedForwardNetwork(Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.config = config
        self.fc = Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.out_proj = Linear(4 * config.d_model, config.d_model, bias=config.bias)

    def forward(self, x):
        output = self.out_proj(self.fc(x))

        return output


class TransformerLayer(Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.config = config

    def forward(self, x):
        pass


@dataclass
class TransformerConfig:
    context_length: int = 1024
    vocab_size: int = 50304
    num_layer: int = 12
    num_head: int = 12
    d_model: int = 768
    bias: bool = False


class Transformer(Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config

    def forward(self, x):
        pass


config = TransformerConfig(
    context_length = ...,
    vocab_size = ...,
    num_layer = ...,
    num_head = ...,
    d_model = ...,
    bias = ...,
)