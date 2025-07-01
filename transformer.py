"""
Vanilla Transformer model implemntation using this AI library

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://www.github.com/srirambandi

MIT License
"""

import math

from ai.graph import G
from ai.parameter import Parameter
from ai.module import Module
from ai.linear import Linear


class SelfAttention(Module):
    """
    self attention for 1 head in the multi head attention.
    this is a causal self attention, because of the type of masking in forward():
    every token only attends to the tokens before it.
    """
    def __init__(self, d_model, bias=True):
        super(SelfAttention, self).__init__()
        self.Q_proj = Linear(d_model, d_model, bias=bias)
        self.K_proj = Linear(d_model, d_model, bias=bias)
        self.V_proj = Linear(d_model, d_model, bias=bias)
        self.A_proj = Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        d_k = math.sqrt(K.shape[-1])    # which is d_model generally
        dot_product = Q @ K.transpose(axis0=1, axis1=2)
        scaled_dot_product = dot_product / d_k
        # TODO: masking step here.
        attention_probs = self.graph.softmax(scaled_dot_product, axis=-1)
        attention = attention_probs @ V
        output = self.A_proj(attention)

        return output


class MultiHeadAttention(Module):
    def __init__(self, num_heads, d_model, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        for i in range(num_heads):
            setattr(self, f"attn_{i}", SelfAttention(d_model, bias=bias))
        
    def forward(self, x):
        xs = self.graph.split(x, sections=self.num_heads, axis=-1)
        ys = []
        for i in range(self.num_heads):
            a = getattr(self, f"attn_{i}")(xs[i])
            ys.append(a)
        output = self.graph.cat(ys, axis=-1)

        return output


class FeedForwardNetwork(Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        pass


class TransformerLayer(Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        pass


class Transformer(Module):
    def __init__(self):
        super(Transformer, self).__init__()
        pass