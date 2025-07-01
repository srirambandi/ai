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
    def __init__(self, d_model, bias=True, graph=G):
        super(SelfAttention, self).__init__()
        self.Q_proj = Linear(d_model, d_model, bias=bias)
        self.K_proj = Linear(d_model, d_model, bias=bias)
        self.V_proj = Linear(d_model, d_model, bias=bias)
        self.A_proj = Linear(d_model, d_model, bias=bias)
        self.graph = graph

    def forward(self, x):
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        d_k = math.sqrt(K.shape[-1])
        dot_product = Q @ K.transpose(axis0=1, axis1=2)
        scaled_dot_product = dot_product / d_k
        # TODO: masking step here.
        attention_probs = self.graph.softmax(scaled_dot_product, axis=-1)
        attention = attention_probs @ V
        output = self.A_proj(attention)

        return output


class MultiHeadAttention(Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        pass


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