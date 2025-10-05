"""
Transformer model implemntation using this AI library

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://www.github.com/srirambandi

MIT License
"""

import ai
import math
import numpy as np
from dataclasses import dataclass


class CausalSelfAttention(ai.Module):
    """
    self attention with multiple heads.
    this is a causal self attention, because of the type of masking in forward():
    every token only attends to the tokens before it.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_head = config.num_head
        # query, key and value projections, for all heads
        self.QKV_proj = ai.Linear(self.d_model, 3 * self.d_model, bias=config.bias)
        # attention output projection
        self.A_proj = ai.Linear(self.d_model, self.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = ai.Dropout(config.dropout)
        self.residual_dropout = ai.Dropout(config.dropout)
        mask = np.ones((1, 1, config.context_length, config.context_length))
        mask = np.tril(mask)
        mask = np.where(mask==0, float("-inf"), mask)
        mask = np.where(mask==1, 0, mask)
        self.causal_mask = ai.Parameter(data=mask, requires_grad=False)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.QKV_proj(x).split(sections=3, axis=-1)
        Q = Q.reshape(B, T, self.num_head, C // self.num_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_head, C // self.num_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_head, C // self.num_head).transpose(1, 2)

        d_k = math.sqrt(K.shape[-1])
        dot_product = Q @ K.transpose(-2, -1)
        scaled_dot_prod = dot_product / d_k
        masked_dot_prod = scaled_dot_prod + self.causal_mask[:, :, :T, :T]
        attention_probs = self.graph.softmax(masked_dot_prod, axis=-1)
        attention_probs = self.attn_dropout(attention_probs)
        attention = attention_probs @ V
        
        attention = attention.transpose(1, 2).reshape(B, T, C)
        output = self.residual_dropout(self.A_proj(attention))

        return output


class FeedForwardNetwork(ai.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = ai.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = ai.GELU()
        self.out_proj = ai.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = ai.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        x = self.dropout(x)

        return x


class TransformerLayer(ai.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = ai.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = ai.LayerNorm(config.d_model)
        self.ffn = FeedForwardNetwork(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


@dataclass
class TransformerConfig:
    context_length: int = 1024
    vocab_size: int = 50304
    num_layer: int = 12
    num_head: int = 12
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = True


class Transformer(ai.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = ai.Embedding(config.vocab_size, config.d_model)
        self.wpe = ai.Embedding(config.vocab_size, config.d_model)
        self.dropout = ai.Dropout(config.dropout)
        for tl in range(len(config.num_layer)):
            setattr(f"tl_{tl}", TransformerLayer(config))
        self.ln_f = ai.LayerNorm(config.d_model)
        self.lm_head = ai.Linear(config.d_model, config.vocab_size)
        # weight tying: https://arxiv.org/abs/1608.05859
        # also this interesting reddit post: https://www.reddit.com/r/MachineLearning/comments/1eqm0lr/r_why_and_when_tying_embedding_a_story/
        self.wte.embedding_table = self.lm_head.W
        self.loss = ai.CrossEntropyLoss()

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, maximum sequence length is: {self.config.context_length}"

        pos = np.arange(0, T)
        tok_emb = self.wte(inputs)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        for tl in range(len(config.num_layer)):
            x = getattr(f"tl_{tl}")(x)
        x = self.ln_f(x)

        if targets is not None:
            # training - compute loss
            logits = self.lm_head(x)
            targets_one_hot = np.zeros((B, T, self.config.vocab_size))
            batch_idxs, time_idxs = np.indices((B, T))
            # numpy advanced indexing for efficiency
            targets_one_hot[batch_idxs, time_idxs, targets] = 1.
            targets_one_hot = ai.Parameter(data=targets_one_hot.reshape(B*T, self.config.vocab_size), requires_grad=False)
            loss = self.loss(logits.reshape(B*T, self.config.vocab_size), targets_one_hot)
        else:
            # inference - only the last position is forwarded through the lm_head
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


config = TransformerConfig(
    context_length = ...,
    vocab_size = ...,
    num_layer = ...,
    num_head = ...,
    d_model = ...,
    dropout = ...,
    bias = ...,
)


if __name__ == "__main__":
    pass
