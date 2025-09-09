import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# embeddings for nlp models
# a look up table that maps ints to embed vectors
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, graph=G):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.weight = Parameter(shape=(self.num_embeddings, self.embedding_dim))

    def __repr__(self):
        return(f'Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})')

    def forward(self, x):
        # making the input compatible with graph operations
        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        # let's only deal with inputs that don't need training as is required in most cases
        assert not x.requires_grad, "Embedding only takes in Parameter with no training requirement."

        B, L = x.shape
        mask = np.zeros((B, L, self.num_embeddings, self.embedding_dim))
        B_idx = np.arange(B)[:, None]
        L_idx = np.arange(L)
        mask[B_idx, L_idx, x, :] = 1.0

        mask = Parameter(data=mask, requires_grad=False)

        emb = self.weight.reshape((1, 1, self.num_embeddings, self.embedding_dim))   # (1, 1, num_embeddings, embedding_dim)
        embeddings = (emb * mask).sum(axis=2)  # (B, L, embedding_dim)

        return embeddings