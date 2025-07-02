import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# embeddings for nlp models
# a look up table that maps ints to embed vectors
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, graph=G):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.embedding_table = Parameter(shape=(self.num_embeddings, self.embedding_dim))

    def __repr__(self):
        return(f'Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})')

    def forward(self, x):
        # making the input compatible with graph operations
        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)
        
        batch_embeddings = []
        for b in range(len(x.shape[0])):
            embeddings = []
            for l in range(len(x.shape[1])):
                embeddings.append(self.embedding_table[x[b, l], :])
            embeddings = self.graph.cat(embeddings, axis=0)
            embeddings = embeddings.reshape(1, *embeddings.shape)
            batch_embeddings.append(embeddings)
        output = self.graph.cat(batch_embeddings, axis=0)

        return output