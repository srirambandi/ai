import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# linear affine transformation: y = Wx + b
# the general feed-forward network
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, graph=G):
        super(Linear, self).__init__()
        self.in_features = in_features  # previous layer units
        self.out_features = out_features  # next layer units
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.in_features)
        self.W = Parameter((self.out_features, self.in_features), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # weight volume
        if self.bias:
            self.b = Parameter((1, self.out_features), uniform=True, low=-root_k, high=root_k, graph=self.graph)   # bias vector

    def __repr__(self):
        return(f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})')

    def forward(self, x):
        # making the input compatible with graph operations
        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        # y = xW.T + b
        out = x @ self.W.transpose() # matmul

        if self.bias:   # adding bias
            out = out + self.b

        return out
