import numpy as np
from .parameter import Parameter
from .graph import ComputationalGraph, G


# linear affine transformation: y = Wx + b
# the general feed-forward network
class Linear:
    def __init__(self, h_p = 0, h_n = 0, init_zeros=False):
        self.h_p = h_p  # previous layer units
        self.h_n = h_n  # next layer units
        self.init_zeros = init_zeros
        self.init_params()

    def init_params(self):
        self.W = Parameter((self.h_n, self.h_p), init_zeros=self.init_zeros)  # weight volume
        self.b = Parameter((self.h_n, 1), init_zeros=True)   # bias vector
        self.parameters = [self.W, self.b]  # easy access of the layer params

    def forward(self, x):
        # making the input compatible with graph operations
        if type(x) is not Parameter:
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=True, init_zeros=True)
            x.w = _

        if len(x.shape) > 2 or x.shape[1] != 1:
            x = G.reshape(x)

        out = G.add(G.dot(self.W, x), self.b)   # y = Wx + b

        return out
