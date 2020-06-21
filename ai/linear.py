import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G
from ai.module import Module


# linear affine transformation: y = Wx + b
# the general feed-forward network
class Linear(Module):
    def __init__(self, input_features=0, output_features=0, bias=True, graph=G):
        super(Linear, self).__init__()
        self.input_features = input_features  # previous layer units
        self.output_features = output_features  # next layer units
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.input_features)
        self.W = Parameter((self.output_features, self.input_features), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # weight volume
        self.b = Parameter((self.output_features, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)   # bias vector

    def __str__(self):
        return('Linear(input_features={}, output_features={}, bias={})'.format(
            self.input_features, self.output_features, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):
        # making the input compatible with graph operations
        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # flatten the input if it came from layers like Conv2d
        if len(x.shape) > 2:
            x = self.graph.reshape(x)

        # y = Wx + b
        out = self.graph.dot(self.W, x) # matmul

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-1,))

        return out
