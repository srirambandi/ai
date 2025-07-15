import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


class ReLU(Module):
    def __init__(self, graph=G):
        super().__init__(graph)
        self.graph = graph

    def __repr__(self):
        return(f'ReLU() Activation')

    def forward(self, x):
        return self.graph.relu(x)


class LeakyReLU(Module):
    def __init__(self, alpha=1e-2, graph=G):
        super().__init__(graph)
        self.alpha = alpha
        self.graph = graph

    def __repr__(self):
        return(f'LeakyReLU(alpha={self.alpha}) Activation')

    def forward(self, x):
        return self.graph.leaky_relu(x, alpha=self.alpha)


class GELU(Module):
    def __init__(self, graph=G):
        super().__init__(graph)
        self.graph = graph

    def __repr__(self):
        return('GELU() Activation')

    def forward(self, x):
        return self.graph.gelu(x)


class Sigmoid(Module):
    def __init__(self, graph=G):
        super().__init__(graph)
        self.graph = graph

    def __repr__(self):
        return('Sigmoid() Activation')

    def forward(self, x):
        return self.graph.sigmoid(x)


class Tanh(Module):
    def __init__(self, graph=G):
        super().__init__(graph)
        self.graph = graph

    def __repr__(self):
        return('Tanh() Activation')

    def forward(self, x):
        return self.graph.tanh(x)


class Softmax(Module):
    def __init__(self, axis=0, graph=G):
        super().__init__(graph)
        self.axis = axis
        self.graph = graph

    def __repr__(self):
        return(f'Softmax(axis={self.axis}) Activation')

    def forward(self, x):
        return self.graph.softmax(x, axis=self.axis)
