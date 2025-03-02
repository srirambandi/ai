import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G
from ai.module import Module


# dropout layer - non-parametrized layer
class Dropout(Module):
    def __init__(self, p=0.5, graph=G):
        super(Dropout, self).__init__()
        self.p = p
        self.graph = graph

    def __repr__(self):
        return('Dropout(p={})'.format(self.p))

    def __call__(self, *args, **kwargs):  # easy callable
        return self.forward(*args, **kwargs)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        out = self.graph.dropout(x, p=self.p)

        return out
