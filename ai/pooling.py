import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G
from ai.module import Module


# maxpool2d layer - non-parametrized layer
class Maxpool2d(Module):
    def __init__(self, kernel_size=None, stride=(1, 1), padding=(0, 0), graph=G):
        super(Maxpool2d, self).__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.graph = graph

    def __str__(self):
        return('Maxpool2d(kernel_size={}, stride={}, padding={})'.format(
            self.kernel_size, self.stride, self.padding))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        out = self.graph.max_pool2d(x, k=self.kernel_size, s=self.stride, p=self.padding)
        
        return out
