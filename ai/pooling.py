import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# maxpool2d layer - non-parametrized layer
class Maxpool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, graph=G):
        super().__init__()

        if stride is None:
            stride = kernel_size
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

    def __repr__(self):
        return(f'Maxpool2d(kernel_size={self.kernel_size}, stride={ self.stride}, padding={self.padding})')

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        out = self.graph.max_pool2d(x, self.kernel_size, s=self.stride, p=self.padding)

        return out
