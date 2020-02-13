import numpy as np
from .parameter import Parameter
from .graph import ComputationalGraph, G


# conv nets
class Conv2d:
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1, 1), padding=(0, 0)):
        self.input_channels = input_channels
        self.output_channels = output_channels

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        if type(stride) is not tuple:
            stride = (stride, stride)
        if type(padding) is not tuple:
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.filter_size = (self.input_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.init_params()

    def init_params(self):
        self.K = Parameter((self.output_channels, *self.filter_size))
        self.b = Parameter((self.output_channels, 1, 1), init_zeros=True)
        self.parameters = [self.K, self.b]

    def forward(self, x):
        if type(x) is not Parameter:
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=False, init_zeros=True)
            x.w = _

        out = G.scalar_add(G.conv2d(x, self.K, self.stride, self.padding), self.b)     # convoulution operation and adding bias

        return out
