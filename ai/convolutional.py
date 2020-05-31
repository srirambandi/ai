import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


# 2D convolutional neural network
class Conv2d:
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1, 1), padding=(0, 0), bias=True, graph=G):
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
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.K = Parameter((self.output_channels, *self.filter_size), graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1, 1), init_zeros=True, graph=self.graph)
        self.parameters = [self.K, self.b]

    def __str__(self):
        return('Conv2d({}, {}, kernel_size={}, stride={}, padding={}, bias={})'.format(
            self.input_channels, self.output_channels, self.kernel_size, self.stride, self.padding, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # convolution operation
        out = self.graph.conv2d(x, self.K, self.stride, self.padding)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-3, -2, -1))

        return out


# 2d transposed convolutional neural network
class ConvTranspose2d:
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1, 1), padding=(0, 0), a=(0, 0), bias=True, graph=G):
        self.input_channels = input_channels
        self.output_channels = output_channels

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        if type(stride) is not tuple:
            stride = (stride, stride)
        if type(padding) is not tuple:
            padding = (padding, padding)
        if type(a) is not tuple:
            a = (a, a)

        self.kernel_size = kernel_size
        self.filter_size = (self.output_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.a = a  # for fixing a single output shape over many possible
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.K = Parameter((self.input_channels, *self.filter_size), graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1, 1), init_zeros=True, graph=self.graph)
        self.parameters = [self.K, self.b]

    def __str__(self):
        return('ConvTranspose2d({}, {}, kernel_size={}, stride={}, padding={}, a={}, bias={})'.format(
            self.input_channels, self.output_channels, self.kernel_size, self.stride, self.padding, self.a, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # convolution transpose operation
        out = self.graph.conv_transpose2d(x, self.K, self.stride, self.padding, self.a)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-3, -2, -1))

        return out
