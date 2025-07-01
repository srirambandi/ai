import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# 1D convolutional neural network
class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, graph=G):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # making kernel_size, stride, padding tuples just for consistency across conv layers
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size,)
        if not isinstance(stride, tuple):
            stride = (stride,)
        if not isinstance(padding, tuple):
            padding = (padding,)

        self.kernel_size = kernel_size
        self.filter_size = (self.in_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / (self.in_channels * self.kernel_size[0]))
        self.K = Parameter((self.out_channels, *self.filter_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b = Parameter((1, self.out_channels, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return(f'Conv1d(in_channels={self.in_channels}, out_channels={self.out_channels},\n \tkernel_size={self.kernel_size}, \
               stride={self.stride}\n, \tpadding={self.padding}, bias={self.bias})')

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        # convolution operation
        out = self.graph.conv1d(x, self.K, self.stride, self.padding)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(0, -1))

        return out


# 2D convolutional neural network
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, graph=G):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.filter_size = (self.in_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.K = Parameter((self.out_channels, *self.filter_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b = Parameter((1, self.out_channels, 1, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return(f'Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels},\n \tkernel_size={self.kernel_size}, \
               stride={self.stride}\n, \tpadding={self.padding}, bias={self.bias})')

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        # convolution operation
        out = self.graph.conv2d(x, self.K, self.stride, self.padding)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(0, -2, -1))

        return out


# 2d transposed convolutional neural network
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, graph=G):
        super(ConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)
        if not isinstance(output_padding, tuple):
            output_padding = (output_padding, output_padding)

        self.kernel_size = kernel_size
        self.filter_size = (self.out_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding  # for fixing a single output shape over many possible, also called 'a' in conv_transpose2d function
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / (self.out_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.K = Parameter((self.in_channels, *self.filter_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b = Parameter((1, self.out_channels, 1, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return(f'ConvTranspose2d(in_channels={self.in_channels}, out_channels={self.out_channels},\n \tkernel_size={self.kernel_size}, \
               stride={self.stride}\n, \tpadding={self.padding}, output_padding={self.output_padding}, bias={self.bias})')

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        # convolution transpose operation
        out = self.graph.conv_transpose2d(x, self.K, self.stride, self.padding, self.output_padding)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(0, -2, -1))

        return out
