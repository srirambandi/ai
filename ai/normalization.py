import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G
from ai.module import Module


# batch normalization layer
class BatchNorm2D(Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1, graph=G):
        super(BatchNorm2D, self).__init__()
        self.num_channels = num_channels  # should be equal to the number of channels in the input
        self.eps = eps    # small value to avoid division by zero
        self.momentum = momentum
        self.graph = graph
        self.init_params()

    def init_params(self):
        # In BatchNorm2d, the input is assumed to be of shape (N, C, H, W), where input is a mini-batch images of size N
        # we normalize across the channel axis C
        shape = (1, self.num_channels, 1, 1)
        self.gamma = Parameter(shape, init_ones=True, graph=self.graph)
        self.beta = Parameter(shape, init_zeros=True, graph=self.graph)
        self.m = Parameter(shape, init_zeros=True, requires_grad=False, graph=self.graph)     # moving mean - not trainable
        self.v = Parameter(shape, init_ones=True, requires_grad=False, graph=self.graph)      # moving variance - not trainable

    def __repr__(self):
        return('BatchNorm2D({}, eps={}, momentum={})'.format(
            self.num_channels, self.eps, self.momentum))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        if self.graph.grad_mode:    # training
            # useful: https://arxiv.org/abs/1502.03167

            normalizing_size = Parameter((1,), init_zeros=True, requires_grad=False, graph=self.graph)
            normalizing_size.data.fill(float(x.data.size // self.num_channels))
            eps = Parameter((1,), init_zeros=True, requires_grad=False, graph=self.graph)
            eps.data.fill(self.eps)

            # calculate mean and variance
            mean = self.graph.divide(self.graph.sum(x, axis=(0, 2, 3)), normalizing_size)
            centered = self.graph.subtract(x, mean, axis=(0, 2, 3))
            var = self.graph.divide(self.graph.sum(self.graph.power(centered, 2), axis=(0, 2, 3)), normalizing_size)

            self.m.data = (1 - self.momentum) * self.m.data + self.momentum * mean.data
            self.v.data = (1 - self.momentum) * self.v.data + self.momentum * var.data

            # normalize the data to zero mean and unit variance
            rstd = self.graph.power(self.graph.add(var, eps), -0.5)
            normalized = self.graph.multiply(centered, rstd, axis=(0, 2, 3))

        else:   # testing/inference

            centered = np.subtract(x.data, self.m.data)
            normalized = np.multiply(centered, np.power(self.v.data + self.eps, -0.5))
            normalized = Parameter(data=normalized, requires_grad=False, graph=self.graph)

        # scale and shift
        out = self.graph.multiply(normalized, self.gamma, axis=(0, 2, 3))    # scale
        out = self.graph.add(out, self.beta, axis=(0, 2, 3))    # shift

        return out
