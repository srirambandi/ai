import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# batch normalization layer
class BatchNorm2d(Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1, graph=G):
        super().__init__()
        self.num_channels = num_channels  # should be equal to the number of channels in the input
        self.eps = eps    # small value to avoid division by zero
        self.momentum = momentum
        self.graph = graph
        self.normalized_axes = (0, 2, 3)
        self.init_params()

    def init_params(self):
        # In BatchNorm2d, the input is assumed to be of shape (N, C, H, W), where input is a mini-batch images of size N
        # we normalize across the channel axis C
        shape = (1, self.num_channels, 1, 1)
        self.weight = Parameter(shape, init_ones=True, graph=self.graph)     # gamma
        self.bias = Parameter(shape, init_zeros=True, graph=self.graph)     # beta
        self.m = Parameter(shape, init_zeros=True, requires_grad=False, graph=self.graph)     # moving mean - not trainable
        self.v = Parameter(shape, init_ones=True, requires_grad=False, graph=self.graph)      # moving variance - not trainable

    def __repr__(self):
        return(f'BatchNorm2d(num_channels={self.num_channels}, eps={self.eps}, momentum={self.momentum})')

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        if self.graph.grad_mode:    # training
            # useful: https://arxiv.org/abs/1502.03167

            normalized_size = Parameter((1,), init_zeros=True, requires_grad=False, graph=self.graph)
            normalized_size.data.fill(float(x.data.size // self.num_channels))
            eps = Parameter((1,), init_zeros=True, requires_grad=False, graph=self.graph)
            eps.data.fill(self.eps)

            # calculate mean and variance
            mean = self.graph.sum(x, axis=self.normalized_axes) / normalized_size
            centered = x - mean
            var = self.graph.sum(self.graph.power(centered, 2), axis=self.normalized_axes) / normalized_size

            self.m.data = (1 - self.momentum) * self.m.data + self.momentum * mean.data
            self.v.data = (1 - self.momentum) * self.v.data + self.momentum * var.data

            # normalize the data to zero mean and unit variance
            rstd = self.graph.power((var + eps), -0.5)
            normalized = centered * rstd

        else:   # testing/inference

            centered = np.subtract(x.data, self.m.data)
            normalized = np.multiply(centered, np.power(self.v.data + self.eps, -0.5))
            normalized = Parameter(data=normalized, requires_grad=False, graph=self.graph)

        # scale and shift
        out = normalized * self.weight    # scale
        out = out + self.bias    # shift

        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, graph=G):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape    # shape in the input from last dim over which to normalize
        self.eps = eps
        self.graph = graph
        self.normalized_axes = [-1 * i for i in range(1, len(normalized_shape) + 1)][::-1]  # axis to normalize over
        self.init_params()

    def init_params(self):
        # we normalize the input over the dimesnsions in normalized_shape(matched from the last dim)
        # example: normalized_shape = (T, C), input shape = (N, T, C), we normalize over T and C element wise
        self.weight = Parameter(self.normalized_shape, init_ones=True, graph=self.graph)     # gamma
        self.bias = Parameter(self.normalized_shape, init_zeros=True, graph=self.graph)     # beta

    def __repr__(self):
        return(f'LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})')
    
    def forward(self, x):
        
        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        if self.graph.grad_mode:
            # useful: https://arxiv.org/abs/1607.06450

            normalized_size = Parameter((1,), init_zeros=True, requires_grad=False, graph=self.graph)
            normalized_size.fill(float(np.prod(self.normalized_shape)))
            eps = Parameter((1,), init_zeros=True, requires_grad=False, graph=self.graph)
            eps.fill(self.eps)

            # calculate mean and variance
            mean = self.graph.sum(x, axis=self.normalized_axes) / normalized_size
            centered = x - mean
            var = self.graph.sum(self.graph.power(centered, 2), axis=self.normalized_axes) / normalized_size

            # normalize the data to zero mean and unit variance
            rstd = self.graph.power((var + eps), -0.5)
            normalized = centered * rstd

        else:
            centered = np.subtract(x.data, np.mean(x.data, axis=self.normalized_axes, keepdims=True))
            normalized = np.multiply(centered, np.power(np.var(x.data, axis=self.normalized_axes, keepdims=True) + self.eps, -0.5))
            normalized = Parameter(data=normalized, requires_grad=False, graph=self.graph)

        # scale and shift
        out = normalized * self.weight    # scale
        out = out + self.bias    # shift

        return out
