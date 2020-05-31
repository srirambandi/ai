import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


# bacth normalization layer
class BatchNorm:
    def __init__(self, hidden_shape, axis=-1, momentum=0.9, bias=True, graph=G):
        self.hidden_shape = hidden_shape  # gamma and beta size; typically D in (D, N) where N is batch size
        self.axis = axis    # along batch channel axis for conv layers and along batches for linear
        self.momentum = momentum
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        shape = (*self.hidden_shape, 1)
        self.gamma = Parameter(shape, init_ones=True, graph=self.graph)
        self.beta = Parameter(shape, init_zeros=True, graph=self.graph)
        self.parameters = [self.gamma, self.beta]
        self.m = np.sum(np.zeros(shape), axis=self.axis, keepdims=True) / shape[self.axis]    # moving mean
        self.v = np.sum(np.ones(shape), axis=self.axis, keepdims=True) / shape[self.axis]     # moving variance

    def __str__(self):
        return('BatchNorm({}, axis={}, momentum={}, bias={})'.format(
            self.hidden_shape, self.axis, self.momentum, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        if self.graph.grad_mode:    # training
            # useful: https://arxiv.org/abs/1502.03167

            batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size/channel size
            batch_size.data.fill(float(x.shape[self.axis]))

            # calculate mean and variance
            mean = self.graph.divide(self.graph.sum(x, axis=self.axis), batch_size)
            centered = self.graph.subtract(x, mean, axis=self.axis)
            var = self.graph.divide(self.graph.sum(self.graph.power(centered, 2), axis=self.axis), batch_size)

            self.m = self.momentum * self.m + (1 - self.momentum) * mean.data
            self.v = self.momentum * self.v + (1 - self.momentum) * var.data

            # normalize the data to zero mean and unit variance
            normalized = self.graph.multiply(centered, self.graph.power(var, -0.5), axis=self.axis)

        else:   # testing

            centered = np.subtract(x.data, self.m)
            normalized = np.multiply(centered, np.power(self.v + 1e-6, -0.5))
            normalized = Parameter(data=normalized, eval_grad=False, graph=self.graph)

        # scale and shift
        out = self.graph.multiply(normalized, self.gamma, axis=(-1,))    # scale

        if self.bias:   # shift
            out = self.graph.add(out, self.beta, axis=(-1,))

        return out
