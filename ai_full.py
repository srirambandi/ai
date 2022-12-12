"""
AI library in python using numpy

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://www.github.com/srirambandi

MIT License
"""

import numpy as np


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), data=None, grad=None, eval_grad=True, node_id=None, graph=None,
                init_zeros=False, init_ones=False, constant=1.0,
                uniform=False, low=-1.0, high = 1.0,
                normal=False, mean=0.0, std=0.01):

        # properties
        self.shape = shape
        self.data = data
        self.grad = grad
        self.eval_grad = eval_grad  # if the parameter is a variable or an input/constant

        # node id - in the bfs like graph walk during forward pass, the node numeber
        # of the path ie., the latest backward op of which this parameter was an output
        self.node_id = node_id

        if graph is not None:   # graph object this parameter belongs to
            self.graph = graph
        else:
            self.graph = G

        # constant initializations
        self.init_zeros = init_zeros
        self.init_ones = init_ones
        self.constant = constant

        # initializing from distributions
        self.uniform = uniform
        self.low = low      # high and low of uniform
        self.high = high    # distribution to initialize the parameter
        self.mean = mean    # mean and variance of the gaussian
        self.std = std      # distribution to initialize the parameter

        # creating weight and gradient tensors
        self.init_params()

    def init_params(self):

        if self.data is not None:
            # initiating weights with passed data object of kind list/numpy-ndarray
            if not isinstance(self.data, np.ndarray):
                self.data = np.array(self.data)
            self.shape = self.data.shape   # resolving conflict with passed shape and data shape

        elif self.init_zeros:
            # initiating with zeros of given shape
            self.data = np.zeros(self.shape)

        elif self.init_ones:
            # initiating with ones(or a constant) of given shape
            self.data = np.ones(self.shape) * self.constant

        elif self.uniform:
            # random initiation with uniform distribution
            self.data = np.random.uniform(self.low, self.high, self.shape)

        else:
            # random initiation with gaussian distribution
            self.normal = True
            self.data = np.random.normal(self.mean, self.std, self.shape)

        # setting gradient of parameter wrt some scalar, as zeros
        if self.grad is None:
            self.grad = np.zeros(self.shape)
        else:
            if not isinstance(self.grad, np.ndarray):
                self.grad = np.array(self.grad)
            assert self.data.shape == self.grad.shape, 'data and grad should be of same shape'

    def __repr__(self):
        parameter_schema = 'Parameter(shape={}, eval_grad={}) containing:\n'.format(self.shape, self.eval_grad)
        parameter_schema += 'Data: {}'.format(self.data)

        return parameter_schema

    # this function computes the gradients of the parameters, by executing
    # the backprop ops in reverse order to the forward propagation with chain rule
    def backward(self, grad=None, to=None):
        # assign gradient

        if self.node_id is None:
            return

        if grad is not None:
            if not isinstance(grad, np.ndarray):
                self.grad = np.array(grad)

        if to is None:
            to_node_id = 0    # execute backward all the way to start
        else:
            to_node_id = to.node_id + 1  # execute backward  to just before this node

        for node in reversed(self.graph.nodes[to_node_id:int(self.node_id) + 1]):
            node['backprop_op']()       # executing the back-propagation operation

    def __getitem__(self, key):

        axis = []
        return_scalar = True
        for _ in range(len(key)):
            if isinstance(key[_], int):
                axis.append(_)
            if isinstance(key[_], slice):
                return_scalar = False
        axis = tuple(axis)

        if return_scalar:
            return self.data[key]
        else:
            return Parameter(data=np.expand_dims(self.data[key], axis=axis),
                             grad=np.expand_dims(self.grad[key], axis=axis))

    def __add__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, ('Objects not of same shape. Use G.add() with axis argument', self.shape, other.shape)

        return self.graph.add(self, other)

    def __sub__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, ('Objects not of same shape. Use G.subtract() with axis argument', self.shape, other.shape)

        return self.graph.subtract(self, other)

    def __mul__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, ('Objects not of same shape. Use G.multiply() with axis argument', self.shape, other.shape)

        return self.graph.multiply(self, other)

    def __matmul__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        return self.graph.dot(self, other)

    def __truediv__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, 'Objects not of same shape. Use G.divide() with axis argument'

        return self.graph.divide(self, other)

    def __pow__(self, other):
        return self.graph.power(self, other)

    # transpose
    def T(self):

        self.data = self.data.T
        self.grad = self.grad.T
        self.shape = tuple(reversed(self.shape))

        return self


# Computational Graph wannabe: stores the backward operation for every
# forward operation during forward-propagation, in a breadth-fist manner
class ComputationalGraph:
    def __init__(self, grad_mode=True):
        self.grad_mode = grad_mode
        self.nodes = list()

    # functions required for deep learning models and their respective backward operations
    def dot(self, W, x):    # dot product of vectors and matrices

        assert W.shape[1] == x.shape[0], 'shape mismatch in dot() operation - W: {}, x: {}'.format(W.shape, x.shape)
        out = Parameter(data=np.dot(W.data, x.data), graph=self)

        if self.grad_mode:
            def backward():
                # useful: http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf
                # print('dot')
                if W.eval_grad:
                    W.grad += np.dot(out.grad, x.data.T)
                if x.eval_grad:
                    x.grad += np.dot(out.grad.T, W.data).T

                # return (x.grad, W.grad)

            node = {'func': '@', 'inputs': [W, x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def add(self, x, y, axis=()):    # element wise addition
        # bias should be passed in position of y
        out = Parameter(data=np.add(x.data, y.data), graph=self)

        if self.grad_mode:
            def backward():
                # print('add')
                if x.eval_grad:
                    x.grad += out.grad
                if y.eval_grad:
                    y.grad += np.sum(out.grad, axis = axis).reshape(y.shape)   # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            node = {'func': '+', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def subtract(self, x, y, axis=()):   # element wise subtraction
        out = Parameter(data=np.subtract(x.data, y.data), graph=self)

        if self.grad_mode:
            def backward():
                # print('subtract')
                if x.eval_grad:
                    x.grad += out.grad
                if y.eval_grad:
                    y.grad -= np.sum(out.grad, axis=axis).reshape(y.shape)  # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            node = {'func': '-', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def multiply(self, x, y, axis=()):   # element wise vector multiplication
        out = Parameter(data=np.multiply(x.data, y.data), graph=self)

        if self.grad_mode:
            def backward():
                # print('multiply')
                if x.eval_grad:
                    x.grad += np.multiply(out.grad, y.data)
                if y.eval_grad:
                    y.grad += np.sum(np.multiply(out.grad, x.data), axis=axis).reshape(y.shape) # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            node = {'func': '*', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def divide(self, x, y, axis=()):   # element wise vector division
        out = Parameter(data= np.divide(x.data, y.data + 1e-8), graph=self)

        if self.grad_mode:
            def backward():
                # print('divide')
                if x.eval_grad:
                    x.grad += np.multiply(out.grad, np.divide(1.0, y.data + 1e-8))
                if y.eval_grad:
                    y.grad += np.sum(np.multiply(out.grad, np.multiply(out.data, np.divide(-1.0, y.data + 1e-8))), axis=axis).reshape(y.shape) # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            node = {'func': '/', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def sum(self, h, axis=None):   # sum of all elements in the matrix
        if axis == None:
            res = np.sum(h.data).reshape(1, 1)
        else:
            res = np.sum(h.data, axis=axis, keepdims=True)
        out = Parameter(data=res, graph=self)

        if self.grad_mode:
            def backward():
                # print('sum')
                if h.eval_grad:
                    h.grad += out.grad

                # return h.grad

            node = {'func': 'sum', 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def power(self, h, exp):   # element wise power
        out = Parameter(h.shape, init_zeros=True, graph=self)
        out.data = np.power(h.data, exp) if exp >= 0 else np.power(h.data + 1e-8, exp)     # numerical stability for -ve power

        if self.grad_mode:
            def backward():
                # print('power')
                if h.eval_grad:
                    if exp  >= 0:
                        h.grad += np.multiply(out.grad, exp * np.power(h.data, exp - 1))
                    else:
                        h.grad += np.multiply(out.grad, exp * np.power(h.data + 1e-8, exp - 1))

                # return h.grad

            node = {'func': '^{}'.format(exp), 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def log(self, h):   # element wise logarithm
        out = Parameter(data=np.log(h.data + 1e-8), graph=self)     # numerical stability for values ~0

        if self.grad_mode:
            def backward():
                # print('log')
                if h.eval_grad:
                    h.grad += np.multiply(out.grad, np.divide(1.0, h.data + 1e-8))

                # return h.grad

            node = {'func': 'log', 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # layers functions
    def conv1d(self, x, K, s=(1,), p=(0,)):
        # faster 1d convolution operation

        if not isinstance(s, tuple):
            s = (s,)
        if not isinstance(p, tuple):
            p = (p,)

        C = K.shape[1]      # number of input channels
        F = K.shape[0]      # number of output filters
        i = x.shape[1:-1]   # input channel shape
        k = K.shape[2:]     # kernel filter shape
        N = x.shape[-1]     # Batch size

        # Figure out output dimensions
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        # padding the input
        pad_x = np.pad(x.data, ((0, 0), (*p, *p), (0, 0)), mode='constant')

        # get strided view of padded input by picking appropriate strides
        shape = (C, *k, *o, N)
        strides = (pad_i[0] * N, N, s[0] * N, 1)
        strides = pad_x.itemsize * np.array(strides)
        stride_x = np.lib.stride_tricks.as_strided(pad_x, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(stride_x)
        x_cols = x_cols.reshape(C * k[0], o[0] * N)

        # convolution operation - matrix multiplication of strided array with kernel
        out = K.data.reshape(F, -1).dot(x_cols)

        # Reshape the output
        out = out.reshape(F, *o, N)
        out = np.ascontiguousarray(out)

        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('conv1d')
                if K.eval_grad:
                    K.grad += np.ascontiguousarray(out.grad.reshape(F, -1).dot(x_cols.T).reshape(K.shape))

                if x.eval_grad:

                    pad_x_grad = np.zeros(pad_x.shape)
                    for r in range(out.shape[1]):

                        # solving gradient for input feature map that caused the elements in r position of every output filter
                        # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimention using sum

                        _ = out.grad[:, r, :].reshape(F, 1, 1, N)
                        pad_x_grad[:, r*s[0]:r*s[0] + k[0], :] += np.sum(np.multiply(_, K.data.reshape(*K.shape, 1)), axis=0)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, p[0]:pad_x_grad.shape[1]-p[0], :]

                # return (K.grad, x.grad)

            node = {'func': 'conv1d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def conv2d_old(self, x, K, s=(1, 1), p=(0, 0)):
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        # 2d convolution operation - simple but inefficient implementation
        # Conv2d lasyer uses conv2d_faster for faster computation
        if not isinstance(s, tuple):
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)

        F = K.shape[0]     # number of output filters
        C = K.shape[1]     # number of input channels
        k = K.shape[2:]    # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel's shape, of some cth-channel in a kth-filter
        i = x.shape[1:-1]  # input shape of any channel of the input feature map before padding
        N = x.shape[-1]    # batch size of the input
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        out = np.zeros((F, *o, N))  # output feature maps

        pad_x = np.pad(x.data, ((0, 0), p, p, (0, 0)), mode='constant')
        pad_x = pad_x.reshape(1, *pad_x.shape)

        # convolution function computing cross-correlation instead of actual convolution - otherwise have to use
        # flipped kernels which doesn't effect learning
        kernel = K.data.reshape(*K.shape, 1)

        for r in range(out.shape[1]):        # convolving operation here
            for c in range(out.shape[2]):    # traversing rous and columns of feature map

                # multiplying traversed grid portions of padded input feature maps with kernel grids element-wise
                # and summing the resulting matrix to produce elements of output maps, over all filters and batches
                out[:, r, c, :] += np.sum(np.multiply(pad_x[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :], kernel), axis=(1, 2, 3))

        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('conv2d')
                if K.eval_grad:

                    for r in range(out.shape[1]):
                        for c in range(out.shape[2]):

                            # solving gradient for each kernel filter that caused the elements in r, c position of every output filter
                            # in every bacth; sketch and think, with input stacked fi times to make computation fast

                            _ = out.grad[:, r, c, :].reshape(F, 1, 1, 1, N)
                            # updating the kernel filter set gradient - there will be RxC such updates
                            K.grad += np.sum(np.multiply(_, pad_x[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]), axis = -1)

                if x.eval_grad:

                    pad_x_grad = np.zeros((C, *pad_i, N))

                    for r in range(out.shape[1]):
                        for c in range(out.shape[2]):

                            # solving gradient for input feature map that caused the elements in r, c position of every output filter
                            # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimention using sum

                            _ = out.grad[:, r, c, :].reshape(F, 1, 1, 1, N)
                            pad_x_grad[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, kernel), axis=0)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, p[0]:pad_x_grad.shape[1]-p[0], p[1]:pad_x_grad.shape[2]-p[1], :]

                # return (K.grad, x.grad)

            node = {'func': 'conv2d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def conv2d(self, x, K, s=(1, 1), p=(0, 0)):
        # faster 2d convolution operation

        if not isinstance(s, tuple):
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)

        C = K.shape[1]      # number of input channels
        F = K.shape[0]      # number of output filters
        i = x.shape[1:-1]   # input channel shape
        k = K.shape[2:]     # kernel filter shape
        N = x.shape[-1]     # Batch size

        # Figure out output dimensions
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        # padding the input
        pad_x = np.pad(x.data, ((0, 0), p, p, (0, 0)), mode='constant')

        # get strided view of padded input by picking appropriate strides
        shape = (C, *k, *o, N)
        strides = (pad_i[0] * pad_i[1] * N, pad_i[1] * N, N, s[0] * pad_i[1] * N, s[1] * N, 1)
        strides = pad_x.itemsize * np.array(strides)
        stride_x = np.lib.stride_tricks.as_strided(pad_x, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(stride_x)
        x_cols = x_cols.reshape(C * k[0] * k[1], o[0] * o[1] * N)

        # convolution operation - matrix multiplication of strided array with kernel
        out = K.data.reshape(F, -1).dot(x_cols)

        # Reshape the output
        out = out.reshape(F, *o, N)
        out = np.ascontiguousarray(out)

        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('conv2d')
                if K.eval_grad:
                    K.grad += np.ascontiguousarray(out.grad.reshape(F, -1).dot(x_cols.T).reshape(K.shape))

                if x.eval_grad:

                    pad_x_grad = np.zeros(pad_x.shape)
                    for r in range(out.shape[1]):
                        for c in range(out.shape[2]):

                            # solving gradient for input feature map that caused the elements in r, c position of every output filter
                            # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimention using sum

                            _ = out.grad[:, r, c, :].reshape(F, 1, 1, 1, N)
                            pad_x_grad[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, K.data.reshape(*K.shape, 1)), axis=0)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, p[0]:pad_x_grad.shape[1]-p[0], p[1]:pad_x_grad.shape[2]-p[1], :]

                # return (K.grad, x.grad)

            node = {'func': 'conv2d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def conv_transpose2d_old(self, x, K, s=(1, 1), p=(0, 0), a=(0, 0)):
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        # 2d convolutional transpose operation - simple but inefficient implementation
        # ConvTranspose2d lasyer uses conv_transpose2d_faster for faster computation
        if not isinstance(s, tuple):
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)
        if not isinstance(a, tuple):
            a = (a, a)

        F = K.shape[0]     # number of filters - here number of feature input planes
        C = K.shape[1]     # number of input channels - here number of image output planes
        k = K.shape[2:]    # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel's shape, of some cth-channel in a kth-filter
        i = x.shape[1:-1]  # input shape of any channel of the input feature map before padding
        N = x.shape[-1]    # batch size of the input
        o = tuple((map(lambda i, k, s, p, a: int((i - 1)*s + a + k - 2*p), i, k, s, p, a)))
        pad_o = tuple(map(lambda o, p: o + 2*p, o, p))

        pad_out = np.zeros((C, *pad_o, N))  # output feature maps

        # convolution function computing cross-correlation instead of actual convolution like conv2d
        kernel = K.data.reshape(*K.shape, 1)

        for r in range(x.shape[1]):
            for c in range(x.shape[2]):

                # computing output image feature map by convolving across each element of input feature map with kernel
                _ = x.data[:, r, c, :].reshape(F, 1, 1, 1, N)
                pad_out[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, kernel), axis=0)

        # cutting the padded portion from the input-feature-map's gradient
        # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
        out = pad_out[:, p[0]:pad_out.shape[1]-p[0], p[1]:pad_out.shape[2]-p[1], :]

        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('conv_transpose2d')

                pad_out_grad = np.pad(out.grad, ((0, 0), p, p, (0, 0)), mode='constant')
                pad_out_grad = pad_out_grad.reshape(1, *pad_out_grad.shape)

                if K.eval_grad:

                    for r in range(x.shape[1]):
                        for c in range(x.shape[2]):

                            # solving gradient for each kernel filter
                            _ = x.data[:, r, c, :].reshape(F, 1, 1, 1, N)
                            # updating the kernel filter set gradient - there will be RxC such updates
                            K.grad += np.sum(np.multiply(_, pad_out_grad[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]), axis = -1)

                if x.eval_grad:

                    for r in range(x.shape[1]):
                        for c in range(x.shape[2]):

                            # solving gradient for input feature map
                            x.grad[:, r, c, :] += np.sum(np.multiply(pad_out_grad[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :], kernel), axis=(1, 2, 3))

                # return (K.grad, x.grad)

            node = {'func': 'conv_transpose2d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def conv_transpose2d(self, x, K, s=(1, 1), p=(0, 0), a=(0, 0)):
        # faster 2d convolution operation

        if not isinstance(s, tuple):
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)
        if not isinstance(a, tuple):
            a = (a, a)

        F = K.shape[0]      # number of input filters
        C = K.shape[1]      # number of output channels
        i = x.shape[1:-1]   # input channel shape
        k = K.shape[2:]     # kernel filter shape
        N = x.shape[-1]     # Batch size

        o = tuple((map(lambda i, k, s, p, a: int((i - 1)*s + a + k - 2*p), i, k, s, p, a)))
        pad_o = tuple(map(lambda o, p: o + 2*p, o, p))

        pad_out = np.zeros((C, *pad_o, N))

        for r in range(x.shape[1]):
            for c in range(x.shape[2]):

                # computing output image feature map by convolving across each element of input feature map with kernel
                _ = x.data[:, r, c, :].reshape(F, 1, 1, 1, N)
                pad_out[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, K.data.reshape(*K.shape, 1)), axis=0)

        # cutting the padded portion from the input-feature-map's gradient
        # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
        out = pad_out[:, p[0]:pad_out.shape[1]-p[0], p[1]:pad_out.shape[2]-p[1], :]

        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('conv_transpose2d')

                # padding the output gradient
                pad_out_grad = np.pad(out.grad, ((0, 0), p, p, (0, 0)), mode='constant')

                # get strided view of padded output gradient by picking appropriate strides
                shape = (C, *k, *i, N)
                strides = (pad_o[0] * pad_o[1] * N, pad_o[1] * N, N, s[0] * pad_o[1] * N, s[1] * N, 1)
                strides = pad_out_grad.itemsize * np.array(strides)
                stride_out_grad = np.lib.stride_tricks.as_strided(pad_out_grad, shape=shape, strides=strides)
                out_grad_cols = np.ascontiguousarray(stride_out_grad)
                out_grad_cols = out_grad_cols.reshape(C * k[0] * k[1], i[0] * i[1] * N)

                if K.eval_grad:
                    K.grad += np.ascontiguousarray(x.data.reshape(F, -1).dot(out_grad_cols.T).reshape(K.shape))

                if x.eval_grad:
                    x_grad = K.data.reshape(F, -1).dot(out_grad_cols)

                    # Reshape the gradient
                    x_grad = x_grad.reshape(F, *i, N)
                    x.grad += np.ascontiguousarray(x_grad)

                # return (K.grad, x.grad)

            node = {'func': 'conv_transpose2d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def max_pool2d(self, x, k=None, s=None, p=(0, 0)):    # maxpool layer(no params), used generally after Conv2d - simple but inefficient implementation
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        if s is None:
            s = k
        if not isinstance(k, tuple):
            k = (k, k)
        if not isinstance(s, tuple):
            s = (s, s)

        F = x.shape[0]     # number of input filter planes
        i = x.shape[1:-1]  # input shape of any channel of the input feature map before padding
        N = x.shape[-1]    # Batch size
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        out = np.zeros((F, *o, N))

        pad_x = np.pad(x.data, ((0, 0), p, p, (0, 0)), mode='constant')

        for r in range(out.shape[1]):       # convolving operation here(kinda)
            for c in range(out.shape[2]):   # traversing rous and columns of feature map

                # Selecting max element in the current position where kernel sits on feature map
                # The kernel moves in a convolution manner similar to conv2d
                _ = pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]
                out[:, r, c, :] = np.max(_, axis=(1, 2))

                if self.grad_mode:      # seems inefficient; will improve this whole maxpool op later

                    # Also storing value 1 at locations in the input that caused the output values(max locations); makes life easy during backprop
                    # if multiple 0s occur and max is 0 then it shouldn't count. weeding out such cases by assigning
                    # NaN and later zeroing out their gradient locations too; this was a bug which is fixed now :)
                    out[:, r, c, :][out[:, r, c, :] == 0] = np.nan
                    _ -= out[:, r, c, :].reshape(F, 1, 1, N)
                    _[np.isnan(_)] = -1     # removing all zeros locations
                    # can't use '_' object from above for the below assignment, so using the entire notation :(
                    pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] = np.where(pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] < 0, 0, 1.0)
                    out[:, r, c, :][np.isnan(out[:, r, c, :])] = 0

        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('maxpool2d')
                if x.eval_grad:

                    for r in range(out.shape[1]):
                        for c in range(out.shape[2]):

                            # multiplying each 'mask' like volume(single 1s in the volumes along all batches) with the gradient
                            # at region whose value was caused by the mask region's input
                            pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] *= out.grad[:, r, c, :].reshape(F, 1, 1, N)

                    # cutting the padded portion from the input gradient
                    # and updating the gradient of actual input(non-padded) - unpadding and updating
                    x.grad += pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :]

                # return (x.grad)

            node = {'func': 'maxpool', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def max_pool2d_faster(self, x, k=(2, 2), s=(2,2), p=(0, 0)):    # maxpool layer(no params)
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        F = x.shape[0]     # number of input filter planes
        i = x.shape[1:-1]  # input shape of any channel of the input feature map before padding
        N = x.shape[-1]    # Batch size

        # Figure out output dimensions
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        # padding the input
        pad_x = np.pad(x.data, ((0, 0), p, p, (0, 0)), mode='constant')

        # get strided view of padded input by picking appropriate strides
        shape = (F, *k, *o, N)
        strides = (pad_i[0] * pad_i[1] * N, pad_i[1] * N, N, s[0] * pad_i[1] * N, s[1] * N, 1)
        strides = pad_x.itemsize * np.array(strides)
        stride_x = np.lib.stride_tricks.as_strided(pad_x, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(stride_x)
        x_cols = x_cols.reshape(F, k[0] * k[1], *o, N)

        # store indices of the max location of each patch
        max_indices = np.argmax(x_cols, axis=1)

        out = np.max(x_cols, axis=1)
        out = Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # print('maxpool2d')
                if x.eval_grad:

                    for r in range(out.shape[1]):
                        for c in range(out.shape[2]):

                            # multiplying each 'mask' like volume(single 1s in the volumes along all batches) with the gradient
                            # at region whose value was caused by the mask region's input
                            pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] *= out.grad[:, r, c, :].reshape(F, 1, 1, N)

                    # cutting the padded portion from the input gradient
                    # and updating the gradient of actual input(non-padded) - unpadding and updating
                    x.grad += pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :]

                # return (x.grad)

            node = {'func': 'maxpool', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def dropout(self, x, p=0.5):    # dropout regularization layer!
        # useful: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

        if self.grad_mode:
            # drop activation units randomly during training
            # a unit is present with probability p
            dropout_mask = np.random.binomial(np.ones(x.shape, dtype='int64'), p)

        else:
            # scale activations of units by p during testing
            # units are always present
            dropout_mask = p

        # drop/sclae
        out = Parameter(data=dropout_mask*x.data, graph=self)

        if self.grad_mode:
            def backward():
                # print('dropout')
                if x.eval_grad:
                    x.grad += out.grad*dropout_mask # only activated units get gradients

                # return x.grad

            node = {'func': 'dropout', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # hidden and output units activations
    def relu(self, z):      # element wise ReLU activations
        out = Parameter(data=np.maximum(z.data, 0), graph=self)

        if self.grad_mode:
            def backward():
                # print('relu')
                if z.eval_grad:
                    z.grad += out.grad.copy()
                    z.grad[z.data < 0] = 0

                # return z.grad

            node = {'func': 'relu', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def lrelu(self, z, alpha=1e-2):      # element wise Leaky ReLU activations
        out = Parameter(data=np.maximum(z.data, alpha * z.data), graph=self)

        if self.grad_mode:
            def backward():
                # print('lrelu')
                if z.eval_grad:
                    z.grad += out.grad.copy()
                    z.grad[z.data < 0] *= alpha

                # return z.grad

            node = {'func': 'lrelu', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def sigmoid(self, z):   # element wise sigmoid activations
        shape = z.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = 1.0/(1.0 + np.exp(-1.0*z.data))

        if self.grad_mode:
            def backward():
                # print('sigmoid')
                if z.eval_grad:
                    z.grad += np.multiply(np.multiply(out.data, 1.0 - out.data), out.grad)

                # return z.grad

            node = {'func': 'sigmoid', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def softmax(self, z):   # calculates probs for the unnormalized log probabilities of previous layer
        shape = z.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.exp(z.data - np.max(z.data)) / np.sum(np.exp(z.data - np.max(z.data)), axis=0).reshape(1, -1)

        if self.grad_mode:
            def backward():
                # print('softmax')
                if z.eval_grad:
                    # directly coding the end result instead of formula - easy this way
                    z.grad += out.data - np.where(out.grad == 0, 0, 1.0)

                # return z.grad

            node = {'func': 'softmax', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def tanh(self, z):      # element wise tanh activations
        out = Parameter(data=np.tanh(z.data), graph=self)

        if self.grad_mode:
            def backward():
                # print('tanh')
                if z.eval_grad:
                    z.grad += np.multiply(1 - np.multiply(out.data, out.data), out.grad)

                # return z.grad

            node = {'func': 'tanh', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # data manipulation/view functions
    def split(self, W, sections=1, axis=0):
        outs = np.split(W.data, sections, axis=axis)
        outs_list = list()
        for e in outs:
            o = Parameter(data=e, graph=self)
            outs_list.append(o)

        if self.grad_mode:
            def backward():
                # print('split')
                outs_grads = [o.grad for o in outs_list]
                if W.eval_grad:
                    W.grad += np.concatenate(outs_grads, axis=axis)

                # return W.grad

            node = {'func': 'split', 'inputs': [W], 'outputs': outs_list, 'backprop_op': lambda: backward()}
            for out in outs_list:
                out.node_id = len(self.nodes)
            self.nodes.append(node)

        return outs_list

    def cat(self, inputs_list, axis=0):
        indices = [input.shape[axis] for input in inputs_list]
        indices = [sum(indices[:_+1]) for _ in range(len(indices))]
        out = Parameter(data=np.concatenate(inputs_list, axis=axis), graph=self)

        if self.grad_mode:
            def backward():
                # print('cat')
                input_grads = np.split(out.grad, indices, axis=axis)
                for _ in range(len(inputs_list)):
                    if inputs_list[_].eval_grad:
                        inputs_list[_].grad += input_grads[_]

                # return *[input.grad for input in inputs_list]

            node = {'func': 'cat', 'inputs': [inputs_list], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def T(self, x):     # transpose
        out = Parameter(data=x.data.T, graph=self)

        if self.grad_mode:
            def backward():
                # print('T')
                if x.eval_grad:
                    x.grad += out.grad.T

                # return x.grad

            node = {'func': 'transpose', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def reshape(self, x, new_shape=None):
        old_shape = x.shape
        batch_size = old_shape[-1]

        if new_shape == None:   # flatten
            new_shape = x.data.reshape(-1, batch_size).shape
        else:
            new_shape = (*new_shape, batch_size)
        out = Parameter(new_shape, init_zeros=True, graph=self)
        out.data = x.data.reshape(new_shape)

        if self.grad_mode:
            def backward():
                # print('reshape')
                if x.eval_grad:
                    x.grad += out.grad.reshape(old_shape)

                # return x.grad

            node = {'func': 'reshape', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out


G = ComputationalGraph()



# generic module class to add useful features like save/load model from files, get parameters etc.
class Module(object):
    def __init__(self):
        pass

    def __repr__(self):
        module_schema = str(self.__class__.__name__) + '(\n'

        for name, layer in self.get_module_layers().items():
            module_schema += '  ' + str(name) + ': ' + str(layer) + '\n'

        module_schema += ')'

        return module_schema

    def save(self, file=None):  # model.save() - saves the state of the network
        print('saving model...')
        save_dict = dict()

        module_layers = self.get_module_layers()
        for layer_name, layer in module_layers.items():

            layer_params = layer.get_module_params()
            for param_name, param in layer_params.items():
                layer_params[param_name] = param.data

            module_layers[layer_name] = layer_params

        module_params = self.get_module_params()
        for param_name, param in module_params.items():
            module_params[param_name] = param.data

        save_dict['module_layers'] = module_layers
        save_dict['module_params'] = module_params

        if file == None:
            file = self.__class__.__name__+'.npy'

        np.save(file, save_dict)
        print('Successfully saved model in {}'.format(file))

    def load(self, file=None):  # model.load() - loads the state of net from a file
        print('loading model...')

        if file == None:
            file = self.__class__.__name__+'.npy'

        load_dict = np.load(file, allow_pickle=True).item()
        module_layers_stored = load_dict['module_layers']
        module_params_stored = load_dict['module_params']

        module_layers_actual = self.get_module_layers()
        module_params_actual = self.get_module_params()

        for layer_name, layer_stored in module_layers_stored.items():
            if layer_name in module_layers_actual:
                for param_name, param in layer_stored.items():
                    layer_actual = module_layers_actual[layer_name]
                    setattr(layer_actual, str(param_name), Parameter(data=param))

        for param_name, param in module_params_stored.items():
            if param_name in module_params_actual:
                setattr(self,str(param_name), Parameter(data=param))

        print('Successfully loaded model from {}'.format(file))

    def get_module_layers(self):   # returns a dictionary of parametrized layers in the module

        attributes = self.__dict__
        layers = ['Linear', 'Conv2d', 'ConvTranspose2d', 'LSTM', 'RNN', 'BatchNorm', 'Maxpool2d', 'Dropout']

        module_layers = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in layers:
                module_layers[name] = attributes[name]

        return module_layers

    def get_module_params(self):    # returns a dictionary of parameters in the module

        attributes = self.__dict__

        module_params = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in ['Parameter']:
                if attributes[name].eval_grad:
                    module_params[name] = attributes[name]

        return module_params

    def parameters(self):   # access parameters of the module with this function

        all_params = list()

        for layer in list(self.get_module_layers().values()):
            all_params.extend(layer.parameters())

        all_params.extend(list(self.get_module_params().values()))

        return all_params


# linear affine transformation: y = Wx + b
# the general feed-forward network
class Linear(Module):
    def __init__(self, input_features=0, output_features=0, bias=True, graph=G):
        super(Linear, self).__init__()
        self.input_features = input_features  # previous layer units
        self.output_features = output_features  # next layer units
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.input_features)
        self.W = Parameter((self.output_features, self.input_features), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # weight volume
        self.b = Parameter((self.output_features, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)   # bias vector

    def __repr__(self):
        return('Linear(input_features={}, output_features={}, bias={})'.format(
            self.input_features, self.output_features, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):
        # making the input compatible with graph operations
        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # flatten the input if it came from layers like Conv2d
        if len(x.shape) > 2:
            x = self.graph.reshape(x)

        # y = Wx + b
        out = self.graph.dot(self.W, x) # matmul

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-1,))

        return out


# 1D convolutional neural network
class Conv1d(Module):
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1,), padding=(0,), bias=True, graph=G):
        super(Conv1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size,)
        if not isinstance(stride, tuple):
            stride = (stride,)
        if not isinstance(padding, tuple):
            padding = (padding,)

        self.kernel_size = kernel_size
        self.filter_size = (self.input_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / (self.input_channels * self.kernel_size[0]))
        self.K = Parameter((self.output_channels, *self.filter_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return('Conv1d({}, {}, kernel_size={}, stride={}, padding={}, bias={})'.format(
            self.input_channels, self.output_channels, self.kernel_size, self.stride, self.padding, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # convolution operation
        out = self.graph.conv1d(x, self.K, self.stride, self.padding)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-2, -1))

        return out


# 2D convolutional neural network
class Conv2d(Module):
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1, 1), padding=(0, 0), bias=True, graph=G):
        super(Conv2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.filter_size = (self.input_channels, *(self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / (self.input_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.K = Parameter((self.output_channels, *self.filter_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return('Conv2d({}, {}, kernel_size={}, stride={}, padding={}, bias={})'.format(
            self.input_channels, self.output_channels, self.kernel_size, self.stride, self.padding, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # convolution operation
        out = self.graph.conv2d(x, self.K, self.stride, self.padding)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-3, -2, -1))

        return out


# 2d transposed convolutional neural network
class ConvTranspose2d(Module):
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1, 1), padding=(0, 0), a=(0, 0), bias=True, graph=G):
        super(ConvTranspose2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)
        if not isinstance(a, tuple):
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
        root_k = np.sqrt(1. / (self.output_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.K = Parameter((self.input_channels, *self.filter_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return('ConvTranspose2d({}, {}, kernel_size={}, stride={}, padding={}, a={}, bias={})'.format(
            self.input_channels, self.output_channels, self.kernel_size, self.stride, self.padding, self.a, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        # convolution transpose operation
        out = self.graph.conv_transpose2d(x, self.K, self.stride, self.padding, self.a)

        if self.bias:   # adding bias
            out = self.graph.add(out, self.b, axis=(-3, -2, -1))

        return out


# sequence models: LSTM cell
class LSTM(Module):
    def __init__(self, input_size, hidden_size, bias=True, graph=G):
        super(LSTM, self).__init__()
        self.input_size = input_size    # size of the input at each recurrent tick
        self.hidden_size = hidden_size  # size of hidden units h and c
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.hidden_size)
        self.W_ih = Parameter((4*self.hidden_size, self.input_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)    # input to hidden weight volume
        self.W_hh = Parameter((4*self.hidden_size, self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)   # hidden to hidden weight volume
        self.b_ih = Parameter((4*self.hidden_size, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # input to hidden bias vector
        self.b_hh = Parameter((4*self.hidden_size, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # hidden to hidden bias vector

    def __repr__(self):
        return('LSTM(input_size={}, hidden_size={}, bias={})'.format(
            self.input_size, self.hidden_size, self.bias))

    def __call__(self, x, hidden):  # easy callable
        return self.forward(x, hidden)

    def forward(self, x, hidden):

        h, c = hidden

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)


        i_h = self.graph.dot(self.W_ih, x)
        if self.bias:
            i_h = self.graph.add(i_h, self.b_ih, axis=(-1,))

        h_h = self.graph.dot(self.W_hh, h)
        if self.bias:
            h_h = self.graph.add(h_h, self.b_hh, axis=(-1,))

        gates = self.graph.add(i_h, h_h)

        # forget, input, gate(also called cell gate - different from cell state), output gates of the lstm cell
        # useful: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        f, i, g, o = self.graph.split(gates, sections=4, axis=0)

        f = self.graph.sigmoid(f)
        i = self.graph.sigmoid(i)
        g = self.graph.tanh(g)
        o = self.graph.sigmoid(o)

        c = self.graph.add(self.graph.multiply(f, c), self.graph.multiply(i, g))
        h = self.graph.multiply(o, self.graph.tanh(c))

        return (h, c)


# sequence models: RNN cell
class RNN(Module):
    def __init__(self, input_size, hidden_size, bias=True, graph=G):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.hidden_size)
        self.W_ih = Parameter((self.hidden_size, self.input_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.W_hh = Parameter((self.hidden_size, self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b_ih = Parameter((self.hidden_size, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)    # not much use
        self.b_hh = Parameter((self.hidden_size, 1), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return('RNN(input_size={}, hidden_size={}, bias={})'.format(
            self.input_size, self.hidden_size, self.bias))

    def __call__(self, x, hidden):  # easy callable
        return self.forward(x, hidden)

    def forward(self, x, hidden):

        h = hidden

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        i_h = self.graph.dot(self.W_ih, x)
        if self.bias:
            i_h = self.graph.add(i_h, self.b_ih, axis=(-1,))

        h_h = self.graph.dot(self.W_hh, h)
        if self.bias:
            h_h = self.graph.add(h_h, self.b_hh, axis=(-1,))

        h = self.graph.add(i_h, h_h)

        h = self.graph.tanh(h)

        return h


# batch normalization layer
class BatchNorm(Module):
    def __init__(self, hidden_shape, axis=-1, momentum=0.9, bias=True, graph=G):
        super(BatchNorm, self).__init__()
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
        self.m = Parameter(data=np.sum(np.zeros(shape), axis=self.axis,
                     keepdims=True) / shape[self.axis], eval_grad=False, graph=self.graph)    # moving mean - not trainable
        self.v = Parameter(data=np.sum(np.ones(shape), axis=self.axis,
                     keepdims=True) / shape[self.axis], eval_grad=False, graph=self.graph)    # moving variance - not trainable

    def __repr__(self):
        return('BatchNorm({}, axis={}, momentum={}, bias={})'.format(
            self.hidden_shape, self.axis, self.momentum, self.bias))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        if self.graph.grad_mode:    # training
            # useful: https://arxiv.org/abs/1502.03167

            batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size/channel size
            batch_size.data.fill(float(x.shape[self.axis]))

            # calculate mean and variance
            mean = self.graph.divide(self.graph.sum(x, axis=self.axis), batch_size)
            centered = self.graph.subtract(x, mean, axis=self.axis)
            var = self.graph.divide(self.graph.sum(self.graph.power(centered, 2), axis=self.axis), batch_size)

            self.m.data = self.momentum * self.m.data + (1 - self.momentum) * mean.data
            self.v.data = self.momentum * self.v.data + (1 - self.momentum) * var.data

            # normalize the data to zero mean and unit variance
            normalized = self.graph.multiply(centered, self.graph.power(var, -0.5), axis=self.axis)

        else:   # testing

            centered = np.subtract(x.data, self.m.data)
            normalized = np.multiply(centered, np.power(self.v.data + 1e-6, -0.5))
            normalized = Parameter(data=normalized, eval_grad=False, graph=self.graph)

        # scale and shift
        out = self.graph.multiply(normalized, self.gamma, axis=(-1,))    # scale

        if self.bias:   # shift
            out = self.graph.add(out, self.beta, axis=(-1,))

        return out


# maxpool2d layer - non-parametrized layer
class Maxpool2d(Module):
    def __init__(self, kernel_size=None, stride=None, padding=(0, 0), graph=G):
        super(Maxpool2d, self).__init__()

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
        return('Maxpool2d(kernel_size={}, stride={}, padding={})'.format(
            self.kernel_size, self.stride, self.padding))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        out = self.graph.max_pool2d(x, k=self.kernel_size, s=self.stride, p=self.padding)

        return out


# dropout layer - non-parametrized layer
class Dropout(Module):
    def __init__(self, p=0.5, graph=G):
        super(Dropout, self).__init__()
        self.p = p
        self.graph = graph

    def __repr__(self):
        return('Dropout(p={})'.format(self.p))

    def __call__(self, x):  # easy callable
        return self.forward(x)

    def forward(self, x):

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        out = self.graph.dropout(x, p=self.p)

        return out


# |    ||
#
# ||   |_
#
# is this loss? Yes, it is.
class Loss:
    def __init__(self, loss_fn=None, graph=G):
        self.loss_fn = loss_fn
        self.graph = graph

    def loss(self, y_out, y_true):

        if self.loss_fn == 'MSELoss':
            return self.mse_loss(y_out, y_true)
        elif self.loss_fn == 'CrossEntropyLoss':
            return self.cross_entropy_loss(y_out, y_true)
        elif self.loss_fn == 'BCELoss':
            return self.bce_loss(y_out, y_true)
        elif self.loss_fn == 'JSDivLoss':
            return self.js_divergence_loss(y_out, y_true)
        elif self.loss_fn == 'TestLoss':
            return self.test_loss(y_out)
        else:
          raise 'No such loss function'

    def __repr__(self):
        return('Loss(loss_fn={})'.format(self.loss_fn))

    def mse_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, eval_grad=False, graph=self.graph)

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size
        batch_size.data.fill(float(y_true.shape[-1]))

        # L = (y_out - y_true)^2
        l = self.graph.sum(self.graph.multiply(self.graph.subtract(y_out, y_true), self.graph.subtract(y_out, y_true)))
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = self.graph.divide(l, batch_size)

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def cross_entropy_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, eval_grad=False, graph=self.graph)

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size
        batch_size.data.fill(float(y_true.shape[-1]))

        neg_one = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph)
        neg_one.data.fill(-1.0)  # just a -1 to make the l.grad look same in all the loss defs (dl/dl = 1)

        # KL(P || Q): Summation(P*log(P)){result: 0} - Summation(P*log(Q))
        l = self.graph.multiply(self.graph.sum(self.graph.multiply(y_true, self.graph.log(y_out))), neg_one)
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = self.graph.divide(l, batch_size)

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def bce_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, eval_grad=False, graph=self.graph)

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size
        batch_size.data.fill(float(y_true.shape[-1]))

        neg_one = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph)
        neg_one.data.fill(-1.0)  # just a -1 to make the l.grad look same in all the loss defs (dl/dl = 1)

        one = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph)
        one.data.fill(1.0)

        # class 2 output: 1 - c1
        c2 = self.graph.multiply(self.graph.subtract(y_out, one), neg_one)
        # class 2 target: 1 - t1
        t2 = self.graph.multiply(self.graph.subtract(y_true, one), neg_one)

        # -Summation(t1*log(c1))
        l1 = self.graph.multiply(self.graph.sum(self.graph.multiply(y_true, self.graph.log(y_out))), neg_one)
        # -Summation((1 - t1)*log(1 - c1))
        l2 = self.graph.multiply(self.graph.sum(self.graph.multiply(t2, self.graph.log(c2))), neg_one)
        # loss = -Summation(t1*log(c1)) -Summation((1 - t1)*log(1 - c1))
        l = self.graph.add(l1, l2)
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = self.graph.divide(l, batch_size)

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def js_divergence_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, eval_grad=False, graph=self.graph)

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size
        batch_size.data.fill(float(y_true.shape[-1]))

        two = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph)
        two.data.fill(2.0)   # just a 2 :p

        neg_one = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph)
        neg_one.data.fill(-1.0)  # just a -1 to make the l.grad look same in all the loss defs (dl/dl = 1)

        # mean probability: (P + Q)/2
        y_mean = self.graph.divide(self.graph.add(y_out, y_true), two)
        # KL(P || (P + Q)/2): Summation(P*log(P)){result: 0} - Summation(P*log((P+Q)/2))
        kl_1 = self.graph.multiply(self.graph.sum(self.graph.multiply(y_true, self.graph.log(y_mean))), neg_one)
        # KL(Q || (P + Q)/2): Summation(Q*log(Q)) - Summation(Q*log((P+Q)/2))
        kl_2 = self.graph.add(self.graph.multiply(y_out, self.graph.log(y_out)), self.graph.multiply(self.graph.multiply(y_out, self.graph.log(y_mean)), neg_one))   # !!!!!
        # JS(P, Q) = 1/2*(KL(P || (P + Q)/2) + KL(Q || (P + Q)/2))
        l = self.graph.divide(self.graph.add(kl_1, kl_2), two)
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = self.graph.divide(l, batch_size)

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def test_loss(self, y_out):

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False, graph=self.graph) # mini-batch size
        batch_size.data.fill(float(y_out.shape[-1]))

        # a test loss score function that measures the sum of elements of each output vector as the loss of that sample
        # helps identify leaks in between samples in a batch
        l = self.graph.sum(y_out)
        l = self.graph.divide(l, batch_size)

        l.grad[0, 0] = 1.0

        return l

    #define loss functions


# Optimizers to take that drunken step down the hill
class Optimizer:
    def __init__(self, parameters, optim_fn='SGD', lr=3e-4, momentum=0.9, eps=1e-8, beta1=0.9, beta2=0.999, rho=0.95, graph=G):
        self.parameters = parameters  # a list of all layers of the model
        self.optim_fn = optim_fn    # the optimizing function(SGD, Adam, Adagrad, RMSProp)
        self.lr = lr    # alpha: size of the step to update the parameters
        self.momentum = momentum
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.graph = graph
        self.t = 0  # iteration count
        self.m = list() # (momentumAdam/Adagrad/Adadelta)
        self.v = list() # (Adam/Adadelta)

        if self.optim_fn != 'SGD' or self.momentum > 0.0:

            # only vanilla SGD doesn't require any lists
            # momentum in SGD: stores deltas of previous iterations
            # Adam: 1st moment(mean) vector of gradients
            # Adagrad: stores square of gradient
            # Adadelta: Accumulates gradient
            for parameter in self.parameters:
                self.m.append(np.zeros(parameter.shape))

            if self.optim_fn == 'Adam' or self.optim_fn == 'Adadelta':

                # Adam: 2nd moment(raw variance here) of gradients
                # Adadelta: Accumulates updates
                for parameter in self.parameters:
                    self.v.append(np.zeros(parameter.shape))

    def __repr__(self):
        return('Optimizer(optim_fn={}, lr={}, momentum={})'.format(
            self.optim_fn, self.lr, self.momentum))

    # a very important step in learning time
    def zero_grad(self):
        # clearing out the backprop operations from the list
        self.graph.nodes = list()
        self.graph.node_count = 0

        # resetting the gradients of model parameters to zero
        for parameter in self.parameters:
            parameter.grad = np.zeros(parameter.shape)

    def step(self):
        # useful: https://arxiv.org/pdf/1609.04747.pdf

        if self.optim_fn == 'SGD':
            return self.stochastic_gradient_descent()
        elif self.optim_fn == 'Adam':
            return self.adam()
        elif self.optim_fn == 'Adagrad':
            return self.adagard()
        elif self.optim_fn == 'Adadelta':
            return self.adadelta()
        elif self.optim_fn == 'RMSProp':
            return self.rms_prop()
        else:
          raise 'No such optimization function'

    # Stochastic Gradient Descent optimization function
    def stochastic_gradient_descent(self):
        if self.t < 1: print('using SGD')

        self.t += 1
        for p in range(len(self.parameters)):

            if self.momentum > 0.0:
                # momentum update
                self.m[p] = self.momentum * self.m[p] + self.lr * self.parameters[p].grad

                # Update parameters with momentum SGD
                self.parameters[p].data -= self.m[p]

            else:
                # Update parameters with vanilla SGD
                self.parameters[p].data -= self.lr * self.parameters[p].grad

    # Adam optimization function
    def adam(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf
        if self.t < 1: print('using Adam')

        self.t += 1
        for p in range(len(self.parameters)):

            # Update biased first moment estimate
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * self.parameters[p].grad

            # Update biased second raw moment estimate
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * self.parameters[p].grad * self.parameters[p].grad

            # (Compute bias-corrected first moment estimate
            m_cap = self.m[p] / (1 - np.power(self.beta1, self.t))

            # Compute bias-corrected second raw moment estimate
            v_cap = self.v[p] / (1 - np.power(self.beta2, self.t))

            # Update parameters
            self.parameters[p].data -= self.lr * m_cap / (np.sqrt(v_cap) + self.eps)

    # Adagrad optimization function
    def adagard(self):
        if self.t < 1: print('using Adagrad')

        self.t += 1
        for p in range(len(self.parameters)):

            # update memory
            self.m[p] += self.parameters[p].grad * self.parameters[p].grad

            # Update parameters
            self.parameters[p].data -= self.lr * self.parameters[p].grad / np.sqrt(self.m[p] + self.eps)

    # Adadelta optimization function
    def adadelta(self):
        # useful: https://arxiv.org/pdf/1212.5701.pdf
        if self.t < 1: print('using Adadelta')

        self.t += 1
        for p in range(len(self.parameters)):

            # Accumulate Gradient:
            self.m[p] = self.rho * self.m[p] + (1 - self.rho) * self.parameters[p].grad * self.parameters[p].grad

            # Compute Update:
            delta = -np.sqrt((self.v[p] + self.eps) / (self.m[p] + self.eps)) * self.parameters[p].grad

            # Accumulate Updates:
            self.v[p] = self.rho * self.v[p] + (1 - self.rho) * delta * delta

            # Apply Update:
            self.parameters[p].data += delta

    # RMSProp optimization function
    def rms_prop(self):
        # useful: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        if self.t < 1: print('using RMSProp')

        self.t += 1
        for p in range(len(self.parameters)):

            # Accumulating moving average of the square of the Gradient:
            self.m[p] = self.rho * self.m[p] + (1 - self.rho) * self.parameters[p].grad * self.parameters[p].grad

            # Apply Update:
            self.parameters[p].data -= self.lr * self.parameters[p].grad / (np.sqrt(self.m[p]) + self.eps)

    #define optimizers


# initializations and utitlity functions
def manual_seed(seed=2357):
    np.random.seed(seed)


# draw the Computational Graph of the ai program
def draw_graph(filename=None, format='svg', graph=G):
    # visualization procedure referred from karpathy's micrograd

    from graphviz import Digraph

    label = 'Computational Graph of {}'.format(filename)
    dot = Digraph(graph_attr={'rankdir': 'LR', 'label': label}, node_attr={'rankdir': 'TB'})

    for cell in graph.nodes:

        # add the op to nodes
        dot.node(name=str(id(cell['backprop_op'])), label=cell['func'], shape='doublecircle',)

        for input in cell['inputs']:

            # add the input to nodes
            color = None if input.eval_grad else 'red'
            dot.node(name=str(id(input)), label='{}'.format(input.node_id), shape='circle', color=color)
            # forward pass edge from input to op
            dot.edge(str(id(input)), str(id(cell['backprop_op'])))

            # # backprop pass edge from op to input
            # if input.eval_grad:
            #     dot.edge(str(id(cell['backprop_op'])), str(id(input)), color='red')

        for output in cell['outputs']:

            # add the output to nodes
            dot.node(name=str(id(output)), label='{}'.format(output.node_id), shape='circle')
            # forward pass edge from op to output
            dot.edge(str(id(cell['backprop_op'])), str(id(output)))

            # # backward pass edge from output to op
            # dot.edge(str(id(output)), str(id(cell['backprop_op'])), color='red')

    dot.render(format=format, filename=filename, directory='assets', cleanup=True)


# clip the gradients of parameters by value
def clip_grad_value(parameters, clip_value):

    for p in parameters:
        # clip gradients by value
        p.grad = np.clip(p.grad, -clip_value, clip_value)



# TODO: define regularizations, asserts, batch, utils, GPU support, examples
