"""
AI library in python using numpy

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://srirambandi.github.io

MIT License
"""

import numpy as np


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), data=None, eval_grad=True, node_id=0, graph=None,
                init_zeros=False, init_ones=False, constant=1.0,
                uniform=False, low = -1.0, high = 1.0,
                mean = 0.0, std = 0.01):

        # properties
        self.shape = shape
        self.data = data
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
            self.data = (self.high - self.low) * np.random.rand(*self.shape) + self.low

        else:
            # random initiation with gaussian distribution
            self.data = self.std*np.random.randn(*self.shape) + self.mean

        # setting gradient of parameter wrt some scalar, as zeros
        self.grad = np.zeros(self.shape)

    def __str__(self):
        parameter_schema = 'Parameter(shape={}, eval_grad={}) containing:\n'.format(self.shape, self.eval_grad)
        parameter_schema += 'Data: {}'.format(self.data)

        return parameter_schema

    # this function computes the gradients of the parameters, by executing
    # the backprop ops in reverse order to the forward propagation with chain rule
    def backward(self, grad=None, to=None):
        # assign gradient
        if grad is not None:
            self.grad = np.array(grad)

        if to is None:
            stop = 0    # execute backward all the way to start
        else:
            stop = to.node_id + 1  # execute backward  to just before this node

        for node in reversed(self.graph.nodes[stop:int(self.node_id) + 1]):
            node['backprop_op']()       # executing the back-propagation operation

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

    def __truediv_(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, 'Objects not of same shape. Use G.divide() with axis argument'

        return self.graph.divide(self, other)

    def __pow_(self, other, modulo):
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
        self.nodes = []

    # operations required for deep learning models and their backward operations
    def dot(self, W, x):    # dot product of vectors and matrices

        assert W.shape[1] == x.shape[0], 'shape mismatch in dot() operation'
        shape = (W.data.shape[0], x.data.shape[1])
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.dot(W.data, x.data)

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
        shape = x.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.add(x.data, y.data)

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
        shape = x.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.subtract(x.data, y.data)

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
        # not for scalar multiply
        shape = x.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.multiply(x.data, y.data)

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
        shape = x.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.divide(x.data, y.data)

        if self.grad_mode:
            def backward():
                # print('divide')
                if x.eval_grad:
                    x.grad += np.multiply(out.grad, np.divide(1.0, y.data))
                if y.eval_grad:
                    y.grad += np.sum(np.multiply(out.grad, np.multiply(out.data, np.divide(-1.0, y.data))), axis=axis).reshape(y.shape) # in case of unequal sizes of inputs

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
        out = Parameter(res.shape, init_zeros=True, graph=self)
        out.data = res

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

    def power(self, h, power):   # element wise power
        out = Parameter(h.shape, init_zeros=True, graph=self)
        out.data = np.power(h.data, power) if power >= 0 else np.power(h.data, power) + 1e-6     # numerical stability for -ve power

        if self.grad_mode:
            def backward():
                # print('power')
                if h.eval_grad:
                    h.grad += np.multiply(out.grad, power * np.power(h.data, power - 1))

                # return h.grad

            node = {'func': '**', 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def log(self, h):   # element wise logarithm
        shape = h.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.log(h.data)

        if self.grad_mode:
            def backward():
                # print('log')
                if h.eval_grad:
                    h.grad += np.multiply(out.grad, np.divide(1.0, h.data))

                # return h.grad

            node = {'func': 'log', 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # ayers operations
    def conv2d(self, x, K, s = (1, 1), p = (0, 0)):     # 2d convolution operation - simple but inefficient implementation
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        fi = K.shape[0]     # number of filters
        ch = K.shape[1]     # number of input channels
        k = K.shape[2:]     # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel's shape, of some cth-channel in a kth-filter
        i = x.shape[1:-1]   # input shape of any channel of the input feature map before padding
        batch = x.shape[-1] # batch size of the input
        output_maps_shape = (fi, *(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p)), batch) # output feature maps shape - (# of filters, o_i, o_j, batch_size)
        pad_shape = (ch, *(map(lambda i, p: i + 2*p, i, p)), batch)   # padded input feature maps shape - (channels, new_i, new_j, batch_size)

        out = np.zeros(output_maps_shape)  # output feature maps

        pad_x = np.zeros(pad_shape)
        # padded input - copying the actual input onto pad input centre
        pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :] += x.data
        pad_x = pad_x.reshape(1, *pad_shape)

        # convolution function computing cross-correlation instead of actual convolution - otherwise have to use
        # flipped kernels which doesn't effect learning
        kernel = K.data.reshape(*K.shape, 1)

        for r in range(out.shape[1]):        # convolving operation here
            for c in range(out.shape[2]):    # traversing rous and columns of feature map

                # multiplying traversed grid portions of padded input feature maps with kernel grids element-wise
                # and summing the resulting matrix to produce elements of output maps, over all filters and batches
                out[:, r, c, :] += np.sum(np.multiply(pad_x[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :], kernel), axis=(1, 2, 3))

        output_feature_maps = Parameter(out.shape, init_zeros=True, graph=self)
        output_feature_maps.data = out    # any set of output feature map from the batch, has same numeber of maps as filters in the kernel set

        if self.grad_mode:
            def backward():
                # print('conv2d')
                if K.eval_grad:

                    for r in range(output_feature_maps.shape[1]):
                        for c in range(output_feature_maps.shape[2]):

                            # solving gradient for each kernel filter that caused the elements in r, c position of every output filter
                            # in every bacth; sketch and think, with input stacked fi times to make computation fast

                            _ = output_feature_maps.grad[:, r, c, :].reshape(fi, 1, 1, 1, batch)
                            # updating the kernel filter set gradient - there will be RxC such updates
                            K.grad += np.sum(np.multiply(_, pad_x[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]), axis = -1)

                if x.eval_grad:

                    pad_x_grad = np.zeros(pad_shape)

                    for r in range(output_feature_maps.shape[1]):
                        for c in range(output_feature_maps.shape[2]):

                            # solving gradient for input feature map that caused the elements in r, c position of every output filter
                            # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimention using sum

                            _ = output_feature_maps.grad[:, r, c, :].reshape(fi, 1, 1, 1, batch)
                            pad_x_grad[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, kernel), axis=0)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, p[0]:pad_x_grad.shape[1]-p[0], p[1]:pad_x_grad.shape[2]-p[1], :]

                # return (K.grad, x.grad)

            node = {'func': 'conv2d', 'inputs': [x, K], 'outputs': [output_feature_maps], 'backprop_op': lambda: backward()}
            output_feature_maps.node_id = len(self.nodes)
            self.nodes.append(node)

        return output_feature_maps

    def conv_transpose2d(self, x, K, s = (1, 1), p = (0, 0), a = (0, 0)):     # 2d convolution transpose operation - simple but inefficient implementation
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        fi = K.shape[0]     # number of filters - here number of feature input planes
        ch = K.shape[1]     # number of input channels - here number of image output planes
        k = K.shape[2:]     # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel's shape, of some cth-channel in a kth-filter
        i = x.shape[1:-1]   # input shape of any channel of the input feature map before padding
        batch = x.shape[-1] # batch size of the input
        output_shape = tuple((map(lambda i, k, s, p, a: int((i - 1)*s + a + k - 2*p), i, k, s, p, a))) # output feature maps shape - (# of channels, o_i, o_j, batch_size)
        pad_output_img_shape = (ch, *(map(lambda o, p: o + 2*p, output_shape, p)), batch)   # padded input feature maps shape - (filters, new_i, new_j, batch_size)

        out = np.zeros(pad_output_img_shape)  # output feature maps

        # convolution function computing cross-correlation instead of actual convolution like conv2d
        kernel = K.data.reshape(*K.shape, 1)

        for r in range(x.shape[1]):
            for c in range(x.shape[2]):

                # computing output image feature map by convolving across each element of input feature map with kernel
                _ = x.data[:, r, c, :].reshape(fi, 1, 1, 1, batch)
                out[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, kernel), axis=0)

        # cutting the padded portion from the input-feature-map's gradient
        # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
        out = out[:, p[0]:out.shape[1]-p[0], p[1]:out.shape[2]-p[1], :]

        output_image = Parameter(out.shape, init_zeros=True, graph=self)
        output_image.data = out    # any set of output feature map from the batch, has same numeber of maps as filters in the kernel set

        if self.grad_mode:
            def backward():
                # print('conv2d')

                pad_output_grad = np.zeros(pad_output_img_shape)
                pad_output_grad[:, p[0]:pad_output_grad.shape[1]-p[0], p[1]:pad_output_grad.shape[2]-p[1], :] += output_image.grad
                pad_output_grad = pad_output_grad.reshape(1, *pad_output_img_shape)

                if K.eval_grad:

                    for r in range(x.shape[1]):
                        for c in range(x.shape[2]):

                            # solving gradient for each kernel filter
                            _ = x.data[:, r, c, :].reshape(fi, 1, 1, 1, batch)
                            # updating the kernel filter set gradient - there will be RxC such updates
                            K.grad += np.sum(np.multiply(_, pad_output_grad[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]), axis = -1)

                if x.eval_grad:

                    for r in range(x.shape[1]):
                        for c in range(x.shape[2]):

                            # solving gradient for input feature map
                            x.grad[:, r, c, :] += np.sum(np.multiply(pad_output_grad[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :], kernel), axis=(1, 2, 3))

                # return (K.grad, x.grad)

            node = {'func': 'conv_transpose2d', 'inputs': [x, K], 'outputs': [output_image], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return output_image

    def maxpool2d(self, x, k=(2, 2), s=(2,2), p=(0, 0)):    # maxpool layer(no params), used generally after Conv2d - simple but inefficient implementation
        # useful: https://arxiv.org/pdf/1603.07285.pdf

        fi = x.shape[0]     # number of input filter planes
        i = x.shape[1:-1]   # input shape of any channel of the input feature map before padding
        batch = x.shape[-1]
        pool_shape = (fi, *(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p)), batch) # shape after maxpool
        pad_shape = (fi, *(map(lambda i, p: i + 2*p, i, p)), batch)  # padded input shape

        out = np.zeros((pool_shape))

        pad_x = np.zeros(pad_shape)
        pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :] += x.data # copying the input onto padded matrix

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
                    _ -= out[:, r, c, :].reshape(fi, 1, 1, batch)
                    _[np.isnan(_)] = -1     # removing all zeros locations
                    # can't use '_' object from above for the below assignment, so using the entire notation :(
                    pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] = np.where(pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] < 0, 0, 1.0)
                    out[:, r, c, :][np.isnan(out[:, r, c, :])] = 0

        pool_maps = Parameter(out.shape, init_zeros=True, graph=self)
        pool_maps.data = out

        if self.grad_mode:
            def backward():
                # print('maxpool2d')
                if x.eval_grad:

                    for r in range(pool_maps.shape[1]):
                        for c in range(pool_maps.shape[2]):

                            # multiplying each 'mask' like volume(single 1s in the volumes along all batches) with the gradient
                            # at region whose value was caused by the mask region's input
                            pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] *= pool_maps.grad[:, r, c, :].reshape(fi, 1, 1, batch)

                    # cutting the padded portion from the input gradient
                    # and updating the gradient of actual input(non-padded) - unpadding and updating
                    x.grad += pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :]

                # return (x.grad)

            node = {'func': 'maxpool', 'inputs': [x], 'outputs': [pool_maps], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return pool_maps

    def dropout(self, x, p=0.5):    # dropout regularization layer!
        # useful: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        shape = x.shape
        # drop activation units randomly during training
        # a unit is present with probability p
        if self.grad_mode:
            dropout_mask = np.random.binomial(np.ones(shape, dtype='int64'), p)
        # scale activations of units by p during testing
        # units are always present
        else:
            dropout_mask = p
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = dropout_mask*x.data    # drop/sclae

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
        shape = z.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.maximum(z.data, 0)

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
        shape = z.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.maximum(z.data, alpha * z.data)

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
        shape = z.shape
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = np.tanh(z.data)

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
        outs_list = []
        for e in outs:
            o = Parameter(e.shape, init_zeros=True, graph=self)
            o.data = e
            outs_list.append(o)

        if self.grad_mode:
            def backward():
                #print('split')
                outs_grads = [o.grad for o in outs_list]
                if W.eval_grad:
                    W.grad += np.concatenate(outs_grads, axis=axis)

            node = {'func': 'split', 'inputs': [W], 'outputs': outs_list, 'backprop_op': lambda: backward()}
            for out in outs_list:
                out.node_id = len(self.nodes)
            self.nodes.append(node)

        return outs_list

    def T(self, x):     # transpose
        shape = tuple(reversed(x.shape))
        out = Parameter(shape, init_zeros=True, graph=self)
        out.data = x.data.T

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



# generic model class to add useful features like save/load model from files, get parameters etc.
class Module(object):
    def __init__(self):
        pass

    def __str__(self):
        model_schema = str(self.__class__.__name__) + '(\n'

        for name, layer in self.get_module_layers().items():
            model_schema += '  ' + str(name) + ': ' + str(layer) + '\n'

        model_schema += ')'

        return model_schema

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

    def get_module_layers(self):   # returns a dictionary of parametrized layers in the model

        attributes = self.__dict__
        parametrized_layers = ['Linear', 'Conv2d', 'ConvTranspose2d', 'LSTM', 'RNN', 'BatchNorm']

        module_layers = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in parametrized_layers:
                module_layers[name] = attributes[name]

        return module_layers

    def get_module_params(self):    # returns a dictionary of parameters in the model

        attributes = self.__dict__

        module_params = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in ['Parameter']:
                module_params[name] = attributes[name]

        return module_params

    def parameters(self):   # access parameters of the model with this function

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
        self.W = Parameter((self.output_features, self.input_features), graph=self.graph)  # weight volume
        self.b = Parameter((self.output_features, 1), init_zeros=True, graph=self.graph)   # bias vector
        self.parameters()

    def __str__(self):
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
        self.K = Parameter((self.output_channels, *self.filter_size), graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1, 1), init_zeros=True, graph=self.graph)
        self.parameters()

    def __str__(self):
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
        self.K = Parameter((self.input_channels, *self.filter_size), graph=self.graph)
        self.b = Parameter((self.output_channels, 1, 1, 1), init_zeros=True, graph=self.graph)
        self.parameters()

    def __str__(self):
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
        self.W_ih = Parameter((4*self.hidden_size, self.input_size), graph=self.graph)    # input to hidden weight volume
        self.W_hh = Parameter((4*self.hidden_size, self.hidden_size), graph=self.graph)   # hidden to hidden weight volume
        self.b_ih = Parameter((4*self.hidden_size, 1), graph=self.graph)  # input to hidden bias vector
        self.b_hh = Parameter((4*self.hidden_size, 1), graph=self.graph)  # hidden to hidden bias vector
        self.parameters()

    def __str__(self):
        return('LSTM(input_size={}, hidden_size={}, bias={})'.format(
            self.input_size, self.hidden_size, self.bias))

    def __call__(self, x, hidden):  # easy callable
        return self.forward(x, hidden)

    def forward(self, x, hidden):

        h, c = hidden

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)


        h_h = self.graph.dot(self.W_hh, h)
        if self.bias:
            h_h = self.graph.add(h_h, self.b_hh, axis=(-1,))

        i_h = self.graph.dot(self.W_ih, x)
        if self.bias:
            i_h = self.graph.add(i_h, self.b_ih, axis=(-1,))

        gates = self.graph.add(h_h, i_h)

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
        self.W_ih = Parameter((self.hidden_size, self.input_size), graph=self.graph)
        self.W_hh = Parameter((self.hidden_size, self.hidden_size), graph=self.graph)
        self.b_ih = Parameter((self.hidden_size, 1), graph=self.graph)    # not much use
        self.b_hh = Parameter((self.hidden_size, 1), graph=self.graph)
        self.parameters()

    def __str__(self):
        return('RNN(input_size={}, hidden_size={}, bias={})'.format(
            self.input_size, self.hidden_size, self.bias))

    def __call__(self, x, hidden):  # easy callable
        return self.forward(x, hidden)

    def forward(self, x, hidden):

        h = hidden

        if not isinstance(x, Parameter):
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        h_h = self.graph.dot(self.W_hh, h)
        if self.bias:
            h_h = self.graph.add(h_h, self.b_hh, axis=(-1,))

        i_h = self.graph.dot(self.W_ih, x)
        if self.bias:
            i_h = self.graph.add(i_h, self.b_ih, axis=(-1,))

        h = self.graph.add(h_h, i_h)

        h = self.graph.tanh(h)

        return h


# bacth normalization layer
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
        self.parameters()
        self.m = np.sum(np.zeros(shape), axis=self.axis, keepdims=True) / shape[self.axis]    # moving mean
        self.v = np.sum(np.ones(shape), axis=self.axis, keepdims=True) / shape[self.axis]     # moving variance

    def __str__(self):
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
            return self.MSELoss(y_out, y_true)
        elif self.loss_fn == 'CrossEntropyLoss':
            return self.CrossEntropyLoss(y_out, y_true)
        elif self.loss_fn == 'BCELoss':
            return self.BCELoss(y_out, y_true)
        elif self.loss_fn == 'JSDivLoss':
            return self.JSDivLoss(y_out, y_true)
        elif self.loss_fn == 'TestLoss':
            return self.TestLoss(y_out)
        else:
          raise 'No such loss function'

    def __str__(self):
        return('Loss(loss_fn={})'.format(self.loss_fn))

    def MSELoss(self, y_out, y_true):

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

    def CrossEntropyLoss(self, y_out, y_true):

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

    def BCELoss(self, y_out, y_true):

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

    def JSDivLoss(self, y_out, y_true):

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

    def TestLoss(self, y_out):

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
    def __init__(self, parameters, optim_fn='SGD', lr=3e-4, momentum=0.0, eps=1e-8, beta1=0.9, beta2=0.999, ro=0.95, graph=G):
        self.parameters = parameters  # a list of all layers of the model
        self.optim_fn = optim_fn    # the optimizing function(SGD, Adam, Adagrad, RMSProp)
        self.lr = lr    # alpha: size of the step to update the parameters
        self.momentum = momentum
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.ro = ro
        self.graph = graph
        self.t = 0  # iteration count
        self.m = [] # (momentumAdam/Adagrad/Adadelta)
        self.v = [] # (Adam/Adadelta)

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

    def __str__(self):
        return('Optimizer(optim_fn={}, lr={}, momentum={})'.format(
            self.optim_fn, self.lr, self.momentum))

    # a very important step in learning time
    def zero_grad(self):
        # clearing out the backprop operations from the list
        self.graph.nodes = []
        self.graph.node_count = 0

        # resetting the gradients of model parameters to zero
        for parameter in self.parameters:
            parameter.grad = np.zeros(parameter.shape)

    def step(self):

        if self.optim_fn == 'SGD':
            return self.SGD()
        elif self.optim_fn == 'Adam':
            return self.Adam()
        elif self.optim_fn == 'Adagrad':
            return self.Adagrad()
        elif self.optim_fn == 'Adadelta':
            self.eps = 1e-6
            return self.Adadelta()
        else:
          raise 'No such optimization function'

    # Stochastic Gradient Descent optimization function
    def SGD(self):
        if self.t < 1: print('using SGD')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

            if self.momentum > 0.0:
                # momentum update
                delta = self.momentum * self.m[p] - self.lr * self.parameters[p].grad

                # store delta for next iteration
                self.m[p] = delta

                # Update parameters with momentum SGD
                self.parameters[p].data += delta

            else:
                # Update parameters with vanilla SGD
                self.parameters[p].data -= self.lr * self.parameters[p].grad


    # Adam optimization function
    def Adam(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf
        if self.t < 1: print('using Adam')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

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
    def Adagrad(self):
        if self.t < 1: print('using Adagrad')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

            # update memory
            self.m[p] += self.parameters[p].grad * self.parameters[p].grad

            # Update parameters
            self.parameters[p].data -= self.lr * self.parameters[p].grad / np.sqrt(self.m[p] + self.eps)

    # Adadelta optimization function
    def Adadelta(self):
        # useful: https://arxiv.org/pdf/1212.5701.pdf
        if self.t < 1: print('using Adadelta')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

            # Accumulate Gradient:
            self.m[p] = self.ro * self.m[p] + (1 - self.ro) * self.parameters[p].grad * self.parameters[p].grad

            # Compute Update:
            delta = -np.sqrt((self.v[p] + self.eps) / (self.m[p] + self.eps)) * self.parameters[p].grad

            # Accumulate Updates:
            self.v[p] = self.ro * self.v[p] + (1 - self.ro) * delta * delta

            # Apply Update:
            self.parameters[p].data += delta

    #define optimizers


# initializations and utitlity functions
def manual_seed(seed=2357):
    np.random.seed(seed)


# TODO: define regularizations, asserts, batch, utils, GPU support, examples
