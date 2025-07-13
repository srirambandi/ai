import numpy as np
import ai.parameter

from typing import Callable, List
from dataclasses import dataclass


# Computational Graph wannabe: stores the backward operation for every
# forward operation during forward-propagation, in a breadth-fist manner
class ComputationalGraph:
    def __init__(self, grad_mode=True):
        self.grad_mode = grad_mode
        self.nodes = list()

    # functions required for deep learning models and their respective backward operations
    def dot(self, x, y):    # dot op is alias for matmul, keeping it here to support old code
        return self.matmul(x, y)

    def matmul(self, x, y):    # matrix multiplication!
        # logic to handle 1D tensors being passed here
        x_data = x.data
        y_data = y.data

        x_is_1d = (x.ndim == 1)
        y_is_1d = (y.ndim == 1)
        if x_is_1d:
            x_data = x.data.reshape(1, -1)
        if y_is_1d:
            y_data = y.data.reshape(-1, 1)

        out = np.matmul(x_data, y_data)

        if x_is_1d and y_is_1d:
            out = out.squeeze()     # (k), (k) -> ()
        elif x_is_1d:
            out = out.squeeze(0)    # (k) (n, k) -> (n)
        elif y_is_1d:
            out = out.squeeze(-1)   # (n, k), (k) -> (n)

        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # useful: http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf
                grad = out.grad

                if x_is_1d and y_is_1d:
                    if x.requires_grad:
                        x.grad += grad * y.data
                    if y.requires_grad:
                        y.grad += grad * x.data
                    return

                if grad.ndim == 1:
                    if x_is_1d:
                        grad = grad.reshape(1, -1)  # (n) -> (1, n)
                    elif y_is_1d:
                        grad = grad.reshape(-1, 1)  # (n) -> (n, 1)
                elif grad.ndim == 0:
                    grad = np.array([[grad]])   # () -> (1, 1)
                
                if x.requires_grad:
                    x.grad += np.matmul(grad, np.swapaxes(y_data, -1, -2)).reshape(x.shape)
                if y.requires_grad:
                    y.grad += np.matmul(np.swapaxes(x_data, -1, -2), grad).reshape(y.shape)
                    

            node = {'func': '@', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def add(self, x, y):    # element wise addition
        # make x and y broadcastable
        x_ndim_orig, y_ndim_orig = x.ndim, y.ndim
        x_data, x_1d_axes, y_data, y_1d_axes = self.__make_broadcastable(x, y)
        out = ai.parameter.Parameter(data=np.add(x_data, y_data), graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x_grad = np.sum(out.grad, axis=x_1d_axes, keepdims=True)
                    if x_ndim_orig < y_ndim_orig:
                        x_grad = np.reshape(x_grad, x.shape)
                    x.grad += x_grad
                if y.requires_grad:
                    y_grad = np.sum(out.grad, axis=y_1d_axes, keepdims=True)
                    if y_ndim_orig < x_ndim_orig:
                        y_grad = np.reshape(y_grad, y.shape)
                    y.grad += y_grad

            node = {'func': '+', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def subtract(self, x, y):   # element wise subtraction
        # make x and y broadcastable
        x_ndim_orig, y_ndim_orig = x.ndim, y.ndim
        x_data, x_1d_axes, y_data, y_1d_axes = self.__make_broadcastable(x, y)
        out = ai.parameter.Parameter(data=np.subtract(x_data, y_data), graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x_grad = np.sum(out.grad, axis=x_1d_axes, keepdims=True)
                    if x_ndim_orig < y_ndim_orig:
                        x_grad = np.reshape(x_grad, x.shape)
                    x.grad += x_grad
                if y.requires_grad:
                    y_grad = np.sum(out.grad, axis=y_1d_axes, keepdims=True)
                    if y_ndim_orig < x_ndim_orig:
                        y_grad = np.reshape(y_grad, y.shape)
                    y.grad -= y_grad

            node = {'func': '-', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def multiply(self, x, y):   # element wise vector multiplication
        # make x and y broadcastable
        x_ndim_orig, y_ndim_orig = x.ndim, y.ndim
        x_data, x_1d_axes, y_data, y_1d_axes = self.__make_broadcastable(x, y)
        out = ai.parameter.Parameter(data=np.multiply(x_data, y_data), graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x_grad = np.sum(np.multiply(out.grad, y_data), axis=x_1d_axes, keepdims=True)
                    if x_ndim_orig < y_ndim_orig:
                        x_grad = np.reshape(x_grad, x.shape)
                    x.grad += x_grad
                if y.requires_grad:
                    y_grad = np.sum(np.multiply(out.grad, x_data), axis=y_1d_axes, keepdims=True)
                    if y_ndim_orig < x_ndim_orig:
                        y_grad = np.reshape(y_grad, y.shape)
                    y.grad += y_grad

            node = {'func': '*', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def divide(self, x, y, eps=1e-8):   # element wise vector division
        # make x and y broadcastable
        x_ndim_orig, y_ndim_orig = x.ndim, y.ndim
        x_data, x_1d_axes, y_data, y_1d_axes = self.__make_broadcastable(x, y)
        out = ai.parameter.Parameter(data= np.divide(x_data, y_data + eps), graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x_grad = np.sum(np.divide(out.grad, y_data + eps), axis=x_1d_axes, keepdims=True)
                    if x_ndim_orig < y_ndim_orig:
                        x_grad = np.reshape(x_grad, x.shape)
                    x.grad += x_grad
                if y.requires_grad:
                    y_grad = np.sum(-np.multiply(out.grad, x_data / np.square(y_data + eps)), axis=y_1d_axes, keepdims=True)
                    if y_ndim_orig < x_ndim_orig:
                        y_grad = np.reshape(y_grad, y.shape)
                    y.grad += y_grad

            node = {'func': '/', 'inputs': [x, y], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def sum(self, h, axis=None):   # sum of all elements in the matrix
        if axis == None:
            res = np.sum(h.data).reshape(1, 1)  # just a choice to represet default shape as (1, 1). Should I do it like this?
        else:
            res = np.sum(h.data, axis=axis, keepdims=True)
        out = ai.parameter.Parameter(data=res, graph=self)

        if self.grad_mode:
            def backward():
                if h.requires_grad:
                    h.grad += out.grad

            node = {'func': 'sum', 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def power(self, h, exp):   # element wise power
        out = ai.parameter.Parameter(h.shape, init_zeros=True, graph=self)
        out.data = np.power(h.data, exp) if exp >= 0 else np.power(h.data + 1e-8, exp)     # numerical stability for -ve power

        if self.grad_mode:
            def backward():
                if h.requires_grad:
                    if exp  >= 0:
                        h.grad += np.multiply(out.grad, exp * np.power(h.data, exp - 1))
                    else:
                        h.grad += np.multiply(out.grad, exp * np.power(h.data + 1e-8, exp - 1))

            node = {'func': '^{}'.format(exp), 'inputs': [h], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def log(self, h):   # element wise logarithm
        out = ai.parameter.Parameter(data=np.log(h.data + 1e-8), graph=self)     # numerical stability for values ~0

        if self.grad_mode:
            def backward():
                if h.requires_grad:
                    h.grad += np.multiply(out.grad, np.divide(1.0, h.data + 1e-8))

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

        N = x.shape[0]      # Batch size
        C = x.shape[1]      # number of input channels
        i = x.shape[2:]     # input channel shape
        F = K.shape[0]      # number of output filters
        k = K.shape[2:]     # kernel filter shape

        # Figure out output dimensions
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        # padding the input
        pad_x = np.pad(x.data, ((0, 0), (0, 0), (p[0], p[0])), mode='constant')

        # get strided view of padded input by picking appropriate strides
        shape = (N, C, *o, *k)
        strides = pad_x.strides[:2] + (pad_x.strides[2]*s[0],) + pad_x.strides[2:]
        strided_x = np.lib.stride_tricks.as_strided(pad_x, shape=shape, strides=strides)
        out = np.tensordot(strided_x, K.data, axes=([1, 3], [1, 2]))
        out = np.transpose(out, (0, 2, 1))

        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                if K.requires_grad:
                    # (N, C, o, k) x (N, F, o) -> (C, k, F)
                    grad_k = np.tensordot(strided_x, out.grad, axes=([0, 2], [0, 2]))
                    # (C, k, F) -> (F, C, k)
                    K.grad += np.transpose(grad_k, (2, 0, 1))

                if x.requires_grad:

                    pad_x_grad = np.zeros(pad_x.shape)
                    for r in range(out.shape[2]):

                        # solving gradient for input feature map that caused the elements in r position of every output filter
                        # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimention using sum

                        _ = out.grad[:, :, r].reshape(N, F, 1, 1)
                        pad_x_grad[:, :, r*s[0]:r*s[0] + k[0]] += np.sum(np.multiply(_, K.data.reshape(1, *K.shape)), axis=1)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, :, p[0]:pad_x_grad.shape[2]-p[0]]

            node = {'func': 'conv1d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def conv2d(self, x, K, s=1, p=0):
        # faster 2d convolution operation
        # look for older, slower but clearer implementation here: https://github.com/srirambandi/ai/blob/3d85bd1cee1eff40a6e86bfea20b63bcd165f07b/ai/graph.py#L244-L319

        if not isinstance(s, tuple):  
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)

        N = x.shape[0]      # Batch size
        C = x.shape[1]      # number of input channels
        i = x.shape[2:]     # input channel shape
        F = K.shape[0]      # number of output filters
        k = K.shape[2:]     # kernel filter shape

        # Figure out output dimensions
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        # padding the input
        pad_x = np.pad(x.data, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')

        # get strided view of padded input by picking appropriate strides
        shape = (N, C, *o, *k)
        strides = pad_x.strides[:2] + (pad_x.strides[2]*s[0], pad_x.strides[3]*s[1]) + pad_x.strides[2:]
        strided_x = np.lib.stride_tricks.as_strided(pad_x, shape=shape, strides=strides)
        out = np.tensordot(strided_x, K.data, axes=([1, 4, 5], [1, 2, 3]))
        out = np.transpose(out, (0, 3, 1, 2))

        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                if K.requires_grad:
                    # (N, C, o, o, k, k) x (N, F, o, o) -> (C, k, k, F)
                    grad_k = np.tensordot(strided_x, out.grad, axes=([0, 2, 3], [0, 2, 3]))
                    # (C, k, k, F) -> (F, C, k, k)
                    K.grad += np.transpose(grad_k, (3, 0, 1, 2))

                if x.requires_grad:

                    pad_x_grad = np.zeros(pad_x.shape)
                    for r in range(out.shape[2]):
                        for c in range(out.shape[3]):

                            # solving gradient for input feature map that caused the elements in r, c position of every output filter
                            # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimension using sum

                            patch = out.grad[:, :, r, c].reshape(N, F, 1, 1, 1)
                            pad_x_grad[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += np.sum(np.multiply(patch, K.data.reshape(1, *K.shape)), axis=1)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, :, p[0]:pad_x_grad.shape[2]-p[0], p[1]:pad_x_grad.shape[3]-p[1]]

            node = {'func': 'conv2d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def conv_transpose2d(self, x, K, s=(1, 1), p=(0, 0), a=(0, 0)):
        # faster 2d convolution operation
        # look for older, slower but clearer implementation here: https://github.com/srirambandi/ai/blob/3d85bd1cee1eff40a6e86bfea20b63bcd165f07b/ai/graph.py#L384-L453

        if not isinstance(s, tuple):
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)
        if not isinstance(a, tuple):
            a = (a, a)

        N = x.shape[0]      # Batch size
        F = x.shape[1]      # number of input filters
        i = x.shape[2:]     # input filter shape
        C = K.shape[1]      # number of output channels
        k = K.shape[2:]     # kernel filter shape

        o = tuple((map(lambda i, k, s, p, a: int((i - 1)*s + a + k - 2*p), i, k, s, p, a)))
        pad_o = tuple(map(lambda o, p: o + 2*p, o, p))

        pad_out = np.zeros((N, C, *pad_o))

        for r in range(x.shape[2]):
            for c in range(x.shape[3]):

                # computing output image feature map by convolving across each element of input feature map with kernel
                patch = x.data[:, :, r, c].reshape(N, F, 1, 1, 1)
                pad_out[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += np.sum(np.multiply(patch, K.data.reshape(1, *K.shape)), axis=1)

        # cutting the padded portion from the input-feature-map's gradient
        # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
        out = pad_out[:, :, p[0]:pad_out.shape[2]-p[0], p[1]:pad_out.shape[3]-p[1]]

        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                # padding the output gradient
                pad_out_grad = np.pad(out.grad, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')

                # get strided view of padded output gradient by picking appropriate strides
                shape = (N, C, *i, *k)
                strides = pad_out_grad.strides[:2] + (pad_out_grad.strides[2]*s[0], pad_out_grad.strides[3]*s[1]) + pad_out_grad.strides[2:]
                strided_out_grad = np.lib.stride_tricks.as_strided(pad_out_grad, shape=shape, strides=strides)
                # out_grad_cols = np.ascontiguousarray(strided_out_grad)
                # out_grad_cols = out_grad_cols.reshape(C * k[0] * k[1], i[0] * i[1] * N)

                if K.requires_grad:
                    # (N, C, i, i, k, k) x (N, F, i, i) -> (C, k, k, F)
                    grad_k = np.tensordot(strided_out_grad, x.data, axes=([0, 2, 3], [0, 2, 3]))
                    # (C, k, k, F) -> (F, C, k, k)
                    K.grad += np.transpose(grad_k, (3, 0, 1, 2))

                if x.requires_grad:
                    # (N, C, i, i, k, k) x (F, C, k, k) -> (N, i, i, F)
                    grad_x = np.tensordot(strided_out_grad, K.data, axes=([1, 4, 5], [1, 2, 3]))
                    # (N, i, i, F) -> (N, F, i, i)
                    x.grad += np.transpose(grad_x, (0, 3, 1, 2))

            node = {'func': 'conv_transpose2d', 'inputs': [x, K], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def max_pool2d(self, x, k, s=None, p=0):    # maxpool layer(no params)
        # useful: https://arxiv.org/pdf/1603.07285.pdf
        # look for older, slower but clearer implementation here: https://github.com/srirambandi/ai/blob/3d85bd1cee1eff40a6e86bfea20b63bcd165f07b/ai/graph.py#L523-L587

        if s is None:
            s = k
        if not isinstance(k, tuple):
            k = (k, k)
        if not isinstance(s, tuple):  
            s = (s, s)
        if not isinstance(p, tuple):
            p = (p, p)

        N = x.shape[0]      # Batch size
        F = x.shape[1]      # number of input filter planes
        i = x.shape[2:]     # input shape of any channel of the input feature map before padding

        # Figure out output dimensions
        o = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))
        pad_i = tuple(map(lambda i, p: i + 2*p, i, p))

        # padding the input
        pad_x = np.pad(x.data, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')

        # get strided view of padded input by picking appropriate strides
        shape = (N, F, *o, *k)
        strides = pad_x.strides[:2] + (pad_x.strides[2]*s[0], pad_x.strides[3]*s[1]) + pad_x.strides[2:]
        strided_x = np.lib.stride_tricks.as_strided(pad_x, shape=shape, strides=strides)
        # fatten the kernel window to a single column, so that we can apply max operation along the last axis
        strided_x_col = strided_x.reshape(N, F, *o, k[0] * k[1])

        out = np.max(strided_x_col, axis=-1)
        max_mask = (strided_x_col - out[..., np.newaxis]).reshape(shape)
        max_mask = np.where(max_mask == 0, 1.0, 0)

        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:

                    pad_x_grad = np.zeros(pad_x.shape)

                    for r in range(out.shape[2]):
                        for c in range(out.shape[3]):

                            # multiplying each 'mask' like volume(single 1s in the volumes along all batches) with the gradient
                            # at region whose value was caused by the mask region's input
                            pad_x_grad[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += max_mask[:, :, r, c] * out.grad[:, :, r, c].reshape(N, F, 1, 1)

                    # cutting the padded portion from the input gradient
                    # and updating the gradient of actual input(non-padded) - unpadding and updating
                    x.grad += pad_x_grad[:, :, p[0]:pad_x_grad.shape[2]-p[0], p[1]:pad_x_grad.shape[3]-p[1]]

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
        out = ai.parameter.Parameter(data=dropout_mask*x.data, graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x.grad += out.grad*dropout_mask # only activated units get gradients

            node = {'func': 'dropout', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # hidden and output units activations
    def relu(self, z):      # element wise ReLU activations
        out = ai.parameter.Parameter(data=np.maximum(z.data, 0), graph=self)

        if self.grad_mode:
            def backward():
                if z.requires_grad:
                    z.grad += out.grad.copy()
                    z.grad[z.data < 0] = 0

            node = {'func': 'relu', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def lrelu(self, z, alpha=1e-2):      # element wise Leaky ReLU activations
        out = ai.parameter.Parameter(data=np.maximum(z.data, alpha * z.data), graph=self)

        if self.grad_mode:
            def backward():
                if z.requires_grad:
                    z.grad += out.grad.copy()
                    z.grad[z.data < 0] *= alpha

            node = {'func': 'lrelu', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def gelu(self, z):      # element wise GELU activations
        # ref: https://arxiv.org/pdf/1606.08415
        #let's do approximate gelu
        x = z.data
        x_square = x * x
        x_cube = x_square * x
        root_2_by_pi = np.sqrt(2/np.pi)
        tanh_term = np.tanh(root_2_by_pi * (x + (0.044715 * x_cube)))
        out = 0.5 * x * (1 + tanh_term)
        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                if z.requires_grad:
                    z.grad += 0.5 * (1 + tanh_term + x * (1 - (tanh_term * tanh_term)) * root_2_by_pi * (1. + 3 * 0.044715 * x_square))

            node = {'func': 'gelu', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def sigmoid(self, z):   # element wise sigmoid activations
        shape = z.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = 1.0/(1.0 + np.exp(-1.0*z.data))

        if self.grad_mode:
            def backward():
                if z.requires_grad:
                    z.grad += np.multiply(np.multiply(out.data, 1.0 - out.data), out.grad)

            node = {'func': 'sigmoid', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def softmax(self, z, axis=0):   # element wise softmax activations
        shape = z.shape
        if axis < 0:
            axis += z.ndim
        assert axis in [1, 2] and axis < len(z.shape), 'Invalid axis for softmax'
        assert len(shape) in [2, 3], 'Invalid shape for softmax'
        is_1d = len(shape) == 2 # if 1D, then axis=1, 0th axis is batch size
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        
        # Subtracting the max for numerical stability
        e_z = np.exp(z.data - np.max(z.data, axis=axis, keepdims=True))

        # Sum along the specified axis
        sum_e_z = np.sum(e_z, axis=axis, keepdims=True)
        sum_e_z[sum_e_z == 0] = 1e-8 # for safe and stable division
        
        # Softmax calculation
        out.data = e_z / sum_e_z

        if self.grad_mode:
            def backward():
                if z.requires_grad:
                    # >>> Old Implementation, which assumes that the gradient of the loss wrt the softmax output is 1
                    # >>> and doesn't handle softmx of multidimensional arrays
                    # # directly coding the end result instead of formula - easy this way
                    # z.grad += out.data - np.where(out.grad == 0, 0, 1.0)

                    # >>> New Implementation, which implements for a general case where the gradient of the loss wrt the softmax output
                    # >>> is not necessarily 1, and handles softmx of multidimensional arrays
                    if is_1d:
                        # making 1D softmax gradient calculation consistent with the 2D implementation
                        # by reshaping the output and gradient tensors to 2D + batch size, and then reshaping the gradient back
                        out_data = np.expand_dims(out.data, axis=len(shape))   # adding new dim at the end
                        out_grad = np.expand_dims(out.grad, axis=len(shape))  # adding new dim at the end
                    else:
                        out_data = out.data
                        out_grad = out.grad
                    out_i = np.expand_dims(out_data, axis=axis + 1)
                    out_j = np.expand_dims(out_data, axis=axis)

                    jacobian = -out_i * out_j  # For i != j
                    ii_indices = np.arange(out.data.shape[axis])
                    # Adding the diagonal part of the jacobian
                    if axis == 1:
                        jacobian[:, ii_indices, ii_indices, :] = out_data * (1 - out_data)
                    elif axis == 2:
                        jacobian[:, :, ii_indices, ii_indices] = out_data * (1 - out_data)

                    # Now, apply this jacobian to grad_out
                    grad_out_expanded = np.expand_dims(out_grad, axis=axis + 1)  # Expanding dims for correct broadcasting
                    jacobian_prod = jacobian * grad_out_expanded
                    z_grad = np.sum(jacobian_prod, axis=axis)  # Sum over the softmax dimension

                    if is_1d:
                        # the last axis is the one we added, it is of size 1, so we remove it
                        z.grad += z_grad.squeeze(axis=len(shape))
                    else:
                        # case where the input is 2D input
                        z.grad += z_grad

            node = {'func': 'softmax', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def tanh(self, z):      # element wise tanh activations
        out = ai.parameter.Parameter(data=np.tanh(z.data), graph=self)

        if self.grad_mode:
            def backward():
                if z.requires_grad:
                    z.grad += np.multiply(1 - np.multiply(out.data, out.data), out.grad)

            node = {'func': 'tanh', 'inputs': [z], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # data manipulation/view functions
    def split(self, W, sections=1, axis=-1):
        if axis < 0:
            axis += W.ndim
        outs = np.split(W.data, sections, axis=axis)
        outs_list = list()
        for e in outs:
            o = ai.parameter.Parameter(data=e, graph=self)
            outs_list.append(o)

        if self.grad_mode:
            def backward():
                outs_grads = [o.grad for o in outs_list]
                if W.requires_grad:
                    W.grad += np.concatenate(outs_grads, axis=axis)

            node = {'func': 'split', 'inputs': [W], 'outputs': outs_list, 'backprop_op': lambda: backward()}
            for out in outs_list:
                out.node_id = len(self.nodes)
            self.nodes.append(node)

        return outs_list
    
    def getitem(self, x, key):
        out = ai.parameter.Parameter(data=x.data[key], graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x.grad[key] += out.grad

            node = {'func': '[,]', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def cat(self, inputs_list, axis=-1):
        if axis < 0:
            axis += inputs_list[0].ndim
        split_points = np.cumsum([i.shape[axis] for i in inputs_list[:-1]])
        out = ai.parameter.Parameter(data=np.concatenate(inputs_list, axis=axis), graph=self)

        if self.grad_mode:
            def backward():
                input_grads = np.split(out.grad, split_points, axis=axis)
                for e in range(len(inputs_list)):
                    if inputs_list[e].requires_grad:
                        inputs_list[e].grad += input_grads[e]

            node = {'func': 'cat', 'inputs': [inputs_list], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def transpose(self, x, axis0=None, axis1=None):     # transpose
        if x.ndim == 1:
                raise ValueError('no transpose operation supported for 1D tensors')
        if axis0 is None or axis1 is None:
            if x.ndim == 2:
                axis0, axis1 = 0, 1
            else:
                raise ValueError('axis0 and axis1 must be specified for transpose operation on tensors with more than 2 dimensions')

        axes = list(range(len(x.shape)))
        axes[axis0] = axis1
        axes[axis1] = axis0
        out = np.ascontiguousarray(np.transpose(x.data, axes=axes))
        out = ai.parameter.Parameter(data=out, graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x.grad += np.transpose(out.grad, axes=axes)

            node = {'func': 'x.T', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    def reshape(self, x, shape):
        old_shape = x.shape
        out = np.ascontiguousarray(x.data).reshape(shape)
        out = ai.parameter.Parameter(data=out, init_zeros=True, graph=self)

        if self.grad_mode:
            def backward():
                if x.requires_grad:
                    x.grad += np.ascontiguousarray(out.grad).reshape(old_shape)

            node = {'func': 'reshape', 'inputs': [x], 'outputs': [out], 'backprop_op': lambda: backward()}
            out.node_id = len(self.nodes)
            self.nodes.append(node)

        return out

    # utility functions for computational graph ops
    def __make_broadcastable(self, x, y):
        x_1d_axes, y_1d_axes = [], []
        x_ndim_orig, y_ndim_orig = x.ndim, y.ndim
        x_shape, y_shape = x.shape, y.shape
        x_data = x.data
        y_data = y.data

        if x_ndim_orig < y_ndim_orig:
            x_data = x_data.reshape(*[1 for _ in range(len(y_ndim_orig - x_ndim_orig))], *x_shape)
        elif y_ndim_orig < x_ndim_orig:
            y_data = y_data.reshape(*[1 for _ in range(len(x_ndim_orig - y_ndim_orig))], *y_shape)

        # assert if broadcastable still
        broadcastable = True
        for ax, ay in zip(x_data.shape, y_data.shape):
            if ax == ay or ax == 1 or ay == 1:
                continue
            else:
                broadcastable = False
                break
        assert broadcastable, f"arrays of shapes {x_shape} and {y_shape} can't be broadcasted."

        for axis in range(len(x_data.shape)):
            if x_data.shape[axis] == 1 and y_data.shape[axis] == 1:
                continue
            if x_data.shape[axis] == 1:
                x_1d_axes.append(axis)
            elif y_data.shape[axis] == 1:
                y_1d_axes.append(axis)
        x_1d_axes, y_1d_axes = tuple(x_1d_axes), tuple(y_1d_axes)
        
        return x_data, x_1d_axes, y_data, y_1d_axes


G = ComputationalGraph()
