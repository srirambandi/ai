"""
AI library in python using numpy
Written by Sri Ram Bandi (@_srirambandi_)

BSD License
"""

import numpy as np


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), data=None, eval_grad=True, init_zeros=False, init_ones=False, uniform=False, mu = 0.0, std = 0.01):
        self.shape = shape
        self.data = data
        self.eval_grad = eval_grad  # if the parameter is a variable or an input/constant
        self.init_zeros = init_zeros
        self.init_ones = init_ones
        self.uniform = uniform
        self.mu = mu    # mean and variance of the gaussian
        self.std = std  # distribution to initialize the parameter
        self.init_params()

    def init_params(self):

        if self.data != None:
            # initiating weights with passed data object of kind list/numpy-ndarray
            self.w = np.array(data)
            self.data = None    # memory conservation
            self.shape = self.w.shape   # resolving conflict with passed shape and data shape

        elif self.init_zeros:
            # initiating with zeros of given shape
            self.w = np.zeros(self.shape)

        elif self.init_ones:
            # initiating with ones of given shape
            self.w = np.ones(self.shape)

        elif self.uniform:
            # random initiation with uniform distribution
            self.w = 2*np.random.rand(*self.shape) - 1.0

        else:
            # random initiation with gaussian distribution
            self.w = self.std*np.random.randn(*self.shape) + self.mu

        # setting gradient of parameter wrt some scalar, as zeros
        self.dw = np.zeros(self.shape)

    # transpose
    def T(self):

        self.w = self.w.T
        self.dw = self.dw.T
        self.shape = tuple(reversed(self.shape))

        return self


# Computational Graph wannabe: stores the backward operation for every
# forward operation during forward-propagarion and later computes them, in a breadth-fist manner
class ComputationalGraph:
    def __init__(self, grad_mode=True):
        self.grad_mode = grad_mode
        self.backprop = []

    # this function when called computes the gradients of the model parameters
    # by executing the backprop operations in reverse order to the forward propagarion;
    # the gradients are computed with chain rule
    def backward(self):
        for backprop_op in reversed(self.backprop):
            backprop_op()


    # operations required for deep learning models and their backward operations
    def dot(self, W, x):    # dot product of vectors and matrices
        shape = (W.w.shape[0], x.w.shape[1])
        out = Parameter(shape, init_zeros=True)
        out.w = np.dot(W.w, x.w)

        if self.grad_mode:
            def backward():
                # useful: http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf
                # print('dot')
                if W.eval_grad:
                    W.dw += np.dot(out.dw, x.w.T)
                if x.eval_grad:
                    x.dw += np.dot(out.dw.T, W.w).T

                # return (x.dw, W.dw)

            self.backprop.append(lambda: backward())

        return out

    def add(self, x, y, axis=()):    # element wise addition
        # bias should be passed in position of y
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.add(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('add')
                if x.eval_grad:
                    x.dw += out.dw
                if y.eval_grad:
                    y.dw += np.sum(out.dw, axis = axis).reshape(y.shape)   # bias adding to batched vector

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def subtract(self, x, y, axis=()):   # element wise subtraction
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.subtract(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('subtract')
                if x.eval_grad:
                    x.dw += out.dw
                if y.eval_grad:
                    y.dw -= np.sum(out.dw, axis=axis).reshape(y.shape)  # for the second parameter

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def multiply(self, x, y, axis=()):   # element wise vector multiplication
        # not for scalar multiply
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.multiply(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('multiply')
                if x.eval_grad:
                    x.dw += np.multiply(out.dw, y.w)
                if y.eval_grad:
                    y.dw += np.sum(np.multiply(out.dw, x.w), axis=axis).reshape(y.shape)

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def divide(self, x, y, axis=()):   # element wise vector division
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.divide(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('divide')
                if x.eval_grad:
                    x.dw += np.multiply(out.dw, np.divide(1.0, y.w))
                if y.eval_grad:
                    y.dw += np.sum(np.multiply(out.dw, np.multiply(out.w, np.divide(-1.0, y.w))), axis=axis).reshape(y.shape)

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def sum(self, h, axis=None):   # sum of all elements in the matrix
        if axis == None:
            res = np.sum(h.w).reshape(1, 1)
        else:
            res = np.sum(h.w, axis=axis, keepdims=True)
        out = Parameter(res.shape, init_zeros=True)
        out.w = res

        if self.grad_mode:
            def backward():
                # print('sum')
                if h.eval_grad:
                    h.dw += out.dw

                # return h.dw

            self.backprop.append(lambda: backward())

        return out

    def power(self, h, power):   # power of all elements in the matrix
        out = Parameter(h.shape, init_zeros=True)
        out.w = np.power(h.w, power) if power >= 0 else np.power(h.w, power) + 1e-6     # numerical stability for -ve power

        if self.grad_mode:
            def backward():
                # print('power')
                if h.eval_grad:
                    h.dw += np.multiply(out.dw, power * np.power(h.w, power - 1))

                # return h.dw

            self.backprop.append(lambda: backward())

        return out

    def log(self, h):   # element wise log
        shape = h.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.log(h.w)

        if self.grad_mode:
            def backward():
                # print('log')
                if h.eval_grad:
                    h.dw += np.multiply(out.dw, np.divide(1.0, h.w))

                # return h.dw

            self.backprop.append(lambda: backward())

        return out

    # layers operations
    # UPDATE: see older commits of conv2d and maxpool2d for easier Understanding
    # the portion for conv2d and maxpool2d operation became little complicated to facilitate fast computation.
    # For an easier logic code, see this commit which is very inefficient but easy to understand
    # https://github.com/srirambandi/ai/commit/f886cbd616b3d808acaa7d6c702d2b8b93fe8d9e#diff-0ef108fef71dfdcd1cbaad80982c92ac
    def conv2d(self, x, K, s = (1, 1), p = (0, 0)):     # 2d convolution operation
        # useful: https://arxiv.org/pdf/1603.07285.pdf
        if type(s) is not tuple:    # already handled in Conv2d class definition
            s = (s, s)              # adding for compatibility direct calling without using Conv2d class
        if type(p) is not tuple:
            p = (p, p)

        fi = K.shape[0]     # number of filters
        ch = x.shape[0]     # number of input channels
        k = K.shape[2:]     # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel  of some cth-channel in a kth-filter
        i = x.shape[1:-1]   # input shape of any channel of the input feature map before padding
        batch = x.shape[-1] # batch size of the input
        output_maps_shape = (fi, *(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p)), batch) # output feature maps shape - (# of filters, o_i, o_j)
        pad_shape = (ch, *(map(lambda a, b: a + 2*b, i, p)), batch)   # padded input feature maps shape - (channels, new_i, new_j)

        out = np.zeros(output_maps_shape)  # output feature maps

        pad_x = np.zeros(pad_shape)
        # padded input - copying the actual input onto pad input centre
        pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :] += x.w
        pad_x = pad_x.reshape(1, *pad_shape)

        # convolution operation is commutative because of this flipping
        # this step and consequently flipping in backward op are not essential for network
        flipped_K = np.flip(np.flip(K.w), axis=(0, 1))

        for r in range(out.shape[1]):        # convolving operation here
            for c in range(out.shape[2]):    # traversing rous and columns of feature map

                # multiplying traversed grid portions of padded input feature maps with kernel grids element-wise
                # and summing the resulting matrix to produce elements of output maps, over all filters and batches
                out[:, r, c, :] += np.sum(np.multiply(pad_x[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :], flipped_K.reshape(*K.shape, 1)), axis=(1, 2, 3))

        output_feature_maps = Parameter(out.shape, init_zeros=True)
        output_feature_maps.w = out    # any set of output feature map from the batch, has same numeber of maps as filters in the kernel set

        if self.grad_mode:
            def backward():
                # print('conv2d')
                if K.eval_grad:

                    flippe_K_grad = np.zeros(K.shape)
                    for r in range(output_feature_maps.shape[1]):
                        for c in range(output_feature_maps.shape[2]):

                            # solving gradient for each kernel filter that caused the elements in r, c position of every output filter
                            # in every bacth; sketch and think, with input stacked fi times to make computation fast

                            _ = output_feature_maps.dw[:, r, c, :].reshape(fi, 1, 1, 1, batch)
                            # updating the kernel filter set gradient - there will be RxC such updates
                            flippe_K_grad += np.sum(np.multiply(_, pad_x[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]), axis = -1)

                    K.dw += np.flip(np.flip(flippe_K_grad), axis=(0, 1))

                if x.eval_grad:

                    pad_x_grad = np.zeros(pad_shape)

                    for r in range(output_feature_maps.shape[1]):
                        for c in range(output_feature_maps.shape[2]):

                            # solving gradient for input feature map that caused the elements in r, c position of every output filter
                            # in every batch; similar to kernel gradient method, but the matrix collapses along filters dimention using sum

                            _ = output_feature_maps.dw[:, r, c, :].reshape(fi, 1, 1, 1, batch)
                            pad_x_grad[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] += np.sum(np.multiply(_, flipped_K.reshape(*K.shape, 1)), axis=0)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    x.dw += pad_x_grad[:, p[0]:pad_x_grad.shape[1]-p[0], p[1]:pad_x_grad.shape[2]-p[1], :]

                # return (K.dw, x.dw)

            self.backprop.append(lambda: backward())

        return output_feature_maps

    def maxpool2d(self, x, k=(2, 2), s=(2,2), p=(0, 0)):    # maxpool layer(no params), used generally after Conv2d
        if type(k) is not tuple:
            k = (k, k)
        if type(s) is not tuple:
            s = (s, s)
        if type(p) is not tuple:
            p = (p, p)

        out = []

        fi = x.shape[0]     # number of input filters(panes)
        i = x.shape[1:-1]   # input shape of any channel of the input feature map before padding
        batch = x.shape[-1]
        pool_shape = (fi, *(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p)), batch) # shape after maxpool
        pad_shape = (fi, *(map(lambda a, b: a + 2*b, i, p)), batch)  # padded input shape

        out = np.zeros((pool_shape))

        pad_x = np.zeros(pad_shape)
        pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :] += x.w # copying the input onto padded matrix

        for r in range(out.shape[1]):       # convolving operation here(kinda)
            for c in range(out.shape[2]):   # traversing rous and columns of feature map

                # Selecting max element in the current position where kernel sits on feature map
                # The kernel moves in a convolution manner similar to conv2d
                _ = pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :]
                out[:, r, c, :] = np.max(_, axis=(1, 2))

                if self.grad_mode:      # seems inefficient; will improve this whole maxpool op later

                    # Also storing value 1 at locations in the input that caused the output values(max locations);
                    # this makes life easy during backprop
                    # if multiple 0s occur and max is 0 then it shouldn't count. weeding out such cases by assigning
                    # NaN and later zeroing out their gradient locations too; this was a bug which is fixed now :)
                    out[:, r, c, :][out[:, r, c, :] == 0] = np.nan
                    _ -= out[:, r, c, :].reshape(fi, 1, 1, batch)
                    _[np.isnan(_)] = -1     # removing all zeros locations
                    # can't use '_' for the below assignment, so using the entire notation :(
                    pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] = np.where(pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] < 0, 0, 1.0)
                    out[:, r, c, :][np.isnan(out[:, r, c, :])] = 0

        pool_maps = Parameter(out.shape, init_zeros=True)
        pool_maps.w = out

        if self.grad_mode:
            def backward():
                # print('maxpool2d')
                if x.eval_grad:

                    for r in range(pool_maps.shape[1]):
                        for c in range(pool_maps.shape[2]):

                            # multiplying each 'mask' like volume(single 1 in the volume along all batches) with the gradient
                            # at region whose value was by the mask region's input caused
                            pad_x[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1], :] *= pool_maps.dw[:, r, c, :].reshape(fi, 1, 1, batch)

                    # cutting the padded portion from the input gradient
                    # and updating the gradient of actual input(non-padded) - unpadding and updating
                    x.dw += pad_x[:, p[0]:pad_x.shape[1]-p[0], p[1]:pad_x.shape[2]-p[1], :]

                # return (x.dw)

            self.backprop.append(lambda: backward())

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
        out = Parameter(shape, init_zeros=True)
        out.w = dropout_mask*x.w    # drop/sclae

        if self.grad_mode:
            def backward():
                # print('dropout')
                if x.eval_grad:
                    x.dw += out.dw*dropout_mask # only activated units get gradients

                # return x.dw

            self.backprop.append(lambda: backward())

        return out

    # hidden and output units activations
    def relu(self, z):      # # element wise RELU activations
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.maximum(z.w, 0)

        if self.grad_mode:
            def backward():
                # print('relu')
                if z.eval_grad:
                    z.dw += out.dw.copy()
                    z.dw[z.w < 0] = 0

                # return z.dw

            self.backprop.append(lambda: backward())

        return out

    def lrelu(self, z, alpha=1e-2):      # # element wise RELU activations
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.maximum(z.w, alpha * z.w)

        if self.grad_mode:
            def backward():
                # print('leaky_relu')
                if z.eval_grad:
                    z.dw += out.dw.copy()
                    z.dw[z.w < 0] *= alpha

                # return z.dw

            self.backprop.append(lambda: backward())

        return out

    def sigmoid(self, z):   # element wise sigmoid activations
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = 1.0/(1.0 + np.exp(-1.0*z.w))

        if self.grad_mode:
            def backward():
                # print('sigmoid')
                if z.eval_grad:
                    z.dw += np.multiply(np.multiply(out.w, 1.0 - out.w), out.dw)

                # return z.dw

            self.backprop.append(lambda: backward())

        return out

    def softmax(self, z):   # calculates probs for the unnormalized log probabilities of previous layer
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.exp(z.w - np.max(z.w)) / np.sum(np.exp(z.w - np.max(z.w)), axis=0).reshape(1, -1)

        if self.grad_mode:
            def backward():
                # print('softmax')
                if z.eval_grad:
                    # directly coding the end result instead of formula - easy this way
                    z.dw += out.w - np.where(out.dw == 0, 0, 1.0)

                # return z.dw

            self.backprop.append(lambda: backward())

        return out

    def tanh(self, z):      # element wise tanh activations
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.tanh(z.w)

        if self.grad_mode:
            def backward():
                # print('tanh')
                if z.eval_grad:
                    z.dw += np.multiply(1 - np.multiply(out.w, out.w), out.dw)

                # return z.dw

            self.backprop.append(lambda: backward())

        return out

    # utility functions
    def split(self, W, sections=1, axis=0):
        outs = np.split(W.w, sections, axis=axis)
        outs_param = []
        for e in outs:
            o = Parameter(e.shape, init_zeros=True)
            o.w = e
            outs_param.append(o)

        if self.grad_mode:
            def backward():
                #print('split')
                outs_grads = [o.dw for o in outs_param]
                if W.eval_grad:
                    W.dw += np.concatenate(outs_grads, axis=axis)

            self.backprop.append(lambda: backward())

        return outs_param

    def T(self, x):     # transpose
        shape = tuple(reversed(x.shape))
        out = Parameter(shape, init_zeros=True)
        out.w = x.w.T

        if self.grad_mode:
            def backward():
                # print('T')
                if x.eval_grad:
                    x.dw += out.dw.T

                # return x.dw

            self.backprop.append(lambda: backward())

        return out

    def reshape(self, x, new_shape=None):
        old_shape = x.shape
        batch_size = old_shape[-1]
        if new_shape == None:
            new_shape = x.w.reshape(-1, batch_size).shape
        out = Parameter(new_shape, init_zeros=True)
        out.w = x.w.reshape(new_shape)

        if self.grad_mode:
            def backward():
                # print('reshape')
                if x.eval_grad:
                    x.dw += out.dw.reshape(old_shape)

                # return x.dw

            self.backprop.append(lambda: backward())

        return out


G = ComputationalGraph()



# linear affine transformation: y = Wx + b
# the general feed-forward network
class Linear:
    def __init__(self, hidden_prev = 0, hidden_next = 0, init_zeros=False, graph=G):
        self.hidden_prev = hidden_prev  # previous layer units
        self.hidden_next = hidden_next  # next layer units
        self.init_zeros = init_zeros
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.W = Parameter((self.hidden_next, self.hidden_prev), init_zeros=self.init_zeros)  # weight volume
        self.b = Parameter((self.hidden_next, 1), init_zeros=True)   # bias vector
        self.parameters = [self.W, self.b]  # easy access of the layer params

    def forward(self, x):
        # making the input compatible with graph operations
        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False)

        # flatten the input if it came from layers like Conv2d
        if len(x.shape) > 2:
            x = self.graph.reshape(x)

        # y = Wx + b
        out = self.graph.add(self.graph.dot(self.W, x), self.b, axis=(1,))

        return out


# conv nets
class Conv2d:
    def __init__(self, input_channels=None, output_channels=None, kernel_size=None, stride=(1, 1), padding=(0, 0), graph=G):
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
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.K = Parameter((self.output_channels, *self.filter_size))
        self.b = Parameter((self.output_channels, 1, 1, 1), init_zeros=True)
        self.parameters = [self.K, self.b]

    def forward(self, x):

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False)

        out = self.graph.add(self.graph.conv2d(x, self.K, self.stride, self.padding), self.b, axis=(-3, -2, -1))     # convoulution operation and adding bias

        return out


# sequence models: LSTM cell
class LSTM:
    def __init__(self, input_size, hidden_size, graph=G):
        self.input_size = input_size    # size of the input at each recurrent tick
        self.hidden_size = hidden_size  # size of hidden units h and c
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.W_ih = Parameter((4*self.hidden_size, self.input_size))    # input to hidden weight volume
        self.W_hh = Parameter((4*self.hidden_size, self.hidden_size))   # hidden to hidden weight volume
        self.b_ih = Parameter((4*self.hidden_size, 1))  # input to hidden bias vector
        self.b_hh = Parameter((4*self.hidden_size, 1))  # hidden to hidden bias vector
        self.parameters = [self.W_ih, self.b_ih, self.W_hh, self.b_hh]

    def forward(self, x, hidden):

        h, c = hidden

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False)


        gates = self.graph.add(self.graph.add(self.graph.dot(self.W_hh, h), self.b_hh, axis=(-1,)), self.graph.add(self.graph.dot(self.W_ih, x), self.b_ih, axis=(-1,)))

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
class RNN:
    def __init__(self, input_size, hidden_size, graph=G):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.W_ih = Parameter((self.hidden_size, self.input_size))
        self.W_hh = Parameter((self.hidden_size, self.hidden_size))
        # self.b_ih = Parameter((self.hidden_size, 1), init_zeros=True)    # not much use
        self.b_hh = Parameter((self.hidden_size, 1), init_zeros=True)
        self.parameters = [self.W_ih, self.W_hh, self.b_hh]

    def forward(self, x, hidden):

        h = hidden

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False)

        h = self.graph.add(self.graph.add(self.graph.dot(self.W_hh, h), self.graph.dot(self.W_ih, x)), self.b_hh, axis=(-1,))
        h = self.graph.tanh(h)

        return h


class BatchNorm:
    def __init__(self, hidden_shape, axis=-1, momentum=0.9, graph=G):
        self.hidden_shape = hidden_shape  # gamma and beta size; typically D in (D, N) where N is batch size
        self.axis = axis    # along batch channel axis for conv layers and along batches for linear
        self.momentum = momentum
        self.graph = graph
        self.init_params()

    def init_params(self):
        shape = (*self.hidden_shape, 1)
        self.gamma = Parameter(shape, init_ones=True)
        self.beta = Parameter(shape, init_zeros=True)
        self.parameters = [self.gamma, self.beta]
        self.m = np.sum(np.zeros(shape), axis=self.axis, keepdims=True) / shape[self.axis]    # moving mean
        self.v = np.sum(np.ones(shape), axis=self.axis, keepdims=True) / shape[self.axis]     # moving variance

    def forward(self, x):

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False)

        if self.graph.grad_mode:    # training

            batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False) # mini-batch size/channel size
            batch_size.w.fill(float(x.shape[self.axis]))

            # calculate mean and variance
            mean = self.graph.divide(self.graph.sum(x, axis=self.axis), batch_size)
            center = self.graph.subtract(x, mean, axis=self.axis)
            var = self.graph.divide(self.graph.sum(self.graph.power(center, 2), axis=self.axis), batch_size)

            self.m = self.momentum * self.m + (1 - self.momentum) * mean.w
            self.v = self.momentum * self.v + (1 - self.momentum) * var.w

            # normalize the data to zero mean and unit variance
            normalized = self.graph.multiply(center, self.graph.power(var, -0.5), axis=self.axis)

        else:   # testing

            center = np.subtract(x.w, self.m)
            normalized = np.multiply(center, np.power(self.v + 1e-6, -0.5))
            normalized = Parameter(data=normalized, eval_grad=False)

        # scale and shift
        out = self.graph.add(self.graph.multiply(normalized, self.gamma, axis=(-1,)), self.beta, axis=(-1,))

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
        elif self.loss_fn == 'TestLoss':
            return self.TestLoss(y_out)
        else:
          print('No such loss function')
          import sys
          sys.exit()

    # backprop is called here, computes gradients of the parameters
    # Loss and Computational Graph can call the back propagarion
    def backward(self):
        self.graph.backward()

    def MSELoss(self, y_out, y_true):

        if type(y_true) is not Parameter:
            y_true = Parameter(data=y_true, eval_grad=False)

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False) # mini-batch size
        batch_size.w.fill(float(y_true.shape[-1]))

        # L = (y_out - y_true)^2
        l = self.graph.sum(self.graph.multiply(self.graph.subtract(y_out, y_true), self.graph.subtract(y_out, y_true)))
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = self.graph.divide(l, batch_size)

        l.dw[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def CrossEntropyLoss(self, y_out, y_true):

        if type(y_true) is not Parameter:
            y_true = Parameter(data=y_true, eval_grad=False)

        batch_size = Parameter((1, 1), init_zeros=True, eval_grad=False) # mini-batch size
        batch_size.w.fill(float(y_true.shape[-1]))

        neg_one = Parameter((1, 1), init_zeros=True, eval_grad=False)
        neg_one.w.fill(-1.0)  # just a -1 to make the l.dw look same in all the loss defs (dl/dl = 1)

        # L = -Summation(y_true*log(y_out))
        l = self.graph.multiply(self.graph.sum(self.graph.multiply(y_true, self.graph.log(y_out))), neg_one)
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = self.graph.divide(l, batch_size)

        l.dw[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def TestLoss(self, y_out):

        mbatch_size = Parameter((1, 1), init_zeros=True, eval_grad=False) # mini-batch size
        batch_size.w.fill(float(y_out.shape[-1]))

        # a test loss score function that measures the sum of each output vector as the loss of that sample
        l = self.graph.sum(y_out)
        l = self.graph.divide(l, batch_size)

        l.dw[0, 0] = 1.0

        return l

    #define loss functions


# Optimizers to take that drunken step down the hill
class Optimizer:
    def __init__(self, model, optim_fn='SGD', lr=3e-4, momentum=0.0, eps=1e-8, beta1=0.9, beta2=0.999, ro=0.95, graph=G):
        self.model = model  # a list of all layers of the model
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
            for layer in self.model:
                layer_param = []
                for parameter in layer.parameters:
                    layer_param.append(np.zeros(parameter.shape))
                self.m.append(layer_param)

            if self.optim_fn == 'Adam' or self.optim_fn == 'Adadelta':

                # Adam: 2nd moment(raw variance here) of gradients
                # Adadelta: Accumulates updates
                for layer in self.model:
                    layer_param = []
                    for parameter in layer.parameters:
                        layer_param.append(np.zeros(parameter.shape))
                    self.v.append(layer_param)

    # a very important step in learning time
    def zero_grad(self):
        # clearing out the backprop operations from the list
        self.graph.backprop = []

        # resetting the gradients of model parameters to zero
        for layer in self.model:
            for parameter in layer.parameters:
                parameter.dw = np.zeros(parameter.shape)

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
          print('No such optimization function')
          import sys
          sys.exit()

    # Stochastic Gradient Descent optimization function
    def SGD(self):
        if self.t < 1: print('using SGD')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                if self.momentum > 0.0:
                    # momentum update
                    delta = self.momentum * self.m[l][p] - self.lr * self.model[l].parameters[p].dw

                    # store delta for next iteration
                    self.m[l][p] = delta

                    # Update parameters with momentum SGD
                    self.model[l].parameters[p].w += delta

                else:
                    # Update parameters with vanilla SGD
                    self.model[l].parameters[p].w -= self.lr * self.model[l].parameters[p].dw


    # Adam optimization function
    def Adam(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf
        if self.t < 1: print('using Adam')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # Update biased first moment estimate
                self.m[l][p] = self.beta1 * self.m[l][p] + (1 - self.beta1) * self.model[l].parameters[p].dw

                # Update biased second raw moment estimate
                self.v[l][p] = self.beta2 * self.v[l][p] + (1 - self.beta2) * self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                # (Compute bias-corrected first moment estimate
                m_cap = self.m[l][p] / (1 - np.power(self.beta1, self.t))

                # Compute bias-corrected second raw moment estimate
                v_cap = self.v[l][p] / (1 - np.power(self.beta2, self.t))

                # Update parameters
                self.model[l].parameters[p].w -= self.lr * m_cap / (np.sqrt(v_cap) + self.eps)

    # Adagrad optimization function
    def Adagrad(self):
        if self.t < 1: print('using Adagrad')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # update memory
                self.m[l][p] += self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                # Update parameters
                self.model[l].parameters[p].w -= self.lr * self.model[l].parameters[p].dw / np.sqrt(self.m[l][p] + self.eps)

    # Adadelta optimization function
    def Adadelta(self):
        # useful: https://arxiv.org/pdf/1212.5701.pdf
        if self.t < 1: print('using Adadelta')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # Accumulate Gradient:
                self.m[l][p] = self.ro * self.m[l][p] + (1 - self.ro) * self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                # Compute Update:
                delta = -np.sqrt((self.v[l][p] + self.eps) / (self.m[l][p] + self.eps)) * self.model[l].parameters[p].dw

                # Accumulate Updates:
                self.v[l][p] = self.ro * self.v[l][p] + (1 - self.ro) * delta * delta

                # Apply Update:
                self.model[l].parameters[p].w += delta

    #define optimizers


# model class to add useful features like save, load model from files
class Model:
    def __init__(self):
        self.layers = []

    def save(self, file=None):  # model.save() - saves the state of the network
        print('saving model...')
        layers = []

        for layer in self.layers:
            parameters = []
            for parameter in layer.parameters:
                parameters.append(parameter.w)
            layers.append(parameters)

        if file == None:
            file = str(self.__class__).strip('<>').split()[1].strip("\'").split('.')[1]

        np.save(file+'.npy', layers)

        print('model saved in', file)

    def load(self, file=None):  # model.load() - loads the state of net from a file
        print('loading model from', file)
        if file == None:
            file = str(self.__class__).strip('<>').split()[1].strip("\'").split('.')[1]+'.npy'

        layers = np.load(file, allow_pickle=True)

        for layer_act, layer in zip(self.layers, layers):
            for parameter_act, parameter in zip(layer_act.parameters, layer):
                parameter_act.w = parameter

        print('model loaded!')

    def get_parameters(self):   # access parameters of the model with this func
        return self.layers


# TODO: define regularizations, asserts, batch, utils, GPU support, examples
