"""
AI library - numpy, python and ai walk into a bar..
Written by Sri Ram Bandi (@_srirambandi_)

BSD License
"""

import numpy as np


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), eval_grad=True, init_zeros=False, mu = 0.0, std = 0.01):
        self.eval_grad = eval_grad  # if the parameter is a variable or an input/scalar
        self.shape = shape
        self.init_zeros = init_zeros
        self.w = np.zeros(shape)
        self.dw = np.zeros(shape)
        self.mu = mu    # mean and variance of the gaussian
        self.std = std  # distribution to initialize the parameter
        self.init_params()

    def init_params(self):
        if not self.init_zeros:
            self.w = self.std*np.random.randn(*self.shape) + self.mu
        return self.w

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

    def add(self, x, y):    # element wise addition
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.add(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('add')
                if x.eval_grad:
                    x.dw += out.dw
                if y.eval_grad:
                    y.dw += out.dw

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def scalar_add(self, x, y):
        # y is the scalar(not exactly, x is a matrix and y is a vector) - this is to support bias addition
        # in Conv2d, might not fit every other situation. Bootstrapped; don't use for other cases
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.add(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('scalar_add')
                if x.eval_grad:
                    x.dw += out.dw
                if y.eval_grad:
                    for ydw, odw in zip(y.dw, out.dw):
                        ydw += np.sum(odw)      # odw volume collapses onto a single element

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def subtract(self, x, y):   # element wise subtraction
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.subtract(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('subtract')
                if x.eval_grad:
                    x.dw += out.dw
                if y.eval_grad:
                    y.dw += -1.0*out.dw  # for the second parameter

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def multiply(self, x, y):   # element wise vector multiplication
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
                    y.dw += np.multiply(out.dw, x.w)

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def scalar_mul(self, x, y):
        # y is the scalar
        if x.shape == (1, 1):
            _ = y
            y = x
            x = _
        shape = x.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.multiply(x.w, y.w)

        if self.grad_mode:
            def backward():
                # print('scalar_mul')
                if x.eval_grad:
                    x.dw += np.multiply(out.dw, y.w)
                if y.eval_grad:
                    # like in the 'multiply', but the vector collapses
                    # onto a single element with sum operation; think
                    y.dw += np.full_like(y.dw, np.sum(np.multiply(out.dw, x.w)))

                # return (x.dw, y.dw)

            self.backprop.append(lambda: backward())

        return out

    def sum(self, h):   # sum of all elements in the matrix
        out = Parameter((1, 1), init_zeros=True)
        out.w[0][0] = np.sum(h.w)

        if self.grad_mode:
            def backward():
                # print('sum')
                if h.eval_grad:
                    h.dw += np.full_like(h.dw, out.dw[0][0])

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
    def conv2d(self, x, K, s = (1, 1), p = (0, 0)):     # 2d convolution operation
        # useful: https://arxiv.org/pdf/1603.07285.pdf
        if type(s) is not tuple:    # these both conditions are already handled in Conv2d class definition
            s = (s, s)              # but adding for compatibility if user calls globally without using Conv2d class
        if type(p) is not tuple:
            p = (p, p)

        out = []

        k = K.w[0, 0].shape     # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel  of some cth-channel in a kth-filter
        i = x.w[0].shape        # input shape of any channel of the input feature map before padding
        output_map_shape = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p)) # output feature map shape
        pad_shape = tuple(map(lambda a, b: a + 2*b, i, p))  # padded input feature map shape

        for kth in range(len(K.w)):  # kth-filter

            output_feat_map = np.zeros(output_map_shape)  # kth-output feature map

            for cth in range(len(x.w)):

                input_feat_map = x.w[cth]   # cth-channel grid (eg., R/G/B)
                pad_channel = np.zeros(pad_shape)
                pad_channel[p[0]:pad_channel.shape[0]-p[0], p[1]:pad_channel.shape[1]-p[1]] += input_feat_map   # padded input channel

                for r in range(output_feat_map.shape[0]):    # traversing rows of kth-output feature map
                    for c in range(output_feat_map.shape[1]):    # traversing columns of kth-output feature map

                        # multiplying traversed grid portions of cth-padded input feature maps with cth-kernel grids element-wise
                        # and summing the resulting matrix to produce elements of k-th kernel
                        output_feat_map[r][c] += np.sum(np.multiply(pad_channel[r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]], K.w[kth, cth]))

            out.append(output_feat_map)

        out = np.stack(out)
        output_feature_maps = Parameter(out.shape, init_zeros=True)
        output_feature_maps.w = out    # the whole set of output feature maps, has same numeber of maps as filters in the kernel set

        if self.grad_mode:
            def backward():
                # print('conv2d')
                if K.eval_grad:
                    for kth in range(len(output_feature_maps.dw)):
                        for cth in range(len(x.w)):

                            kdw = np.zeros(k)   # cth-channel kernel gradient in kth-kernel filter
                            input_feat_map = x.w[cth]   # cth-channel grid (eg., R/G/B)
                            pad_channel = np.zeros(pad_shape)
                            pad_channel[p[0]:pad_channel.shape[0]-p[0], p[1]:pad_channel.shape[1]-p[1]] += input_feat_map

                            map_grad = output_feature_maps.dw[kth]
                            for r in range(map_grad.shape[0]):
                                for c in range(map_grad.shape[1]):
                                    # solving gradient for cth-channel kernel in kth-kernel filter (sketh on paper and think)
                                    kdw += map_grad[r][c]*pad_channel[r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]]

                            # updating gradient of kernel-filter-set at kth-filter and cth-channel
                            K.dw[kth, cth] += kdw

                if x.eval_grad:
                    for kth in range(len(output_feature_maps.dw)):
                        for cth in range(len(K.w[kth])):

                            pad_channel_grad = np.zeros(pad_shape)
                            map_grad = output_feature_maps.dw[kth]
                            for r in range(map_grad.shape[0]):
                                for c in range(map_grad.shape[1]):
                                    pad_channel_grad[r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += map_grad[r][c]*K.w[kth, cth]

                            # cutting the padded portion from the input-feature-map's cth-channel gradient
                            # and updating the cth channel gradient of input feature map(non-padded)
                            pad_channel_grad = pad_channel_grad[p[0]:pad_channel_grad.shape[0]-p[0], p[1]:pad_channel_grad.shape[1]-p[1]]
                            x.dw[cth] += pad_channel_grad

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

        i = x.w[0].shape    # input shape of any channel of the input feature map before padding
        pool_shape = tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p)) # shape after maxpool
        pad_shape = tuple(map(lambda a, b: a + 2*b, i, p))  # padded conv output feature map shape

        for feat_map in x.w:

            pool_map = np.zeros((pool_shape))
            pad_map = np.zeros(pad_shape)
            pad_map[p[0]:pad_map.shape[0]-p[0], p[1]:pad_map.shape[1]-p[1]] += feat_map # copying the input onto padded matrix

            for r in range(pool_map.shape[0]):
                for c in range(pool_map.shape[1]):

                    # selecting max element in the current position where kernel
                    # sits on feature map; the kernel moves in a convolution manner similar to conv2d
                    pool_map[r][c] = np.max(pad_map[r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]])

            out.append(pool_map)

        out = np.stack(out)
        pool_maps = Parameter(out.shape, init_zeros=True)
        pool_maps.w = out

        if self.grad_mode:
            def backward():
                # print('maxpool2d')
                if x.eval_grad:
                    x_grads = []
                    for feat_map, pool_map_grad in zip(x.w, pool_maps.dw):

                        pad_map = np.zeros(pad_shape)
                        pad_map_grad = np.zeros(pad_shape)
                        pad_map[p[0]:pad_map.shape[0]-p[0], p[1]:pad_map.shape[1]-p[1]] += feat_map

                        for r in range(pool_map_grad.shape[0]):
                            for c in range(pool_map_grad.shape[1]):

                                mask = np.zeros(k)
                                a = pad_map[r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]]
                                ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
                                mask[ind] = 1*pool_map_grad[r][c]   # mask with only maximum element receiving the gradient in current position
                                pad_map_grad[r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += mask # copying mask onto gradient matrix

                        pad_map_grad = pad_map_grad[p[0]:pad_map_grad.shape[0]-p[0], p[1]:pad_map_grad.shape[1]-p[1]]
                        x_grads.append(pad_map_grad)

                    x.dw += np.stack(x_grads)

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

    def sigmoid(self, z):   # element wise sigmoid activations
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = 1.0/(1.0 + np.exp(-1.0*z.w))

        if self.grad_mode:
            def backward():
                # print('sigmoid')
                if z.eval_grad:
                    z.dw += np.multiply(np.multiply(out.w, 1 - out.w), out.dw)

                # return z.dw

            self.backprop.append(lambda: backward())

        return out

    def softmax(self, z):   # calculates probs for the unnormalized log probabilities of previous layer
        shape = z.shape
        out = Parameter(shape, init_zeros=True)
        out.w = np.exp(z.w - np.max(z.w))/np.sum(np.exp(z.w - np.max(z.w)))

        if self.grad_mode:
            def backward():
                # print('softmax')
                if z.eval_grad:
                    z.dw += np.dot(np.diag(out.w[:,0]) - np.dot(out.w, out.w.T), out.dw)

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
        if new_shape == None:
            new_shape = x.w.reshape(-1, 1).shape
        out = Parameter(new_shape, init_zeros=True)
        out.w = x.w.reshape(-1, 1)

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
    def __init__(self, h_p = 0, h_n = 0, init_zeros=False):
        self.h_p = h_p  # previous layer units
        self.h_n = h_n  # next layer units
        self.init_zeros = init_zeros
        self.init_params()

    def init_params(self):
        self.W = Parameter((self.h_n, self.h_p), init_zeros=self.init_zeros)  # weight volume
        self.b = Parameter((self.h_n, 1), init_zeros=True)   # bias vector
        self.parameters = [self.W, self.b]  # easy access of the layer params

    def forward(self, x):
        # making the input compatible with graph operations
        if type(x) is not Parameter:
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=True, init_zeros=True)
            x.w = _

        if len(x.shape) > 2 or x.shape[1] != 1:
            x = G.reshape(x)

        out = G.add(G.dot(self.W, x), self.b)   # y = Wx + b

        return out


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


# sequence models: LSTM cell
class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size    # size of the input at each recurrent tick
        self.hidden_size = hidden_size  # size of hidden units h and c
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
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=False, init_zeros=True)
            x.w = _


        gates = G.add(G.add(G.dot(self.W_hh, h), self.b_hh), G.add(G.dot(self.W_ih, x), self.b_ih))

        # forget, input, gate(also called cell gate - different from cell state), output gates of the lstm cell
        # useful: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        f, i, g, o = G.split(gates, sections=4, axis=0)

        f = G.sigmoid(f)
        i = G.sigmoid(i)
        g = G.tanh(g)
        o = G.sigmoid(o)

        c = G.add(G.multiply(f, c), G.multiply(i, g))
        h = G.multiply(o, G.tanh(c))

        return (h, c)


# sequence models: RNN cell
class RNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
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
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=False, init_zeros=True)
            x.w = _

        h = G.add(G.add(G.dot(self.W_hh, h), G.dot(self.W_ih, x)), self.b_hh)
        h = G.tanh(h)

        return h


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

    # backprop is called here, computes gradients of the parameters
    # Loss and Computational Graph can call the back propagarion
    def backward(self):
        self.graph.backward()

    def MSELoss(self, y_out, y_true):
        # loss as a list for the case when the graph has many outputs, like in an LSTM roll-out with outputs at every tick
        loss = []

        for y_o, y_t in zip(y_out, y_true):
            if type(y_t) is not Parameter:
                shape = y_t.shape
                _ = y_t
                y_t = Parameter(shape, eval_grad=False, init_zeros=True)
                y_t.w = _

            # L = (y_o - y_t)^2
            l = self.graph.dot(self.graph.T(self.graph.subtract(y_o, y_t)), self.graph.subtract(y_o, y_t))

            l.dw[0][0] = 1.0  # dl/dl = 1.0

            loss.append(l)

        return loss

    def CrossEntropyLoss(self, y_out, y_true):
        loss = []

        for y_o, y_t in zip(y_out, y_true):
            if type(y_t) is not Parameter:
                shape = y_t.shape
                _ = y_t
                y_t = Parameter(shape, eval_grad=False, init_zeros=True)
                y_t.w = _

            neg_one = Parameter((1, 1), init_zeros=True, eval_grad=False)
            neg_one.w[0][0] = -1.0  # just a -1 to make the l.dw look same in all the loss defs (dl/dl = 1)

            # L = -Summation(y_t*log(y_o))
            l = self.graph.scalar_mul(self.graph.sum(self.graph.multiply(y_t, self.graph.log(y_o))), neg_one)

            l.dw[0][0] = 1.0  # dl/dl = 1.0

            loss.append(l)

        return loss

    #define loss functions


# Optimizers to take that drunken step down the hill
class Optimizer:
    def __init__(self, model, optim_fn='SGD', lr=3e-4, momentum=0.0, eps=1e-8, beta1=0.9, beta2=0.999, graph=G):
        self.model = model  # a list of all layers of the model
        self.optim_fn = optim_fn    # the optimizing function(SGD, Adam, Adagrad, RMSProp)
        self.lr = lr    # alpha: size of the step to update the parameters
        self.momentum = momentum
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.graph = graph
        self.t = 0  # iteration count
        self.m = [] # 1st moment(mean) vector of gradients
        self.v = [] # 2nd moment(raw variance here) of gradients

        if optim_fn == 'Adam' or optim_fn == 'Adagrad':
            for layer in self.model:
                layer_param = []
                for parameter in layer.parameters:
                    layer_param.append(np.zeros(parameter.shape))
                self.m.append(layer_param)

            if optim_fn == 'Adam':
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

        if self.optim_fn == 'Adam':
            return self.Adam()
        elif self.optim_fn == 'SGD':
            return self.SGD()
        elif self.optim_fn == 'Adagrad':
            return self.Adagrad()

    # Stochastic Gradient Descent optimizing function
    def SGD(self):
        if self.t < 1: print('using SGD')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # update
                self.model[l].parameters[p].w -= self.lr * self.model[l].parameters[p].dw

    # Adam optimizing function
    def Adam(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf
        if self.t < 1: print('using Adam')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # updates of 1st and 2nd moments
                self.m[l][p] = self.beta1 * self.m[l][p] + (1 - self.beta1) * self.model[l].parameters[p].dw
                self.v[l][p] = self.beta2 * self.v[l][p] + (1 - self.beta2) * self.model[l].parameters[p].dw * self.model[l].parameters[p].dw
                m_cap = self.m[l][p] / (1 - np.power(self.beta1, self.t))     # bias correction of first moment
                v_cap = self.v[l][p] / (1 - np.power(self.beta2, self.t))     # bias correction of 2nd moment

                # update
                self.model[l].parameters[p].w -= self.lr * m_cap / (np.sqrt(v_cap) + self.eps)

    # Adagrad optim function
    def Adagrad(self):
        if self.t < 1: print('using Adagrad')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # update memory
                self.m[l][p] += self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                #update
                self.model[l].parameters[p].w -= self.lr * self.model[l].parameters[p].dw / np.sqrt(self.m[l][p] + self.eps)

    #define optimizers


# model class to add useful features like save, load model from files
class Model:
    def __init__(self):
        self.layers = []

    def save(self, file=None):
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

    def load(self, file=None):
        print('loading model from', file)
        if file == None:
            file = str(self.__class__).strip('<>').split()[1].strip("\'").split('.')[1]+'.npy'

        layers = np.load(file, allow_pickle=True)

        for layer_act, layer in zip(self.layers, layers):
            for parameter_act, parameter in zip(layer_act.parameters, layer):
                parameter_act.w = parameter

        print('model loaded!')

    def get_parameters(self):
        return self.layers


#define operations, loss, optim, models, regularizations, asserts, batch, n-dimentional-inputs, utils, GPU support, tests
