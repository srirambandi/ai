import numpy as np
from .parameter import Parameter


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
        ch = K.shape[1]     # number of input channels
        k = K.shape[2:]     # don't confuse b/w K(big) - the kernel set and k(small) - a single kernel  of some cth-channel in a kth-filter
        i = x.shape[1:]     # input shape of any channel of the input feature map before padding
        output_maps_shape = (fi, *tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))) # output feature maps shape - (# of filters, o_i, o_j)
        pad_shape = (ch, *tuple(map(lambda a, b: a + 2*b, i, p)))   # padded input feature maps shape - (channels, new_i, new_j)

        out = np.zeros(output_maps_shape)  # output feature maps

        pad_input = np.zeros(pad_shape)
        # padded input - copying the actual input onto pad input centre
        pad_input[:, p[0]:pad_input.shape[1]-p[0], p[1]:pad_input.shape[2]-p[1]] += x.w
        pad_input = np.stack([pad_input for i in range(fi)])    # stack fi times to help fast computation

        for r in range(out.shape[1]):        # convolving operation here
            for c in range(out.shape[2]):    # traversing rous and columns of feature map

                # multiplying traversed grid portions of padded input feature maps with kernel grids element-wise
                # and summing the resulting matrix to produce elements of output maps
                out[:, r, c] += np.sum(np.multiply(pad_input[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]], K.w), axis=(1, 2, 3))

        output_feature_maps = Parameter(out.shape, init_zeros=True)
        output_feature_maps.w = out    # the whole set of output feature maps, has same numeber of maps as filters in the kernel set

        if self.grad_mode:
            def backward():
                # print('conv2d')
                if K.eval_grad:

                    for r in range(output_feature_maps.shape[1]):
                        for c in range(output_feature_maps.shape[2]):

                            # solving gradient for each kernel filter that caused the elements in r, c position of every output filter
                            # sketch and think, with input stacked fi times to make computation fast
                            _ = np.stack([np.full_like(np.zeros(K.shape[1:]), output_feature_maps.dw[i, r, c]) for i in range(fi)])

                            # updating the kernel filter set gradient - there will be RxC such updates
                            K.dw += _*pad_input[:, :, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]]

                if x.eval_grad:

                    pad_input_grad = np.zeros(pad_shape)

                    for r in range(output_feature_maps.shape[1]):
                        for c in range(output_feature_maps.shape[2]):

                            # solving gradient for input feature map that caused the elements in r, c position of every output filter
                            # similar to kernel gradient method, but the matrix collapses along filters dimention using sum
                            _ = np.stack([np.full_like(np.zeros(K.shape[1:]), output_feature_maps.dw[i, r, c]) for i in range(fi)])
                            pad_input_grad[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += np.sum(_*K.w, axis=0)

                    # cutting the padded portion from the input-feature-map's gradient
                    # and updating the gradient of actual input feature map(non-padded) - unpadding and updating
                    pad_input_grad = pad_input_grad[:, p[0]:pad_input_grad.shape[1]-p[0], p[1]:pad_input_grad.shape[2]-p[1]]
                    x.dw += pad_input_grad

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

        ch = x.shape[0]     # number of input channels(panes)
        i = x.shape[1:]     # input shape of any channel of the input feature map before padding
        pool_shape = (ch, *tuple(map(lambda i, k, s, p: int((i + 2*p - k)/s + 1), i, k, s, p))) # shape after maxpool
        pad_shape = (ch, *tuple(map(lambda a, b: a + 2*b, i, p)))  # padded input shape

        out = np.zeros((pool_shape))

        pad_input = np.zeros(pad_shape)
        pad_input[:, p[0]:pad_input.shape[1]-p[0], p[1]:pad_input.shape[2]-p[1]] += x.w # copying the input onto padded matrix

        for r in range(out.shape[1]):       # convolving operation here(kinda)
            for c in range(out.shape[2]):   # traversing rous and columns of feature map

                # selecting max element in the current position where kernel
                # sits on feature map; the kernel moves in a convolution manner similar to conv2d
                out[:, r, c] = np.max(pad_input[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]], axis=(1, 2))

        pool_maps = Parameter(out.shape, init_zeros=True)
        pool_maps.w = out

        if self.grad_mode:
            def backward():
                # print('maxpool2d')
                if x.eval_grad:

                    pad_input_grad = np.zeros(pad_shape)    # padded input gradient

                    for r in range(pool_maps.shape[1]):
                        for c in range(pool_maps.shape[2]):

                            # mask that captures location of max values in the cuboid volume of padded input panes
                            # that caused elements in every respective pool map pane at (r, c) location
                            mask = np.zeros((ch, *k))
                            for count in range(ch):
                                _ = pad_input[count, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]]
                                ind = (count, *np.unravel_index(np.argmax(_, axis=None), _.shape))
                                # mask updated for only maximum element receiving the gradient in current channel and position(r, c)
                                mask[ind] = 1.0*pool_maps.dw[count, r, c]

                            # copying mask onto gradient matrix, the volume that
                            # caused the values in (:, r, c). RxC such updates
                            pad_input_grad[:, r*s[0]:r*s[0] + k[0], c*s[1]:c*s[1] + k[1]] += mask

                    # cutting the padded portion from the input gradient
                    # and updating the gradient of actual input(non-padded) - unpadding and updating
                    pad_input_grad = pad_input_grad[:, p[0]:pad_input_grad.shape[1]-p[0], p[1]:pad_input_grad.shape[2]-p[1]]
                    x.dw += pad_input_grad

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
