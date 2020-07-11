import numpy as np
import ai.parameter


# Computational Graph wannabe: stores the backward operation for every
# forward operation during forward-propagation, in a breadth-fist manner
class ComputationalGraph:
    def __init__(self, grad_mode=True):
        self.grad_mode = grad_mode
        self.nodes = list()

    # functions required for deep learning models and their respective backward operations
    def dot(self, W, x):    # dot product of vectors and matrices

        assert W.shape[1] == x.shape[0], 'shape mismatch in dot() operation - W: {}, x: {}'.format(W.shape, x.shape)
        out = ai.parameter.Parameter(data=np.dot(W.data, x.data), graph=self)

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
        out = ai.parameter.Parameter(data=np.add(x.data, y.data), graph=self)

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
        out = ai.parameter.Parameter(data=np.subtract(x.data, y.data), graph=self)

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
        out = ai.parameter.Parameter(data=np.multiply(x.data, y.data), graph=self)

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
        out = ai.parameter.Parameter(data= np.divide(x.data, y.data + 1e-8), graph=self)

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
        out = ai.parameter.Parameter(data=res, graph=self)

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
        out = ai.parameter.Parameter(h.shape, init_zeros=True, graph=self)
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
        out = ai.parameter.Parameter(data=np.log(h.data + 1e-8), graph=self)

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

        out = ai.parameter.Parameter(data=out, graph=self)

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

        out = ai.parameter.Parameter(data=out, graph=self)

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

        out = ai.parameter.Parameter(data=out, graph=self)

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

        out = ai.parameter.Parameter(data=out, graph=self)

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

        out = ai.parameter.Parameter(data=out, graph=self)

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
        out = ai.parameter.Parameter(data=dropout_mask*x.data, graph=self)

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
        out = ai.parameter.Parameter(data=np.maximum(z.data, 0), graph=self)

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
        out = ai.parameter.Parameter(data=np.maximum(z.data, alpha * z.data), graph=self)

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
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
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
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
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
        out = ai.parameter.Parameter(data=np.tanh(z.data), graph=self)

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
            o = ai.parameter.Parameter(data=e, graph=self)
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
        out = ai.parameter.Parameter(data=x.data.T, graph=self)

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
        out = ai.parameter.Parameter(new_shape, init_zeros=True, graph=self)
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
