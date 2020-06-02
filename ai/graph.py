import numpy as np
import ai.parameter


# Computational Graph wannabe: stores the backward operation for every
# forward operation during forward-propagation, in a breadth-fist manner
class ComputationalGraph:
    def __init__(self, grad_mode=True):
        self.grad_mode = grad_mode
        self.backprop = []
        self.nodes = 0

    # operations required for deep learning models and their backward operations
    def dot(self, W, x):    # dot product of vectors and matrices

        assert W.shape[1] == x.shape[0], 'shape mismatch in dot() operation'
        shape = (W.data.shape[0], x.data.shape[1])
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
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

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def add(self, x, y, axis=()):    # element wise addition
        # bias should be passed in position of y
        shape = x.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.add(x.data, y.data)

        if self.grad_mode:
            def backward():
                # print('add')
                if x.eval_grad:
                    x.grad += out.grad
                if y.eval_grad:
                    y.grad += np.sum(out.grad, axis = axis).reshape(y.shape)   # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def subtract(self, x, y, axis=()):   # element wise subtraction
        shape = x.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.subtract(x.data, y.data)

        if self.grad_mode:
            def backward():
                # print('subtract')
                if x.eval_grad:
                    x.grad += out.grad
                if y.eval_grad:
                    y.grad -= np.sum(out.grad, axis=axis).reshape(y.shape)  # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def multiply(self, x, y, axis=()):   # element wise vector multiplication
        # not for scalar multiply
        shape = x.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.multiply(x.data, y.data)

        if self.grad_mode:
            def backward():
                # print('multiply')
                if x.eval_grad:
                    x.grad += np.multiply(out.grad, y.data)
                if y.eval_grad:
                    y.grad += np.sum(np.multiply(out.grad, x.data), axis=axis).reshape(y.shape) # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def divide(self, x, y, axis=()):   # element wise vector division
        shape = x.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.divide(x.data, y.data)

        if self.grad_mode:
            def backward():
                # print('divide')
                if x.eval_grad:
                    x.grad += np.multiply(out.grad, np.divide(1.0, y.data))
                if y.eval_grad:
                    y.grad += np.sum(np.multiply(out.grad, np.multiply(out.data, np.divide(-1.0, y.data))), axis=axis).reshape(y.shape) # in case of unequal sizes of inputs

                # return (x.grad, y.grad)

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def sum(self, h, axis=None):   # sum of all elements in the matrix
        if axis == None:
            res = np.sum(h.data).reshape(1, 1)
        else:
            res = np.sum(h.data, axis=axis, keepdims=True)
        out = ai.parameter.Parameter(res.shape, init_zeros=True, graph=self)
        out.data = res

        if self.grad_mode:
            def backward():
                # print('sum')
                if h.eval_grad:
                    h.grad += out.grad

                # return h.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def power(self, h, power):   # element wise power
        out = ai.parameter.Parameter(h.shape, init_zeros=True, graph=self)
        out.data = np.power(h.data, power) if power >= 0 else np.power(h.data, power) + 1e-6     # numerical stability for -ve power

        if self.grad_mode:
            def backward():
                # print('power')
                if h.eval_grad:
                    h.grad += np.multiply(out.grad, power * np.power(h.data, power - 1))

                # return h.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def log(self, h):   # element wise logarithm
        shape = h.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.log(h.data)

        if self.grad_mode:
            def backward():
                # print('log')
                if h.eval_grad:
                    h.grad += np.multiply(out.grad, np.divide(1.0, h.data))

                # return h.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

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

        output_feature_maps = ai.parameter.Parameter(out.shape, init_zeros=True, graph=self)
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

            self.backprop.append(lambda: backward())
            output_feature_maps.node = int(self.nodes)
            self.nodes += 1

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

        output_image = ai.parameter.Parameter(out.shape, init_zeros=True, graph=self)
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

            self.backprop.append(lambda: backward())
            output_image.node = int(self.nodes)
            self.nodes += 1

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

        pool_maps = ai.parameter.Parameter(out.shape, init_zeros=True, graph=self)
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

            self.backprop.append(lambda: backward())
            pool_maps.node = int(self.nodes)
            self.nodes += 1

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
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = dropout_mask*x.data    # drop/sclae

        if self.grad_mode:
            def backward():
                # print('dropout')
                if x.eval_grad:
                    x.grad += out.grad*dropout_mask # only activated units get gradients

                # return x.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    # hidden and output units activations
    def relu(self, z):      # element wise ReLU activations
        shape = z.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.maximum(z.data, 0)

        if self.grad_mode:
            def backward():
                # print('relu')
                if z.eval_grad:
                    z.grad += out.grad.copy()
                    z.grad[z.data < 0] = 0

                # return z.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def lrelu(self, z, alpha=1e-2):      # element wise Leaky ReLU activations
        shape = z.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.maximum(z.data, alpha * z.data)

        if self.grad_mode:
            def backward():
                # print('lrelu')
                if z.eval_grad:
                    z.grad += out.grad.copy()
                    z.grad[z.data < 0] *= alpha

                # return z.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

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

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

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

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    def tanh(self, z):      # element wise tanh activations
        shape = z.shape
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = np.tanh(z.data)

        if self.grad_mode:
            def backward():
                # print('tanh')
                if z.eval_grad:
                    z.grad += np.multiply(1 - np.multiply(out.data, out.data), out.grad)

                # return z.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out

    # data manipulation/view functions
    def split(self, W, sections=1, axis=0):
        outs = np.split(W.data, sections, axis=axis)
        outs_list = []
        for e in outs:
            o = ai.parameter.Parameter(e.shape, init_zeros=True, graph=self)
            o.data = e
            outs_list.append(o)

        if self.grad_mode:
            def backward():
                #print('split')
                outs_grads = [o.grad for o in outs_list]
                if W.eval_grad:
                    W.grad += np.concatenate(outs_grads, axis=axis)

            self.backprop.append(lambda: backward())
            for out in outs_list:
                out.node = int(self.nodes)
            self.nodes += 1

        return outs_list

    def T(self, x):     # transpose
        shape = tuple(reversed(x.shape))
        out = ai.parameter.Parameter(shape, init_zeros=True, graph=self)
        out.data = x.data.T

        if self.grad_mode:
            def backward():
                # print('T')
                if x.eval_grad:
                    x.grad += out.grad.T

                # return x.grad

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

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

            self.backprop.append(lambda: backward())
            out.node = int(self.nodes)
            self.nodes += 1

        return out


G = ComputationalGraph()
