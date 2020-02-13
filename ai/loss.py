import numpy as np
from .parameter import Parameter
from .graph import ComputationalGraph, G


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
