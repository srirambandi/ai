import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


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

        if type(y_true) is not Parameter:
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

        if type(y_true) is not Parameter:
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

        if type(y_true) is not Parameter:
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

        if type(y_true) is not Parameter:
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
