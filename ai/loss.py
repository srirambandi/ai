from ai.parameter import Parameter
from ai.graph import G


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
        return(f'Loss(loss_fn={self.loss_fn})')

    def mse_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, requires_grad=False, graph=self.graph)

        # L = (y_out - y_true)^2
        l = self.graph.sum(self.graph.power((y_out - y_true), 2))
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = l / float(y_true.shape[0])

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def cross_entropy_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, requires_grad=False, graph=self.graph)

        # KL(P || Q): Summation(P*log(P)){result: 0} - Summation(P*log(Q))
        l = self.graph.sum((y_true * self.graph.log(y_out))) * -1.0
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = l / float(y_true.shape[0])

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def bce_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, requires_grad=False, graph=self.graph)

        # class 2 output: 1 - c1
        c2 = (y_out - 1.0) * -1.0
        # class 2 target: 1 - t1
        t2 = (y_true - 1.0) * -1.0

        # -Summation(t1*log(c1))
        l1 = self.graph.sum((y_true * self.graph.log(y_out))) * -1.0
        # -Summation((1 - t1)*log(1 - c1))
        l2 = self.graph.sum((t2 * self.graph.log(c2))) * -1.0
        # loss = -Summation(t1*log(c1)) -Summation((1 - t1)*log(1 - c1))
        l = l1 + l2
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = l / float(y_true.shape[0])

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def js_divergence_loss(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=y_true, requires_grad=False, graph=self.graph)

        # mean probability: (P + Q)/2
        y_mean = (y_out + y_true) / 2.0
        # KL(P || (P + Q)/2): Summation(P*log(P)){result: 0} - Summation(P*log((P+Q)/2))
        kl_1 = self.graph.sum(self.graph.multiply(y_true, self.graph.log(y_mean))) * -1.0
        # KL(Q || (P + Q)/2): Summation(Q*log(Q)) - Summation(Q*log((P+Q)/2))
        kl_2 = self.graph.sum((y_out * (self.graph.log(y_out) - self.graph.log(y_mean))))
        # JS(P, Q) = 1/2*(KL(P || (P + Q)/2) + KL(Q || (P + Q)/2))
        l = (kl_1 + kl_2) / 2.0
        # avg_loss = (1/m)*sigma{i = 1,..,m}(loss[i])
        l = l / float(y_true.shape[0])

        l.grad[0, 0] = 1.0  # dl/dl = 1.0

        return l

    def test_loss(self, y_out):

        # a test loss score function that measures the sum of elements of each output vector as the loss of that sample
        # helps identify leaks in between samples in a batch
        l = self.graph.sum(y_out)
        l = l / float(y_out.shape[0])

        l.grad[0, 0] = 1.0

        return l

    #define loss functions


class MSELoss(Loss):
    def __init__(self, loss_fn=None, graph=G):
        super().__init__(loss_fn, graph)