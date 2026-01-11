import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# |    ||
#
# ||   |_
#
# is this loss? Yes, it is.
class MSELoss(Module):
    def __init__(self, graph=G):
        super().__init__()
        self.graph = graph

    def __repr__(self):
        return(f'MSELoss()')

    def forward(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=np.array(y_true, dtype=float), requires_grad=False, graph=self.graph)
        elif y_true.data.dtype != float:
            y_true = Parameter(data=np.array(y_true.data, dtype=float), requires_grad=False, graph=self.graph)

        # L = (y_out - y_true)^2
        l = self.graph.sum(self.graph.power((y_out - y_true), 2))
        # avg_loss = (1/(B * D)) * sigma{i = 1,..,B}(loss[i])
        l = l / float(y_true.numel())

        return l


class CrossEntropyLoss(Module):
    def __init__(self, graph=G):
        super().__init__()
        self.graph = graph

    def __repr__(self):
        return(f'CrossEntropyLoss()')

    def forward(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=np.array(y_true, dtype=float), requires_grad=False, graph=self.graph)
        elif y_true.data.dtype != float:
            y_true = Parameter(data=np.array(y_true.data, dtype=float), requires_grad=False, graph=self.graph)

        # softmax on logits
        y_prob = self.graph.softmax(y_out, axis=-1)
        # -Summation(t * log(p))
        l = self.graph.sum((y_true * self.graph.log(y_prob))) * -1.0
        # avg_loss = (1/B)*sigma{i = 1,..,B}(loss[i])
        l = l / float(y_true.shape[0])

        return l


class BCELoss(Module):
    def __init__(self, graph=G):
        super().__init__()
        self.graph = graph
        self.eps = 1e-8

    def __repr__(self):
        return(f'BCELoss()')

    def forward(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=np.array(y_true, dtype=float), requires_grad=False, graph=self.graph)
        elif y_true.data.dtype != float:
            y_true = Parameter(data=np.array(y_true.data, dtype=float), requires_grad=False, graph=self.graph)

        # clamp probabilities into (eps, 1 - eps)
        y_out = self.graph.relu(y_out - self.eps) + self.eps
        y_out = 1.0 - self.graph.relu((1.0 - self.eps) - y_out)

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

        return l


class JSDivLoss(Module):
    def __init__(self, graph=G):
        super().__init__()
        self.graph = graph

    def __repr__(self):
        return(f'JSDivLoss()')

    def forward(self, y_out, y_true):

        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=np.array(y_true, dtype=float), requires_grad=False, graph=self.graph)
        elif y_true.data.dtype != float:
            y_true = Parameter(data=np.array(y_true.data, dtype=float), requires_grad=False, graph=self.graph)

        # mean probability: (P + Q)/2
        y_mean = (y_out + y_true) / 2.0
        # KL(P || M)
        kl_1 = KLDivLoss(graph=self.graph).forward(self.graph.log(y_mean), y_true)
        # KL(Q || M)
        kl_2 = KLDivLoss(graph=self.graph).forward(self.graph.log(y_mean), y_out)
        # JS(P, Q) = 1/2*(KL(P || (P + Q)/2) + KL(Q || (P + Q)/2))
        l = (kl_1 + kl_2) / 2.0

        return l


class KLDivLoss(Module):
    def __init__(self, graph=G):
        super().__init__()
        self.graph = graph

    def __repr__(self):
        return(f'KLDivLoss()')

    def forward(self, y_out, y_true):
        if not isinstance(y_true, Parameter):
            y_true = Parameter(data=np.array(y_true, dtype=float), requires_grad=False, graph=self.graph)
        elif y_true.data.dtype != float:
            y_true = Parameter(data=np.array(y_true.data, dtype=float), requires_grad=False, graph=self.graph)

        # KL(P || Q) = Summation(P * (log(P) - log(Q)))
        l = self.graph.sum(y_true * (self.graph.log(y_true) - y_out))
        l = l / float(y_true.shape[0])

        return l


class TestLoss(Module):
    def __init__(self, graph=G):
        super().__init__()
        self.graph = graph

    def __repr__(self):
        return(f'TestLoss()')

    def forward(self, y_out):

        # a test loss score function that measures the sum of elements of each output vector as the loss of that sample
        # helps identify leaks in between samples in a batch
        l = self.graph.sum(y_out)
        l = l / float(y_out.shape[0])

        return l

#define more loss functions
