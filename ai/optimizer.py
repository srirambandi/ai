import numpy as np
from abc import ABC, abstractmethod
from ai.graph import G


# Optimizers to take that drunken step down the hill
# useful: https://arxiv.org/pdf/1609.04747.pdf
class Optimizer(ABC):
    def __init__(self, parameters, graph=G):
        self.parameters = parameters  # a list of all layers of the model
        self.t = 0  # iteration count
        self.graph = graph        

    # a very important step in learning time
    def zero_grad(self):
        # clearing out the backprop operations from the list
        self.graph.nodes = list()
        self.graph.node_count = 0

        # resetting the gradients of model parameters to zero
        for parameter in self.parameters:
            parameter.grad = np.zeros(parameter.shape)

    @abstractmethod
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, momentum=0, graph=G):
        super().__init__(parameters, graph=graph)
        self.lr = lr    # size of the step to update the parameters
        self.momentum = momentum
        self.m = list()
        
        for parameter in self.parameters:
            self.m.append(np.zeros(parameter.shape))

    def __repr__(self):
        return(f'SGD(lr={self.lr}, momentum={self.momentum})')

    def step(self):
        for p in range(len(self.parameters)):

            if self.momentum > 0.0:
                # momentum update
                self.m[p] = self.momentum * self.m[p] + self.lr * self.parameters[p].grad

                # Update parameters with momentum SGD
                self.parameters[p].data -= self.m[p]

            else:
                # Update parameters with vanilla SGD
                self.parameters[p].data -= self.lr * self.parameters[p].grad


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, graph=G):
        super().__init__(parameters, graph=graph)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = list()
        self.v = list()

        for parameter in self.parameters:
            self.m.append(np.zeros(parameter.shape))
            self.v.append(np.zeros(parameter.shape))

    def __repr__(self):
        return(f'Adam()')

    def step(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf

        self.t += 1
        for p in range(len(self.parameters)):

            # Update biased first moment estimate
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * self.parameters[p].grad

            # Update biased second raw moment estimate
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * self.parameters[p].grad * self.parameters[p].grad

            # (Compute bias-corrected first moment estimate
            m_cap = self.m[p] / (1 - np.power(self.beta1, self.t))

            # Compute bias-corrected second raw moment estimate
            v_cap = self.v[p] / (1 - np.power(self.beta2, self.t))

            # Update parameters
            self.parameters[p].data -= self.lr * m_cap / (np.sqrt(v_cap) + self.eps)


class Adagrad(Optimizer):
    def __init__(self, parameters, lr=0.001, eps=1e-8, graph=G):
        super().__init__(parameters, graph=graph)
        self.lr = lr
        self.eps = eps
        self.grad_square

        for parameter in self.parameters:
            self.grad_square.append(np.zeros(parameter.shape))

    def __repr__(self):
        return(f'Adagrad()')

    def step(self):
        for p in range(len(self.parameters)):

            # update memory
            self.grad_square[p] += self.parameters[p].grad * self.parameters[p].grad

            # Update parameters
            self.parameters[p].data -= self.lr * self.parameters[p].grad / np.sqrt(self.grad_square[p] + self.eps)


class Adadelta(Optimizer):
    def __init__(self, parameters, rho=0.95, eps=1e-8, graph=G):
        super().__init__(parameters, graph=graph)
        self.rho = rho
        self.eps = eps
        self.m = list()
        self.v = list()

        for parameter in self.parameters:
            self.m.append(np.zeros(parameter.shape))
            self.v.append(np.zeros(parameter.shape))

    def __repr__(self):
        return(f'Adadelta()')

    def step(self):
        # useful: https://arxiv.org/pdf/1212.5701.pdf

        for p in range(len(self.parameters)):

            # Accumulate Gradient:
            self.m[p] = self.rho * self.m[p] + (1 - self.rho) * self.parameters[p].grad * self.parameters[p].grad

            # Compute Update:
            delta = -np.sqrt((self.v[p] + self.eps) / (self.m[p] + self.eps)) * self.parameters[p].grad

            # Accumulate Updates:
            self.v[p] = self.rho * self.v[p] + (1 - self.rho) * delta * delta

            # Apply Update:
            self.parameters[p].data += delta


class RMSprop(Optimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, graph=G):
        super().__init__(parameters, graph=graph)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = list()
        
        for parameter in self.parameters:
            self.v.append(np.zeros(parameter.shape))

    def __repr__(self):
        return(f'RMSprop()')

    def step(self):
        # useful: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

        for p in range(len(self.parameters)):

            # Accumulating moving average of the square of the Gradient:
            self.v[p] = self.alpha * self.v[p] + (1 - self.alpha) * self.parameters[p].grad * self.parameters[p].grad

            # Apply Update:
            self.parameters[p].data -= self.lr * self.parameters[p].grad / (np.sqrt(self.v[p]) + self.eps)

#define optimizers
