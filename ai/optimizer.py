import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


# Optimizers to take that drunken step down the hill
class Optimizer:
    def __init__(self, parameters, optim_fn='SGD', lr=3e-4, momentum=0.0, eps=1e-8, beta1=0.9, beta2=0.999, ro=0.95, graph=G):
        self.parameters = parameters  # a list of all layers of the model
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
            for parameter in self.parameters:
                self.m.append(np.zeros(parameter.shape))

            if self.optim_fn == 'Adam' or self.optim_fn == 'Adadelta':

                # Adam: 2nd moment(raw variance here) of gradients
                # Adadelta: Accumulates updates
                for parameter in self.parameters:
                    self.v.append(np.zeros(parameter.shape))

    def __str__(self):
        return('Optimizer(optim_fn={}, lr={}, momentum={})'.format(
            self.optim_fn, self.lr, self.momentum))

    # a very important step in learning time
    def zero_grad(self):
        # clearing out the backprop operations from the list
        self.graph.backprop = []
        self.graph.nodes = 0

        # resetting the gradients of model parameters to zero
        for parameter in self.parameters:
            parameter.grad = np.zeros(parameter.shape)

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
          raise 'No such optimization function'

    # Stochastic Gradient Descent optimization function
    def SGD(self):
        if self.t < 1: print('using SGD')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

            if self.momentum > 0.0:
                # momentum update
                delta = self.momentum * self.m[p] - self.lr * self.parameters[p].grad

                # store delta for next iteration
                self.m[p] = delta

                # Update parameters with momentum SGD
                self.parameters[p].data += delta

            else:
                # Update parameters with vanilla SGD
                self.parameters[p].data -= self.lr * self.parameters[p].grad


    # Adam optimization function
    def Adam(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf
        if self.t < 1: print('using Adam')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

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

    # Adagrad optimization function
    def Adagrad(self):
        if self.t < 1: print('using Adagrad')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

            # update memory
            self.m[p] += self.parameters[p].grad * self.parameters[p].grad

            # Update parameters
            self.parameters[p].data -= self.lr * self.parameters[p].grad / np.sqrt(self.m[p] + self.eps)

    # Adadelta optimization function
    def Adadelta(self):
        # useful: https://arxiv.org/pdf/1212.5701.pdf
        if self.t < 1: print('using Adadelta')

        self.t += 1
        for p in range(len(self.parameters)):
            # clip gradients
            self.parameters[p].grad = np.clip(self.parameters[p].grad, -5.0, 5.0)

            # Accumulate Gradient:
            self.m[p] = self.ro * self.m[p] + (1 - self.ro) * self.parameters[p].grad * self.parameters[p].grad

            # Compute Update:
            delta = -np.sqrt((self.v[p] + self.eps) / (self.m[p] + self.eps)) * self.parameters[p].grad

            # Accumulate Updates:
            self.v[p] = self.ro * self.v[p] + (1 - self.ro) * delta * delta

            # Apply Update:
            self.parameters[p].data += delta

    #define optimizers
