import numpy as np
from .parameter import Parameter
from .graph import ComputationalGraph, G


# Optimizers to take that drunken step down the hill
class Optimizer:
    def __init__(self, model, optim_fn='SGD', lr=3e-4, momentum=0.0, eps=1e-8, beta1=0.9, beta2=0.999, ro=0.95, graph=G):
        self.model = model  # a list of all layers of the model
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
            for layer in self.model:
                layer_param = []
                for parameter in layer.parameters:
                    layer_param.append(np.zeros(parameter.shape))
                self.m.append(layer_param)

            if self.optim_fn == 'Adam' or self.optim_fn == 'Adadelta':

                # Adam: 2nd moment(raw variance here) of gradients
                # Adadelta: Accumulates updates
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

        if self.optim_fn == 'SGD':
            return self.SGD()
        elif self.optim_fn == 'Adam':
            return self.Adam()
        elif self.optim_fn == 'Adagrad':
            return self.Adagrad()
        elif self.optim_fn == 'Adadelta':
            self.eps = 1e-6
            return self.Adadelta()

    # Stochastic Gradient Descent optimization function
    def SGD(self):
        if self.t < 1: print('using SGD')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                if self.momentum > 0.0:
                    # momentum update
                    delta = self.momentum * self.m[l][p] - self.lr * self.model[l].parameters[p].dw

                    # store delta for next iteration
                    self.m[l][p] = delta

                    # Update parameters with momentum SGD
                    self.model[l].parameters[p] += delta

                else:
                    # Update parameters with vanilla SGD
                    self.model[l].parameters[p].w -= self.lr * self.model[l].parameters[p].dw


    # Adam optimization function
    def Adam(self):
        # useful: https://arxiv.org/pdf/1412.6980.pdf
        if self.t < 1: print('using Adam')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # Update biased first moment estimate
                self.m[l][p] = self.beta1 * self.m[l][p] + (1 - self.beta1) * self.model[l].parameters[p].dw

                # Update biased second raw moment estimate
                self.v[l][p] = self.beta2 * self.v[l][p] + (1 - self.beta2) * self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                # (Compute bias-corrected first moment estimate
                m_cap = self.m[l][p] / (1 - np.power(self.beta1, self.t))

                # Compute bias-corrected second raw moment estimate
                v_cap = self.v[l][p] / (1 - np.power(self.beta2, self.t))

                # Update parameters
                self.model[l].parameters[p].w -= self.lr * m_cap / (np.sqrt(v_cap) + self.eps)

    # Adagrad optimization function
    def Adagrad(self):
        if self.t < 1: print('using Adagrad')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # update memory
                self.m[l][p] += self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                # Update parameters
                self.model[l].parameters[p].w -= self.lr * self.model[l].parameters[p].dw / np.sqrt(self.m[l][p] + self.eps)

    # Adadelta optimization function
    def Adadelta(self):
        # useful: https://arxiv.org/pdf/1212.5701.pdf
        if self.t < 1: print('using Adadelta')

        self.t += 1
        for l in range(len(self.model)):
            for p in range(len(self.model[l].parameters)):
                # clip gradients
                self.model[l].parameters[p].dw = np.clip(self.model[l].parameters[p].dw, -5.0, 5.0)

                # Accumulate Gradient:
                self.m[l][p] = self.ro * self.m[l][p] + (1 - self.ro) * self.model[l].parameters[p].dw * self.model[l].parameters[p].dw

                # Compute Update:
                delta = -np.sqrt((self.v[l][p] + self.eps) / (self.m[l][p] + self.eps)) * self.model[l].parameters[p].dw

                # Accumulate Updates:
                self.v[l][p] = self.ro * self.v[l][p] + (1 - self.ro) * delta * delta

                # Apply Update:
                self.model[l].parameters[p].dw += delta

    #define optimizers
