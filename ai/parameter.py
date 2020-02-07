import numpy as np


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), eval_grad=True, init_zeros=False, mu = 0.0, std = 0.01):
        self.eval_grad = eval_grad  # if the parameter is a variable or an input/scalar
        self.shape = shape
        self.init_zeros = init_zeros
        self.w = np.zeros(shape)
        self.dw = np.zeros(shape)
        self.mu = mu    # mean and variance of the gaussian
        self.std = std  # distribution to initialize the parameter
        self.init_params()

    def init_params(self):
        if not self.init_zeros:
            self.w = self.std*np.random.randn(*self.shape) + self.mu
        return self.w

    # transpose
    def T(self):
        self.w = self.w.T
        self.dw = self.dw.T
        self.shape = tuple(reversed(self.shape))

        return self
