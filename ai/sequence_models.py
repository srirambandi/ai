import numpy as np
from .parameter import Parameter
from .graph import ComputationalGraph, G


# sequence models: LSTM cell
class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size    # size of the input at each recurrent tick
        self.hidden_size = hidden_size  # size of hidden units h and c
        self.init_params()

    def init_params(self):
        self.W_ih = Parameter((4*self.hidden_size, self.input_size))    # input to hidden weight volume
        self.W_hh = Parameter((4*self.hidden_size, self.hidden_size))   # hidden to hidden weight volume
        self.b_ih = Parameter((4*self.hidden_size, 1))  # input to hidden bias vector
        self.b_hh = Parameter((4*self.hidden_size, 1))  # hidden to hidden bias vector
        self.parameters = [self.W_ih, self.b_ih, self.W_hh, self.b_hh]

    def forward(self, x, hidden):

        h, c = hidden

        if type(x) is not Parameter:
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=False, init_zeros=True)
            x.w = _


        gates = G.add(G.add(G.dot(self.W_hh, h), self.b_hh), G.add(G.dot(self.W_ih, x), self.b_ih))

        # forget, input, gate(also called cell gate - different from cell state), output gates of the lstm cell
        # useful: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        f, i, g, o = G.split(gates, sections=4, axis=0)

        f = G.sigmoid(f)
        i = G.sigmoid(i)
        g = G.tanh(g)
        o = G.sigmoid(o)

        c = G.add(G.multiply(f, c), G.multiply(i, g))
        h = G.multiply(o, G.tanh(c))

        return (h, c)


# sequence models: RNN cell
class RNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_params()

    def init_params(self):
        self.W_ih = Parameter((self.hidden_size, self.input_size))
        self.W_hh = Parameter((self.hidden_size, self.hidden_size))
        # self.b_ih = Parameter((self.hidden_size, 1), init_zeros=True)    # not much use
        self.b_hh = Parameter((self.hidden_size, 1), init_zeros=True)
        self.parameters = [self.W_ih, self.W_hh, self.b_hh]

    def forward(self, x, hidden):

        h = hidden

        if type(x) is not Parameter:
            shape = x.shape
            _ = x
            x = Parameter(shape, eval_grad=False, init_zeros=True)
            x.w = _

        h = G.add(G.add(G.dot(self.W_hh, h), G.dot(self.W_ih, x)), self.b_hh)
        h = G.tanh(h)

        return h
