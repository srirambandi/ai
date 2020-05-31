import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


# sequence models: LSTM cell
class LSTM:
    def __init__(self, input_size, hidden_size, bias=True, graph=G):
        self.input_size = input_size    # size of the input at each recurrent tick
        self.hidden_size = hidden_size  # size of hidden units h and c
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.W_ih = Parameter((4*self.hidden_size, self.input_size), graph=self.graph)    # input to hidden weight volume
        self.W_hh = Parameter((4*self.hidden_size, self.hidden_size), graph=self.graph)   # hidden to hidden weight volume
        self.b_ih = Parameter((4*self.hidden_size, 1), graph=self.graph)  # input to hidden bias vector
        self.b_hh = Parameter((4*self.hidden_size, 1), graph=self.graph)  # hidden to hidden bias vector
        self.parameters = [self.W_ih, self.b_ih, self.W_hh, self.b_hh]

    def __str__(self):
        return('LSTM(input_size={}, hidden_size={}, bias={})'.format(
            self.input_size, self.hidden_size, self.bias))

    def __call__(self, x, hidden):  # easy callable
        return self.forward(x, hidden)

    def forward(self, x, hidden):

        h, c = hidden

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False, graph=self.graph)


        h_h = self.graph.dot(self.W_hh, h)
        if self.bias:
            h_h = self.graph.add(h_h, self.b_hh, axis=(-1,))

        i_h = self.graph.dot(self.W_ih, x)
        if self.bias:
            i_h = self.graph.add(i_h, self.b_ih, axis=(-1,))

        gates = self.graph.add(h_h, i_h)

        # forget, input, gate(also called cell gate - different from cell state), output gates of the lstm cell
        # useful: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        f, i, g, o = self.graph.split(gates, sections=4, axis=0)

        f = self.graph.sigmoid(f)
        i = self.graph.sigmoid(i)
        g = self.graph.tanh(g)
        o = self.graph.sigmoid(o)

        c = self.graph.add(self.graph.multiply(f, c), self.graph.multiply(i, g))
        h = self.graph.multiply(o, self.graph.tanh(c))

        return (h, c)


# sequence models: RNN cell
class RNN:
    def __init__(self, input_size, hidden_size, bias=True, graph=G):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        self.W_ih = Parameter((self.hidden_size, self.input_size), graph=self.graph)
        self.W_hh = Parameter((self.hidden_size, self.hidden_size), graph=self.graph)
        self.b_ih = Parameter((self.hidden_size, 1), graph=self.graph)    # not much use
        self.b_hh = Parameter((self.hidden_size, 1), graph=self.graph)
        self.parameters = [self.W_ih, self.W_hh, self.b_hh]

    def __str__(self):
        return('RNN(input_size={}, hidden_size={}, bias={})'.format(
            self.input_size, self.hidden_size, self.bias))

    def __call__(self, x, hidden):  # easy callable
        return self.forward(x, hidden)

    def forward(self, x, hidden):

        h = hidden

        if type(x) is not Parameter:
            x = Parameter(data=x, eval_grad=False, graph=self.graph)

        h_h = self.graph.dot(self.W_hh, h)
        if self.bias:
            h_h = self.graph.add(h_h, self.b_hh, axis=(-1,))

        i_h = self.graph.dot(self.W_ih, x)
        if self.bias:
            i_h = self.graph.add(i_h, self.b_ih, axis=(-1,))

        h = self.graph.add(h_h, i_h)

        h = self.graph.tanh(h)

        return h
