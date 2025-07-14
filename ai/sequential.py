import numpy as np
from ai.parameter import Parameter
from ai.graph import G
from ai.module import Module


# sequence models: LSTM cell
class LSTM(Module):
    def __init__(self, input_size, hidden_size, bias=True, graph=G):
        super().__init__()
        self.input_size = input_size    # size of the input at each recurrent tick
        self.hidden_size = hidden_size  # size of hidden units h and c
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.hidden_size)
        self.W_ih = Parameter((4*self.hidden_size, self.input_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)    # input to hidden weight volume
        self.W_hh = Parameter((4*self.hidden_size, self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)   # hidden to hidden weight volume
        self.b_ih = Parameter((1, 4*self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # input to hidden bias vector
        self.b_hh = Parameter((1, 4*self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)  # hidden to hidden bias vector

    def __repr__(self):
        return(f'LSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, bias={self.bias})')

    def forward(self, x, hidden):

        h, c = hidden

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        i_h = x @ self.W_ih.transpose()
        if self.bias:
            i_h = i_h + self.b_ih

        h_h = h @ self.W_hh.transpose()
        if self.bias:
            h_h = h_h + self.b_hh

        gates = i_h + h_h

        # forget, input, gate(also called cell gate - different from cell state), output gates of the lstm cell
        # useful: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        f, i, g, o = self.graph.split(gates, sections=4, axis=1)

        f = self.graph.sigmoid(f)
        i = self.graph.sigmoid(i)
        g = self.graph.tanh(g)
        o = self.graph.sigmoid(o)

        c = (f * c) + (i * g)
        h = o * self.graph.tanh(c)

        return (h, c)


# sequence models: RNN cell
class RNN(Module):
    def __init__(self, input_size, hidden_size, bias=True, graph=G):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.graph = graph
        self.init_params()

    def init_params(self):
        root_k = np.sqrt(1. / self.hidden_size)
        self.W_ih = Parameter((self.hidden_size, self.input_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.W_hh = Parameter((self.hidden_size, self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)
        self.b_ih = Parameter((1, self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)    # not much use
        self.b_hh = Parameter((1, self.hidden_size), uniform=True, low=-root_k, high=root_k, graph=self.graph)

    def __repr__(self):
        return(f'RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, bias={self.bias})')

    def forward(self, x, hidden):

        h = hidden

        if not isinstance(x, Parameter):
            x = Parameter(data=x, requires_grad=False, graph=self.graph)

        i_h = x @ self.W_ih.transpose()
        if self.bias:
            i_h = i_h + self.b_ih

        h_h = h @ self.W_hh.transpose()
        if self.bias:
            h_h = h_h + self.b_hh

        h = i_h + h_h

        h = self.graph.tanh(h)

        return h
