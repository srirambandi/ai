"""
AI library in python using numpy

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://srirambandi.github.io

MIT License
"""


import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G

from ai.linear import Linear
from ai.convolutional import Conv2d, ConvTranspose2d
from ai.sequence_models import RNN, LSTM
from ai.batch_norm import BatchNorm

from ai.loss import Loss
from ai.optimizer import Optimizer
from ai.module import Module


# initializations and utitlity functions
def manual_seed(seed=2357):
    np.random.seed(seed)

def draw_graph(filename='graph', format='svg', graph=G):
    # visualization procedure referred from karpathy's micrograd

    from graphviz import Digraph

    label = 'Computational Graph of {}'.format(filename)
    dot = Digraph(filename=filename, directory='assets',
            format=format, graph_attr={'rankdir': 'LR', 'label': label}, node_attr={'rankdir': 'TB'})

    for cell in graph.nodes:

        # add the op to nodes
        dot.node(name=str(id(cell['backprop_op'])), label=cell['func'], shape='doublecircle',)

        for input in cell['inputs']:

            # add the input to nodes
            color = None if input.eval_grad else 'red'
            dot.node(name=str(id(input)), label='{}'.format(input.node_id), shape='circle', color=color)
            # forward pass edge from input to op
            dot.edge(str(id(input)), str(id(cell['backprop_op'])))

            # # backprop pass edge from op to input
            # if input.eval_grad:
            #     dot.edge(str(id(cell['backprop_op'])), str(id(input)), color='red')

        for output in cell['outputs']:

            # add the output to nodes
            dot.node(name=str(id(output)), label='{}'.format(output.node_id), shape='circle')
            # forward pass edge from op to output
            dot.edge(str(id(cell['backprop_op'])), str(id(output)))

            # # backward pass edge from output to op
            # dot.edge(str(id(output)), str(id(cell['backprop_op'])), color='red')


    dot.render(cleanup=True)
