"""
AI library in python using numpy

Author: Sri Ram Bandi (srirambandi.654@gmail.com)
        https://www.github.com/srirambandi

MIT License
"""


import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G

# module and its children including loss
from ai.module import Module
from ai.linear import Linear
from ai.convolutional import Conv1d, Conv2d, ConvTranspose2d
from ai.sequential import RNN, LSTM
from ai.normalization import BatchNorm2d, LayerNorm
from ai.pooling import Maxpool2d
from ai.regularization import Dropout
from ai.embedding import Embedding
from ai.loss import MSELoss, CrossEntropyLoss, BCELoss, JSDivLoss, TestLoss
from ai.activation import ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax

# beloved optimizers
from ai.optimizer import SGD, Adam, Adagrad, Adadelta, RMSprop

from ai.utils import draw_graph, clip_grad_value


# initializations and utitlity functions
def manual_seed(seed=2357):
    np.random.seed(seed)
