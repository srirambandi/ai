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
from ai.model import Model


# initializations and utitlity functions
def manual_seed(seed=2357):
    np.random.seed(seed)

manual_seed()
