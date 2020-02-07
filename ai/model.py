import numpy as np
from .parameter import Parameter
from .graph import ComputationalGraph, G


# model class to add useful features like save, load model from files
class Model:
    def __init__(self):
        self.layers = []

    def save(self, file=None):  # model.save() - saves the state of the network
        print('saving model...')
        layers = []

        for layer in self.layers:
            parameters = []
            for parameter in layer.parameters:
                parameters.append(parameter.w)
            layers.append(parameters)

        if file == None:
            file = str(self.__class__).strip('<>').split()[1].strip("\'").split('.')[1]

        np.save(file+'.npy', layers)

        print('model saved in', file)

    def load(self, file=None):  # model.load() - loads the state of net from a file
        print('loading model from', file)
        if file == None:
            file = str(self.__class__).strip('<>').split()[1].strip("\'").split('.')[1]+'.npy'

        layers = np.load(file, allow_pickle=True)

        for layer_act, layer in zip(self.layers, layers):
            for parameter_act, parameter in zip(layer_act.parameters, layer):
                parameter_act.w = parameter

        print('model loaded!')

    def get_parameters(self):   # access parameters of the model with this func
        return self.layers


# TODO: define regularizations, asserts, batch, utils, GPU support, examples
