import numpy as np
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


# model class to add useful features like save/load model from files, get parameters etc.
class Model:
    def __init__(self):
        pass

    def __str__(self):
        model_schema = str(self.__class__.__name__) + '(\n'

        for name, layer in self.layers().items():
            model_schema += '  ' + str(name) + ': ' + str(layer) + '\n'

        model_schema += ')'

        return model_schema

    def save(self, file=None):  # model.save() - saves the state of the network
        print('saving model...')
        layers_data = dict()

        for name, layer in self.layers().items():
            parameter_list = []
            for parameter in layer.parameters:
                parameter_list.append(parameter.data)

            layers_data[name] = parameter_list

        if file == None:
            file = self.__class__.__name__+'.npy'

        np.save(file, layers_data)
        return('Successfully saved model in {}'.format(file))

    def load(self, file=None):  # model.load() - loads the state of net from a file
        print('loading model from')

        if file == None:
            file = self.__class__.__name__+'.npy'

        layers_data = np.load(file, allow_pickle=True).items()
        model_layers = self.layers()

        for name, data_list in layers_data:
            for parameter_act, data_stored in zip(model_layers[name].parameters, data_list):
                parameter_act.data = data_stored

        return('Successfully loaded model in {}'.format(file))

    def layers(self):   # returns a dictionary of parametrized layers
        attributes = self.__dict__
        parametrized_layers = ['Linear', 'Conv2d', 'ConvTranspose2d', 'LSTM', 'RNN', 'BatchNorm']

        layers = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in parametrized_layers:
                layers[name] = attributes[name]

        return layers

    def parameters(self):   # access parameters of the model with this function
        parameters = list()

        for layer in list(self.layers().values()):
            for parameter in layer.parameters:
                parameters.append(parameter)

        return parameters
