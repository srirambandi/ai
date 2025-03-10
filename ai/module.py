import numpy as np
from abc import ABC, abstractmethod
from ai.parameter import Parameter
from ai.graph import ComputationalGraph, G


# generic module class to add useful features like save/load model from files, get parameters etc.
class Module(ABC):
    def __init__(self):
        pass

    def __repr__(self):
        module_schema = str(self.__class__.__name__) + '(\n'

        for name, layer in self.get_module_layers().items():
            module_schema += '  ' + str(name) + ': ' + str(layer) + '\n'

        module_schema += ')'

        return module_schema

    def __call__(self, *args, **kwargs):  # easy callable
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwards):
        raise NotImplementedError

    def save(self, file=None):  # model.save() - saves the state of the network
        print('saving model...')
        save_dict = dict()

        module_layers = self.get_module_layers()
        for layer_name, layer in module_layers.items():

            layer_params = layer.get_module_params()
            for param_name, param in layer_params.items():
                layer_params[param_name] = param.data

            module_layers[layer_name] = layer_params

        module_params = self.get_module_params()
        for param_name, param in module_params.items():
            module_params[param_name] = param.data

        save_dict['module_layers'] = module_layers
        save_dict['module_params'] = module_params

        if file == None:
            file = self.__class__.__name__+'.npy'

        np.save(file, save_dict)
        print('Successfully saved model in {}'.format(file))

    def load(self, file=None):  # model.load() - loads the state of net from a file
        print('loading model...')

        if file == None:
            file = self.__class__.__name__+'.npy'

        load_dict = np.load(file, allow_pickle=True).item()
        module_layers_stored = load_dict['module_layers']
        module_params_stored = load_dict['module_params']

        module_layers_actual = self.get_module_layers()
        module_params_actual = self.get_module_params()

        for layer_name, layer_stored in module_layers_stored.items():
            if layer_name in module_layers_actual:
                for param_name, param in layer_stored.items():
                    layer_actual = module_layers_actual[layer_name]
                    setattr(layer_actual, str(param_name), Parameter(data=param))

        for param_name, param in module_params_stored.items():
            if param_name in module_params_actual:
                setattr(self,str(param_name), Parameter(data=param))

        print('Successfully loaded model from {}'.format(file))

    def get_module_layers(self):   # returns a dictionary of parametrized layers in the module

        attributes = self.__dict__
        layers = ['Linear', 'Conv1d', 'Conv2d', 'ConvTranspose2d', 'LSTM', 'RNN', 'BatchNorm', 'Maxpool2d', 'Dropout']

        module_layers = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in layers:
                module_layers[name] = attributes[name]

        return module_layers

    def get_module_params(self):    # returns a dictionary of parameters in the module

        attributes = self.__dict__

        module_params = dict()
        for name in attributes:
            if attributes[name].__class__.__name__ in ['Parameter']:
                if attributes[name].requires_grad:
                    module_params[name] = attributes[name]

        return module_params

    def parameters(self):   # access parameters of the module with this function

        all_params = list()

        for layer in list(self.get_module_layers().values()):
            all_params.extend(layer.parameters())

        all_params.extend(list(self.get_module_params().values()))

        return all_params
