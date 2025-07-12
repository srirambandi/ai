import numpy as np
import ai.graph


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), data=None, grad=None, requires_grad=True, node_id=None, graph=None,
                init_zeros=False, init_ones=False, constant=1.0,
                uniform=False, low=-1.0, high = 1.0,
                normal=False, mean=0.0, std=0.01):

        # properties
        self.shape = shape
        self.data = data
        self.grad = grad
        self.requires_grad = requires_grad  # if the parameter is a variable or an input/constant

        # node id - in the bfs like graph walk during forward pass, the node number
        # of the path ie., the latest backward op of which this parameter was an output
        self.node_id = node_id

        if graph is not None:   # graph object this parameter belongs to
            self.graph = graph
        else:
            self.graph = ai.graph.G

        # constant initializations
        self.init_zeros = init_zeros
        self.init_ones = init_ones
        self.constant = constant

        # initializing from distributions
        self.uniform = uniform
        self.low = low      # high and low of uniform
        self.high = high    # distribution to initialize the parameter
        self.mean = mean    # mean and variance of the gaussian
        self.std = std      # distribution to initialize the parameter

        # creating weight and gradient tensors
        self.init_params()

    def init_params(self):

        if self.data is not None:
            # initiating weights with passed data object of kind list/numpy-ndarray
            if not isinstance(self.data, np.ndarray):
                self.data = np.array(self.data)
            self.shape = self.data.shape   # resolving conflict with passed shape and data shape

        elif self.init_zeros:
            # initiating with zeros of given shape
            self.data = np.zeros(self.shape)

        elif self.init_ones:
            # initiating with ones(or a constant) of given shape
            self.data = np.ones(self.shape) * self.constant

        elif self.uniform:
            # random initiation with uniform distribution
            self.data = np.random.uniform(self.low, self.high, self.shape)

        else:
            # random initiation with gaussian distribution
            self.normal = True
            self.data = np.random.normal(self.mean, self.std, self.shape)

        # setting gradient of parameter wrt some scalar, as zeros
        if self.grad is None:
            self.grad = np.zeros(self.shape)
        else:
            if not isinstance(self.grad, np.ndarray):
                self.grad = np.array(self.grad)
            assert self.data.shape == self.grad.shape, 'data and grad should be of same shape'

    def __repr__(self):
        parameter_schema = f'Parameter(shape={self.shape}, requires_grad={self.requires_grad}) containing:\n'
        parameter_schema += f'Data: {self.data}'

        return parameter_schema

    # this function computes the gradients of the parameters, by executing
    # the backprop ops in reverse order to the forward propagation with chain rule
    def backward(self, grad=None, to=None):
        # assign gradient

        if self.node_id is None:
            return

        if grad is not None:
            if not isinstance(grad, np.ndarray):
                self.grad = np.array(grad)

        if to is None:
            to_node_id = 0    # execute backward all the way to start
        else:
            to_node_id = to.node_id + 1  # execute backward  to just before this node

        for node in reversed(self.graph.nodes[to_node_id:self.node_id + 1]):
            node['backprop_op']()       # executing the back-propagation operation

    def __getitem__(self, key):

        scalar_indexing = all([isinstance(i, int) for i in key])
        if scalar_indexing:
            raise Exception('Cannot do scalar indexing on the Paramter. Use x.data for scalar indexing or use slices.')

        new_key = tuple([slice(i, i + 1) if isinstance(i, int) else i for i in key])

        return self.graph.getitem(self, new_key)

    def __add__(self, other):

        if not isinstance(other, Parameter):
            if isinstance(other, int) or isinstance(other, float):
                constant = np.empty(self.shape)
                constant.fill(float(other))
                other = constant
            other = Parameter(data=other, requires_grad=False, graph=self.graph)

        assert self.shape == other.shape, (f'Objects not of same shape: {self.shape} and {other.shape}. Use G.add() with axis argument.')

        return self.graph.add(self, other)

    def __sub__(self, other):

        if not isinstance(other, Parameter):
            if isinstance(other, int) or isinstance(other, float):
                constant = np.empty(self.shape)
                constant.fill(float(other))
                other = constant
            other = Parameter(data=other, requires_grad=False, graph=self.graph)

        assert self.shape == other.shape, (f'Objects not of same shape: {self.shape} and {other.shape}. Use G.subtract() with axis argument.')

        return self.graph.subtract(self, other)

    def __mul__(self, other):

        if not isinstance(other, Parameter):
            if isinstance(other, int) or isinstance(other, float):
                constant = np.empty(self.shape)
                constant.fill(float(other))
                other = constant
            other = Parameter(data=other, requires_grad=False, graph=self.graph)

        assert self.shape == other.shape, (f'Objects not of same shape: {self.shape} and {other.shape}. Use G.multiply() with axis argument.')

        return self.graph.multiply(self, other)

    def __matmul__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, requires_grad=False, graph=self.graph)

        return self.graph.matmul(self, other)

    def __truediv__(self, other):

        if not isinstance(other, Parameter):
            if isinstance(other, int) or isinstance(other, float):
                constant = np.empty(self.shape)
                constant.fill(float(other))
                other = constant
            other = Parameter(data=other, requires_grad=False, graph=self.graph)

        assert self.shape == other.shape, (f'Objects not of same shape: {self.shape} and {other.shape}. Use G.divide() with axis argument.')

        return self.graph.divide(self, other)

    def __pow__(self, other):
        return self.graph.power(self, other)

    def transpose(self, axis0=None, axis1=None):
        return self.graph.transpose(self, axis0=axis0, axis1=axis1)

    def reshape(self, shape):
        return self.graph.reshape(self, shape)

    def split(self, sections=1, axis=-1):
        return self.graph.split(self, sections=sections, axis=axis)

    # transpose
    @property
    def T(self):

        data = np.ascontiguousarray(self.data.T)
        grad = np.ascontiguousarray(self.grad.T)
        shape = tuple(reversed(self.shape))

        return Parameter(shape=shape, data=data, grad=grad, requires_grad=self.requires_grad, graph=self.graph)

    # shape
    @property
    def shape(self):
        assert self.shape == self.data.shape and self.data.shape == self.grad.shape, f'Something is wrong with the Parameter, \
            shape={self.shape}, data.shape={self.data.shape}, grad.shape={self.grad.shape}'

        return self.shape

    # number of dimensions
    @property
    def ndim(self):
        assert self.shape == self.data.shape and self.data.shape == self.grad.shape, f'Something is wrong with the Parameter, \
            shape={self.shape}, data.shape={self.data.shape}, grad.shape={self.grad.shape}'

        return self.data.ndim

    @property
    def data(self):
        return self.data

    @data.setter
    def data(self, data):
        assert data is not None, "can't assign None to data"
        assert isinstance(data, np.ndarray), f"can't assign data of type {type(data)}."
        assert data.shapa == self.shape, f"can't assign data of shape {data.shape} to Parameter of shape {self.shape}"

        self.data = data

    @property
    def grad(self):
        return self.grad

    @grad.setter
    def grad(self, grad):
        assert grad is not None, "can't assign None to grad"
        assert isinstance(grad, np.ndarray), f"can't assign grad of type {type(grad)}."
        assert grad.shapa == self.shape, f"can't assign grad of shape {grad.shape} to Parameter of shape {self.shape}"

        self.grad = grad

    @property
    def requires_grad(self):
        return self.requires_grad
