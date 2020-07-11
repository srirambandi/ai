import numpy as np
import ai.graph


# the Parameter object: stores weights and derivatives of weights(after backprop)
# of each layer in the model
class Parameter:
    def __init__(self, shape=(0, 0), data=None, eval_grad=True, node_id=None, graph=None,
                init_zeros=False, init_ones=False, constant=1.0,
                uniform=False, low=-1.0, high = 1.0,
                normal=False, mean=0.0, std=0.01):

        # properties
        self.shape = shape
        self.data = data
        self.eval_grad = eval_grad  # if the parameter is a variable or an input/constant

        # node id - in the bfs like graph walk during forward pass, the node numeber
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
        self.grad = np.zeros(self.shape)

    def __str__(self):
        parameter_schema = 'Parameter(shape={}, eval_grad={}) containing:\n'.format(self.shape, self.eval_grad)
        parameter_schema += 'Data: {}'.format(self.data)

        return parameter_schema

    # this function computes the gradients of the parameters, by executing
    # the backprop ops in reverse order to the forward propagation with chain rule
    def backward(self, grad=None, to=None):
        # assign gradient

        if self.node_id is None:
            return

        if grad is not None:
            self.grad = np.array(grad)

        if to is None:
            to_node_id = 0    # execute backward all the way to start
        else:
            to_node_id = to.node_id + 1  # execute backward  to just before this node

        for node in reversed(self.graph.nodes[to_node_id:int(self.node_id) + 1]):
            node['backprop_op']()       # executing the back-propagation operation

    def __add__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, ('Objects not of same shape. Use G.add() with axis argument', self.shape, other.shape)

        return self.graph.add(self, other)

    def __sub__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, ('Objects not of same shape. Use G.subtract() with axis argument', self.shape, other.shape)

        return self.graph.subtract(self, other)

    def __mul__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, ('Objects not of same shape. Use G.multiply() with axis argument', self.shape, other.shape)

        return self.graph.multiply(self, other)

    def __matmul__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        return self.graph.dot(self, other)

    def __truediv__(self, other):

        if not isinstance(other, Parameter):
            other = Parameter(data=other, eval_grad=False, graph=self.graph)

        assert self.shape == other.shape, 'Objects not of same shape. Use G.divide() with axis argument'

        return self.graph.divide(self, other)

    def __pow__(self, other):
        return self.graph.power(self, other)

    # transpose
    def T(self):

        self.data = self.data.T
        self.grad = self.grad.T
        self.shape = tuple(reversed(self.shape))

        return self
