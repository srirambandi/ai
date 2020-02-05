import ai
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file = 'cifar10/data_batch_'

class cifar10(ai.Model):
    def __init__(self, ):
        self.conv1 = ai.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ai.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = ai.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = ai.Conv2d(64, 64, kernel_size=3)
        self.fc1 = ai.Linear(2304, 512)
        self.fc2 = ai.Linear(512, 10)
        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2]

    def forward(self, x):
        o1 = ai.G.relu(self.conv1.forward(x))
        o2 = ai.G.relu(self.conv2.forward(o1))
        o3 = ai.G.dropout(ai.G.maxpool2d(o2), p=0.75)
        o4 = ai.G.relu(self.conv3.forward(o3))
        o5 = ai.G.relu(self.conv4.forward(o4))
        o6 = ai.G.dropout(ai.G.maxpool2d(o5), p=0.75)
        o7 = ai.G.dropout(ai.G.relu(self.fc1.forward(o6)), p=0.5)
        o8 = ai.G.softmax(self.fc2.forward(o7))

        return o8

model = cifar10()
L = ai.Loss(loss_fn='CrossEntropyLoss')
optim = ai.Optimizer(model.layers, optim_fn='Adam', lr=1e-3)


it = 0
loss = np.inf
while loss > 1:

    for set in range(1, 7):
        dataset = file + str(set)
        dict = unpickle(dataset)

        inputs = dict[b'data']
        outputs = dict[b'labels']

        for input, output in zip(inputs, outputs):

            input = input.reshape(3, 32, 32)
            onehot = np.zeros((10, 1))
            onehot[output] = 1

            scores = model.forward(input)

            loss = L.loss([scores], [onehot])[0].w[0][0]
            L.backward()

            optim.step()        # update parameters with optimization functions
            optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

            if it%1 == 0:
                print('iter: {}, loss: {}'.format(it, loss))

            it += 1
