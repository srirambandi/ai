import ai
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_file = 'CIFAR10/data_batch_'
test_file = 'CIFAR10/test_batch'

class CIFAR10(ai.Model):
    def __init__(self):
        self.conv1 = ai.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ai.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = ai.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = ai.Conv2d(64, 64, kernel_size=3)
        self.fc1 = ai.Linear(2304, 512)
        self.fc2 = ai.Linear(512, 10)

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

cifar10 = CIFAR10()
print(cifar10)

L = ai.Loss(loss_fn='CrossEntropyLoss')
optim = ai.Optimizer(cifar10.parameters(), optim_fn='Adam', lr=1e-3)


it, epoch = 0, 0
m = 8


def evaluate():
    ai.G.grad_mode = False

    file = test_file
    dict = unpickle(file)

    inputs = dict[b'data']
    outputs = dict[b'labels']

    correct, total = 0, 0

    test_m = m

    for batch in range(int(len(outputs) / m)):

        input = inputs[batch * test_m : (batch + 1) * test_m].reshape(test_m, 3, 32, 32) / 255
        input =  np.stack([_ for _ in input], axis = -1)
        output = np.array(outputs[batch * test_m : (batch + 1) * test_m])

        scores = cifar10.forward(input)
        preds = np.argmax(scores.data, axis=0)

        correct += np.sum(np.equal(output, preds))
        total += test_m

    accuracy = float(correct / total)

    ai.G.grad_mode = True

    return accuracy


while epoch < 15:
    epoch += 1
    it = 0
    for set in range(1, 6):
        print('Set #{} started.'.format(set))

        dataset = train_file + str(set)
        dict = unpickle(dataset)

        inputs = dict[b'data']
        outputs = dict[b'labels']

        for batch in range(int(len(outputs) / m)):

            input = inputs[batch * m : (batch + 1) * m].reshape(m, 3, 32, 32) / 255
            input =  np.stack([_ for _ in input], axis = -1)
            output = outputs[batch * m : (batch + 1) * m]
            onehot = np.zeros((10, m))
            for _ in range(m):
                onehot[output[_], _] = 1.0

            scores = cifar10.forward(input)

            loss = L.loss(scores, onehot)
            loss.backward()

            optim.step()        # update parameters with optimization functions
            optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

            if it%1 == 0:
                print('epoch: {}, iter: {}, loss: {}'.format(epoch, it, loss.data[0][0]))

            it += 1

    print('\n\n', 'Epoch {} completed. Accuracy {}'.format(epoch, evaluate()))


cifar10.save()