import ai_full as ai
import numpy as np

def load_data(file):
    dict = np.load(file, allow_pickle=True)
    return dict

train_file = 'mnist/train.npy'
test_file = 'mnist/test.npy'

bag = []

class mnist(ai.Model):
    def __init__(self, ):
        self.conv1 = ai.Conv2d(1, 8, kernel_size=3, stride=1)
        self.conv2 = ai.Conv2d(8, 16, kernel_size=3, stride=1)
        self.fc1 = ai.Linear(2304, 128)
        self.fc2 = ai.Linear(128, 10)
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):
        o1 = ai.G.relu(self.conv1.forward(x))
        o2 = ai.G.relu(self.conv2.forward(o1))
        o3 = ai.G.dropout(ai.G.maxpool2d(o2), p=0.75)
        o4 = ai.G.dropout(ai.G.relu(self.fc1.forward(o3)), p=0.5)
        o5 = self.fc2.forward(o4)
        o6 = ai.G.softmax(o5)

        bag.append(o5)

        return o6

model = mnist()
L = ai.Loss(loss_fn='CrossEntropyLoss')
optim = ai.Optimizer(model.layers, optim_fn='Adadelta', lr=1e-3)


train_dict = load_data(train_file)
inputs = train_dict.item()['data']
outputs = train_dict.item()['labels']

del train_dict

it, epoch = 0, 0
loss = np.inf
m = 8


def evaluate():
    ai.G.grad_mode = False

    file = test_file
    dict = load_data(file)

    inputs = dict.item()['data']
    outputs = dict.item()['labels']

    correct, total = 0, 0

    test_m = m

    for batch in range(int(len(outputs) / m)):

        input = inputs[batch * test_m : (batch + 1) * test_m].reshape(test_m, 1, 28, 28) / 255
        input =  np.stack([_ for _ in input], axis = -1)
        output = np.array(outputs[batch * test_m : (batch + 1) * test_m])

        scores = model.forward(input)
        preds = np.argmax(scores.data, axis=0)

        correct += np.sum(np.equal(output, preds))
        total += test_m

    accuracy = float(correct / total)

    ai.G.grad_mode = True

    return accuracy


while loss > 0.1:
    epoch += 1
    it = 0
    for batch in range(int(len(outputs) / m)):
    # for batch in range(1):

        input = inputs[batch * m : (batch + 1) * m].reshape(m, 1, 28, 28) / 255
        input =  np.stack([_ for _ in input], axis = -1)
        output = outputs[batch * m : (batch + 1) * m]
        onehot = np.zeros((10, m))
        for _ in range(m):
            onehot[output[_], _] = 1.0

        scores = model.forward(input)

        loss = L.loss(scores, onehot).data[0][0]
        loss.backward()

        optim.step()        # update parameters with optimization functions
        optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

        if it%10 == 0:
            print('epoch: {}, iter: {}, loss: {}'.format(epoch, it, loss))

        it += 1

    print('\n\n', 'Epoch {} completed. Accuracy: {}'.format(epoch, evaluate()))


model.save()
