import ai
import numpy as np

def load_data(file):
    dict = np.load(file, allow_pickle=True)
    return dict

train_file = 'MNIST/train.npy'
test_file = 'MNIST/test.npy'


class MLP(ai.Model):
    def __init__(self, ):
        self.fc1 = ai.Linear(784, 200)
        self.fc2 = ai.Linear(200, 10)

    def forward(self, x):
        o1 = ai.G.dropout(ai.G.relu(self.fc1.forward(x)), p=0.75)
        o2 = ai.G.softmax(self.fc2.forward(o1))


        return o2

mlp = MLP()
print(mlp)

L = ai.Loss(loss_fn='CrossEntropyLoss')
optim = ai.Optimizer(mlp.parameters(), optim_fn='Adam', lr=1e-3)


train_dict = load_data(train_file)
inputs = train_dict.item()['data']
outputs = train_dict.item()['labels']

del train_dict

it, epoch = 0, 0
m = 32


def evaluate():
    ai.G.grad_mode = False

    file = test_file
    dict = load_data(file)

    inputs = dict.item()['data']
    outputs = dict.item()['labels']

    correct, total = 0, 0

    test_m = m

    for batch in range(int(len(outputs) / m)):

        input = inputs[batch * test_m : (batch + 1) * test_m] / 255
        input =  np.stack([_ for _ in input], axis = -1)
        output = np.array(outputs[batch * test_m : (batch + 1) * test_m])

        scores = mlp.forward(input)
        preds = np.argmax(scores.data, axis=0)

        correct += np.sum(np.equal(output, preds))
        total += test_m

    accuracy = float(correct / total)

    ai.G.grad_mode = True

    return accuracy


while epoch < 10:
    epoch += 1
    it = 0
    for batch in range(int(len(outputs) / m)):
    # for batch in range(1):

        input = inputs[batch * m : (batch + 1) * m] / 255
        input =  np.stack([_ for _ in input], axis = -1)
        output = outputs[batch * m : (batch + 1) * m]
        onehot = np.zeros((10, m))
        for _ in range(m):
            onehot[output[_], _] = 1.0

        scores = mlp.forward(input)

        loss = L.loss(scores, onehot)
        loss.backward()

        optim.step()        # update parameters with optimization functions
        optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

        if it%10 == 0:
            print('epoch: {}, iter: {}, loss: {}'.format(epoch, it, loss.data[0, 0]))

        it += 1

    print('\n\n', 'Epoch {} completed. Accuracy: {}'.format(epoch, evaluate()))


mlp.save()
