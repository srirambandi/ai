import ai_full as ai
import numpy as np

def load_data(file):
    dict = np.load(file, allow_pickle=True)
    return dict

train_file = 'mnist/train.npy'
test_file = 'mnist/test.npy'

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
        o5 = ai.G.softmax(self.fc2.forward(o4))

        return o5

model = mnist()
L = ai.Loss(loss_fn='CrossEntropyLoss')
optim = ai.Optimizer(model.layers, optim_fn='Adam', lr=1e-3)


it = 0
loss = np.inf
m = 2

train_dict = load_data(train_file)
inputs = train_dict.item()['data']
outputs = train_dict.item()['labels']

del train_dict

loss = np.inf
while loss > 0.05:

    # for batch in range(int(len(outputs) / m)):
    for batch in range(1):

        input = inputs[batch * m : (batch + 1) * m].reshape(1, 28, 28, m) / 255
        output = outputs[batch * m : (batch + 1) * m]
        onehot = np.zeros((10, m))
        for _ in range(m):
            onehot[output[_], _] = 1.0

        scores = model.forward(input)

        loss = L.loss(scores, onehot).w[0][0]
        L.backward()

        optim.step()        # update parameters with optimization functions
        optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

        if it%1 == 0:
            print('iter: {}, loss: {}'.format(it, loss))

        it += 1

model.save()
