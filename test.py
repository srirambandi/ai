import ai_full as ai
import numpy as np

# I/O preparation
X = np.stack([np.zeros((3, 6, 6,)), np.random.randint(0, 6, (3, 6, 6))], axis = -1)

class test(ai.Model):
    def __init__(self, ):
        self.conv1 = ai.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
        self.fc1 = ai.Linear(18, 18)
        self.layers = [self.conv1, self.fc1]

    def forward(self, x):
        o1 = self.conv1.forward(x)
        o2 = ai.G.relu(o1)
        o3 = ai.G.maxpool2d(o2)
        o4 = ai.G.dropout(o3, p=0.75)
        o5 = self.fc1.forward(o4)
        # o6 = ai.G.softmax(o5)

        return (o1, o2, o3, o4, o5)

model = test()
L = ai.Loss(loss_fn='TestLoss')
optim = ai.Optimizer(model.layers, optim_fn='Adam', lr=1e-3)


it, epoch = 0, 0
loss = np.inf
m = 8

loss = np.inf
while it < 1:

    res = model.forward(X)

    loss = L.loss(res[-1], None).w[0][0]
    L.backward()

    # testing the library here....
    print(X.reshape(3, 2, 6, 6))
    print('Conv2d kernel\n', model.conv1.K.w, '\n')
    print('Conv2d bias\n', model.conv1.b.w, '\n')
    print('Conv2d ouput\n', res[0].w, '\n gradient\n', res[0].dw, '\n\n')

    print('ReLU output\n', res[1].w, '\n gradient\n', res[1].dw, '\n\n')

    print('MaxPool2d output\n', res[2].w, '\n gradient\n', res[2].dw, '\n\n')

    print('Dropout output\n', res[3].w, '\n gradient\n', res[3].dw, '\n\n')

    # print('Linear weight\n', model. fc1.W.w, '\n')
    # print('Linear bias\n', model.fc1.b.w, '\n')
    print('Linear layer output\n', res[4].w, '\n gradient\n', res[4].dw, '\n\n')

    # print('Softmax output', res[5].w, res[5].dw, '\n\n')


    optim.step()        # update parameters with optimization functions
    optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

    it += 1

# model.save()
