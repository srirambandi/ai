import ai_full as ai
import numpy as np

# I/O preparation
X = np.stack([np.zeros((3, 6, 6)), np.random.randint(-5, 6, (3, 6, 6))])

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
while it < 1:

    X = np.stack([_ for _ in X], axis = -1)
    res = model.forward(X)

    loss = L.loss(res[-1], None).w[0][0]
    L.backward()

    # testing the library here....
    print([X[...,i] for i in range(X.shape[-1])])
    print('Conv2d kernel\n', [model.conv1.K.w[i,...] for i in range(model.conv1.K.w.shape[0])],'\n')
    print('Conv2d bias\n', model.conv1.b.w, '\n')
    print('Conv2d ouput\n', [res[0].w[...,i] for i in range(res[0].w.shape[-1])], '\n gradient\n', [res[0].dw[...,i] for i in range(res[0].dw.shape[-1])], '\n\n')

    print('ReLU output\n', [res[1].w[...,i] for i in range(res[1].w.shape[-1])], '\n gradient\n', [res[1].dw[...,i] for i in range(res[1].dw.shape[-1])], '\n\n')

    print('MaxPool2d output\n', [res[2].w[...,i] for i in range(res[2].w.shape[-1])], '\n gradient\n', [res[2].dw[...,i] for i in range(res[2].dw.shape[-1])], '\n\n')

    print('Dropout output\n', [res[3].w[...,i] for i in range(res[3].w.shape[-1])], '\n gradient\n', [res[3].dw[...,i] for i in range(res[3].dw.shape[-1])], '\n\n')

    # print('Linear weight\n', model. fc1.W.w, '\n')
    # print('Linear bias\n', model.fc1.b.w, '\n')
    print('Linear layer output\n', [res[4].w[...,i] for i in range(res[4].w.shape[-1])], '\n gradient\n', [res[4].dw[...,i] for i in range(res[4].dw.shape[-1])], '\n\n')

    # print('Softmax output', res[5].w, res[5].dw, '\n\n')


    optim.step()        # update parameters with optimization functions
    optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

    it += 1

# model.save()
