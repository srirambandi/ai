import ai_full as ai
import numpy as np

# I/O preparation
X = ai.Parameter((1, 2, 2, 1), init_ones=True)

class test(ai.Model):
    def __init__(self, ):
        self.convt = ai.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)
        # self.fc1 = ai.Linear(18, 18)
        self.layers = [self.convt]

    def forward(self, x):
        o1 = self.convt.forward(x)
        # o2 = ai.G.relu(o1)
        # o3 = ai.G.maxpool2d(o2)
        # o4 = ai.G.dropout(o3, p=0.75)
        # o5 = self.fc1.forward(o4)
        # o6 = ai.G.softmax(o5)

        return (o1)

model = test()
L = ai.Loss(loss_fn='TestLoss')
optim = ai.Optimizer(model.layers, optim_fn='Adam', lr=1e-3)


it, epoch = 0, 0
loss = np.inf
m = 8
while it < 1:

    model.convt.K.data = np.ones((1, 1, 3, 3))

    # X = np.stack([_ for _ in X], axis = -1)
    res = model.forward(X)

    res.grad[0, 1, 1, 0] = 23
    res.grad[0, 3, 2, 0] = 73

    for i in reversed(ai.G.backprop):
        i()

    # summaey
    print(X.data[0, :, :, 0])
    print(res.data[0, :, :, 0])
    print(res.grad[0, :, :, 0])
    print(model.convt.K.grad[0, 0])
    print(model.convt.b.grad)
    print(X.grad[0, :, :, 0])

    # loss = L.loss(res[-1], None).data[0][0]
    # L.backward()

    # testing the library here....
    # print([X[...,i] for i in range(X.shape[-1])])
    # print('Conv2d kernel\n', [model.conv1.K.data[i,...] for i in range(model.conv1.K.data.shape[0])],'\n')
    # print('Conv2d bias\n', model.conv1.b.data, '\n')
    # print('Conv2d ouput\n', [res[0].data[...,i] for i in range(res[0].data.shape[-1])], '\n gradient\n', [res[0].grad[...,i] for i in range(res[0].grad.shape[-1])], '\n\n')
    #
    # print('ReLU output\n', [res[1].data[...,i] for i in range(res[1].data.shape[-1])], '\n gradient\n', [res[1].grad[...,i] for i in range(res[1].grad.shape[-1])], '\n\n')
    #
    # print('MaxPool2d output\n', [res[2].data[...,i] for i in range(res[2].data.shape[-1])], '\n gradient\n', [res[2].grad[...,i] for i in range(res[2].grad.shape[-1])], '\n\n')
    #
    # print('Dropout output\n', [res[3].data[...,i] for i in range(res[3].data.shape[-1])], '\n gradient\n', [res[3].grad[...,i] for i in range(res[3].grad.shape[-1])], '\n\n')
    #
    # # print('Linear weight\n', model. fc1.W.data, '\n')
    # # print('Linear bias\n', model.fc1.b.data, '\n')
    # print('Linear layer output\n', [res[4].data[...,i] for i in range(res[4].data.shape[-1])], '\n gradient\n', [res[4].grad[...,i] for i in range(res[4].grad.shape[-1])], '\n\n')

    # print('Softmax output', res[5].data, res[5].grad, '\n\n')


    # optim.step()        # update parameters with optimization functions
    # optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

    it += 1

# model.save()
