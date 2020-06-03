import ai
import numpy as np

# @karpathy 's min-char-rnn input trained with rnn
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

input_size = vocab_size
output_size = vocab_size
hidden_size = 100
seq_length = 25

np.random.seed(42)


class CharRNN(ai.Model):
    def __init__(self, ):
        self.rnn = ai.RNN(input_size, hidden_size)
        self.fc = ai.Linear(hidden_size, output_size)

    def forward(self, x, h):
        scores = []
        # hidden = []

        # hidden.append((h, c))
        for i in range(len(x)):
            h = self.rnn.forward(x[i], h)
            o = ai.G.softmax(self.fc.forward(h))
            scores.append(o)
            # hidden.append((h, c))

        return scores, h


charrnn = CharRNN()
print(charrnn)

L = ai.Loss(loss_fn='CrossEntropyLoss')
optim = ai.Optimizer(charrnn.parameters(), optim_fn='Adam', lr=1e-3)


def sample_chars(h, seed_ix, n):

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    chars = []

    ai.G.grad_mode = False

    for seq in range(n):
        o, h = charrnn.forward([x], h)
        ix = np.random.choice(range(vocab_size), p=o[0].data.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        chars.append(ix_to_char[ix])

    ai.G.grad_mode = True
    return chars


it, p = 0, 0
smooth_loss = 0.0
while True:

    if p+seq_length+1 >= len(data) or it == 0:
      h = ai.Parameter((hidden_size, 1), init_zeros=True)
      p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    x, y = [np.zeros((vocab_size, 1)) for i in range(seq_length)], [np.zeros((vocab_size, 1)) for i in range(seq_length)]
    for i in range(seq_length):
        x[i][inputs[i]] = 1
        y[i][targets[i]] = 1

    scores, h = charrnn.forward(x, h)
    loss = []
    for out, true in zip(scores, y):
        loss.append(L.loss(out, true))
    loss[-1].backward()

    optim.step()        # update parameters with optimization functions
    optim.zero_grad()   # clearing the backprop list and resetting the gradients to zero

    if it%100 == 0:

        curr_loss = sum([loss[i].data[0][0] for i in range(seq_length)])
        if smooth_loss == 0:
            smooth_loss = curr_loss

        smooth_loss = smooth_loss*0.99 + curr_loss*0.01
        print('Loss: iter', it, smooth_loss)
        # txt = ''.join(sample_chars(h, inputs[0], 500))
        # print('----\n %s \n----' % (txt, ))

        if it == 20000:
            charrnn.save()
            break

    p += seq_length
    it += 1
