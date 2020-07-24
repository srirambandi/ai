# import ai

AI library in python using numpy, with end-to-end reverse auto-differentiable dynamic Computational Graph. Implements general Deep Learning library components with the end API similar to that of my favourite, Pytorch.

### About

Begineers in Deep Learning will find this repo useful. The purpose of this library is to serve as an educational tool, a reference guide to better understand the mechanics of deep concepts of AI by implementing everything from scratch. I want to expose the functions of Deep Learning APIs as clearly as possible. I originally built this for myself to understand Deep Learning critically, whose importance is pointed by one of favourite AI researchers [Andrej Karpath](https://twitter.com/karpathy), in [this](https://youtu.be/_au3yw46lcg?t=786) video.

So, as you have guessed, the best way to utilise this library is by implementing your own from scratch. Refer to this library when you don't understand how a Deep Learning component is built, tweak it and have fun :)

### Features

This library implements:
  - [Parameter](https://github.com/srirambandi/ai/blob/master/ai/parameter.py) object - that holds the weights and gradients(wrt to a scalar quantity)
  - [Computational Graph](https://github.com/srirambandi/ai/blob/master/ai/graph.py) - to store operations during forward pass, as a bfs walk in Directed Graph and execute them in reverse order during backprop. This has all the necessary functions to help realise many layers to do deep learning(basic operations, activations, regularizers etc.)
  - Layers - the fundamental [Linear](https://github.com/srirambandi/ai/blob/master/ai/linear.py) layer, [LSTM and RNN](https://github.com/srirambandi/ai/blob/master/ai/sequential_models.py), [Conv2d and ConvTranspose2d](https://github.com/srirambandi/ai/blob/master/ai/convolutional.py), [BatchNorm](https://github.com/srirambandi/ai/blob/master/ai/batch_norm.py) and few other non-parametrized layers([pooling](https://github.com/srirambandi/ai/blob/master/ai/pooling.py), [regularization](https://github.com/srirambandi/ai/blob/master/ai/regularization.py))
  - [Loss](https://github.com/srirambandi/ai/blob/master/ai/loss.py) - Mean Square, Cross Entropy, BCELoss and few other loss functions.
  - [Optimizers](https://github.com/srirambandi/ai/blob/master/ai/optimizer.py) - basic SGD(momentum), Adam, Adagrad, Adadelta and RMSProp optimizer functions.
  - Visualization tool to draw the computational graph of any neural network you program.
  - Some example implementations using this library.

I will keep updating the library with more explanations, documentation and a similar library in my favourite language, c++ soon!

### Installation

This library requires Python>=3.6 and numpy. Install the library as below(Installation takes care of the dependencies):

````bash
pip install import-ai
````

or you could just clone this repo and work locally as below:

````bash
git clone https://github.com/srirambandi/ai.git
pip install -r requirements.txt
````

### Usage

1. **You can directly work with Parameter objects, ComputationGraph and have fun!**

(The graph engine takes care of the reverse-mode auto-differentiation - the backpropagation algorithm. It is of highest importance that you actually understand how these internal mechanics work together, that's the foremost intended purpose of this library.)

import and initiate
````python
>>> import ai
>>> x = ai.Parameter((3, 1), eval_grad=False)
>>> W = ai.Parameter((3, 3))
>>> b = ai.Parameter((3, 1), init_zeros=True)
>>> print(W)
Parameter(shape=(3, 3), eval_grad=True) containing:
Data: [[-0.01092495  0.00542457 -0.00562512]
 [ 0.00911396 -0.00143499 -0.0160998 ]
 [-0.01601084  0.01146977  0.00797995]]
````
do operations
````python
>>> y = (W @ x) + b       # supports basic arithmetic
>>> print(y)
Parameter(shape=(3, 1), eval_grad=True) containing:
Data: [[-0.00011536]
 [ 0.00012833]
 [-0.00023106]]
````
backward
````python
>>> y.grad[1, 0] = 1.0
>>> y.backward()
>>> print(W.grad)
array([[ 0.        ,  0.        ,  0.        ],
       [ 0.00873683, -0.00623124, -0.00246939],
       [ 0.        ,  0.        ,  0.        ]])
````
see the Computational Graph for the above program
````python
>>> ai.draw_graph(filename='linear')
````
![Computational Graph of linear](/assets/linear.svg)


Parameters(single circles) interact with functions(double circles) and output Parameters. The values in the circles of parameters are the node ids indexed with the bfs-walk of graph during forward pass, goes from lowest node id circle to highest node id circle. The backward pass is the same graph with edges reversed, goes from highest node id circle to lowest node id circle. The circles with ````None```` doesn't have any backward operations attached to them. The circes with red line doesn't need gradients(inputs, outputs, constants). Also, checkout some other nice renderings of computational graphs in the assets folder.


2. **Or a more schematic code to use in Deep Learning projects as below.**


````python
import ai


ai.manual_seed(2357)

def data_generator(file):
    yield data_batch

class Net(ai.Module):
    def __init__(self):
        self.conv = ai.Conv2d(3, 16, kernel=3, stride=1, padding=0)
        self.pool = ai.Maxpool2d(kernel_size=2, stride=2)
        self.drop = ai.Dropout(p=0.5)
        self.fc = ai.Linear(x, 10)

     def forward(self, x):
        o1 = ai.G.relu(self.conv(x))
        o2 = self.drop(self.pool(o1))
        o3 = self.fc(o2)

        return ai.G.softmax(o3)

model = Net()
print(model)

# loss and optimizer functions
L = ai.Loss('CrossEntropyLoss')
optim = ai.Optimizer(model.parameters(), optim_fn='Adam', lr=1e-3)

# inference
def evaluate():
    # testing and inference
    ai.G.grad_mode = False

    predicttion = model.forward(test_input)

    ai.G.grad_mode = True

# some control parameters
...

# training loop
while not converged:

    # get scores and compute gradients
    scores = model.forward(train_input)
    loss = L.loss(scores, outputs)
    loss.backward()

    # update weights
    optim.step()
    optim.zero_grad()
    
    # logging info
    print(...)
    

model.save()
````

### Implementations

Examples directory contains some basic popular Deep Learning implementations, and I will add challenging ones soon.

Other examples using this library, resting in their stand-alone repos are:

  * [GAN/Wasserstein-GAN Implementations](https://github.com/srirambandi/GAN)
  * [Neural Turing Machines Implementation](https://github.com/srirambandi/NTM)
  * ["Deep Learning for Symbolic Mathematics" - paper implementation](https://github.com/srirambandi/symbolic-mathematics) - Work In Progress

### Goals

To implement other learning paradigms besides Supervised such as, Unsupervised and Reinforcement Learning algorithms into the library.

I want to implement every model in the below Deep Learning Toolkit picture [source tweet](https://twitter.com/OriolVinyalsML/status/1212422497339105280?s=20)

![DL Toolkit](/assets/dl_toolbox.jpeg)

### License

MIT

### Author

Sri Ram Bandi / [@\_srirambandi\_](https://twitter.com/_srirambandi_)
