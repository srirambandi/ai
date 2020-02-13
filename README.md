# ai
AI library in python using numpy

The main purpose of this library is to understand the deep concepts of AI by implementing everything from scratch. I want to expose the functions of Deep Learning APIs as clearly as possible.

This library develops:
  - a Parameter object - that holds the weights and derivatives
  - a Computational Graph - to store operations during forward pass and execute them in reverse order during backprop. This has all the necessary functions to help realise many layers to do deep learning
  - Layers/models - the fundamental Linear layer, LSTM and RNN, Convolutional NN for now, and a generic model template for util functions
  - Loss - Mean Square, Cross Entropy loss functions, also has backward call function
  - Optimizers - basic SGD, Adam, Adagrad optimizer functions
  - some examples using this library

I will keep updating the library with more explanations, documentation and a similar library in my favourite language c++ soon!

# Goal
I want to be able to implement every model in the below Deep Learning Toolkit picture [source tweet](https://twitter.com/OriolVinyalsML/status/1212422497339105280?s=20)

![DL Toolkit](/assets/dl_toolbox.jpeg)
