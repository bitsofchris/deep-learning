# YouTube Course

https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=HwrhpqahytAixshd

# Chapter 1
Do the image recognition problem

A plain vanilla neural network with no additional architecture is a multilayer perceptron

A neuron is just a thing that holds a number between 0 and 1

Number inside a neuron is its activation 

Input neurons are the first layer 

Hidden layers are in between

Output neurons are the last layer (activation here is the likelihood of model )

Activations in one layer determine activations in next layer.

A neurons activation = weighted sum of all the input activations
Each connection has a weight.

Sigmoid function to squish weight sum of activations into 0 to 1. Old school, slower to train?
ReLU - easier to train, more modern.

Bias for inactivity- some constant to delay when the neuron is meaningfully active 

Weights and bias for each neuron

Learning or training is finding the correct weights and biases 

Try to dig into what happens at each layer- the sub parts

Neuron is a function input is the outputs of previous layer and returns a number itself.

# Chapter 2 - Training
Gradient descent

Learning is finding the weights and biases that minimize a certain cost function 

# Chapter 3 - back prop
Weights and bias at each layer have the gradients summed at the layer so you have a list of nudges for a given layer aggregating all the gradients at that layer.
Then you back up another layer and sum there.

Randomly shuffle training data, divide into mini batch, take a step with the average of gradients for each mini batch. Enables faster computation.

Backprop is the algorithm to find the nudges for the entire network given a single example.

# chapter 4 - calculus of backprop



