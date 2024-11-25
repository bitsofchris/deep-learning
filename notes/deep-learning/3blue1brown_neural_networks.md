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

Chain rule shows how the nudge in weights, impacts the nudge in z, and how they impacts the activation, which ultimately impacts cost.

So to see how a change in a weight or bias impacts cost, we use chain rule form calculus to compute.

Weights and previous activations impact the current activation which impacts cost. “neurons that fire together wire together”

Backprop basically helps us trace back all the impacts of a weight and bias have on a cost function. Walking backward from output layer

# chapter 5

Embeddings are the input vectors from the data.

Weight parameters are the matrices of tunable weights at various layers.

Dot product- element wise multiplication that is summed, results in single scalar value. Measure of similarity.

Weighted sums - matrix x vector

Transformers with attention take the token embedding vector and add to it more information - like its position and context of surrounding words.

Context size limits how much context is used with attention mechanisms.

Last step - of transformer take the vector and use a matrix from the vocabulary to predict next token.

Use only last vector because the others are used in training? 
Last vector x the unembedding matrix to get the softmax probabilities of next token.

Foundation of attention
Embedding start, unembedding the end.
Matrix multiplication and dot products for similarity.
Softmax normalizes the vector to be a distribution, temperature introduces some flexibility here by softening how close to max distribution the biggest value gets.

Softmax - normalizes output into probability.


