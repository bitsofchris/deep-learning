# Notes from Neural Networks and Deep Learning
[http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

# intro

This book about core principles of neural networks 

Don’t stress the math in chapter 2

Definitely do the exercises as a sanity check I understand 

Skip the problems- use them as inspiration for my own neural network project (something on financial or time series data).

# Chapter 1 - basic NN

### My Notes
Gradient descent is our learning algorithm, our cost function or objective function. It seeks to find the direction to make a small change of the weights and biases to minimize the cost.
The cost being how off the output layer was from the desired result.

Feedforward NN mean data keeps moving forward never looping back,
Recurrent NN has data that can later loop back but not instantaneously.

Designing your network is a bit of art like how many hidden layers to trade off training time and accuracy. And how many output or what the output is.

Deep learning are neural networks with more than one hidden layer. Concepts or sub networks are basically decomposing the problem into sub problems at each layer. That’s how deep nets break down problems.

Deep networks use different newer learning techniques.


### My Highlights from reading
- neural networks -> learn to solve problems, they don't need explicit instructions on how to solve the problem
- stochastic gradient descent as the learning algorithm
- bitwise == binary representation of a number

##### perceptron neurons
- perceptron neurons (1950s) - binary 0 or 1
	- output = sum(weights * inputs) > threshold
	- its a device that makes decisions by weighing up evidence
	- a perceptron in second layer of network is making a decision at a more complex and abstract level
	- bias = threshold, its a measure of how easy it is to get that perceptron to "fire"
	- perceptions can compute any logical function

##### sigmoid neurons
- sigmoid neuron - 0 to 1 continuous
	- learning - the goal is to make small changes in the parameters (weights and bias) that produce small changes in the output. sigmoid neurons allow this b/c of their continuous nature
		- we want this b/c large jumps when learning might cause us to regress (get things wrong we were previously correct on)
	- sigmoid function (or logistic function) = really big values become 1, really negative values become 0, everything else is in between
		- it's a smooth shape between 0 and 1, rather than the discrete jump of a perceptron neuron
	- sigmoid is the "activation function" here - it can be something else

##### architecture of neural networks
- input layer (our input data) - hidden layer (not an input or output layer) - output layer
	- deep learning == more than 1 hidden layer
- art to the design of hidden layers
	- 1 trade off is the number of hidden layers vs time to train
- feedforward NN - output of 1 layers is input to the next, information only ever fed forward
- recurrent NN - loops are possible

##### Learning with gradient descent
- a cost function (loss or objective function) helps measure how close we are to our target output
	- in this example we use mean squared error
- this is used to inform how we can tweak weights and biases to get closer to the desired output
- Training algorithm purpose is to minimize the Cost function for our sets of weights and biases
	- Minimize `C(w, b)`
- We use the gradient descent algorithm
	- gradient vector relates changes to our weights in biases to changes in our cost
	- repeatdelty compute the gradient and then move in opposite direction to minimize cost
	- gradient descent = taking small steps in the direction which does the most to lower the Cost
	- find the weights and biases which minimize the cost
	- have to compute the gradients separately for each input (partial derivative)
- stochastic gradient descent - use a small sample of inputs to compute the gradient, speeds up the computation
	- m random samples into "mini-batches"
	- 1 epoch - all training examples use after continually sampling with mini-batches
- learning rate - how big of a step we take to minimize loss. do not want too big (might not converge) or too small (might take forever)
- high dimensional thinking - often not visual, but using other tools like algebraic representations

##### implementing our network
- training set - used to train the network
- validation set - used repeatedly to tune hyper parameters
- test set - used once to evaluate
- randomly initialized weights and biases in the network
- activations = sigmoid(weights * previous activations + biases)
- learning rate - size of steps when learning
	- .001 -> .01 -> .1 -> 1, etc
- epoch - one complete pass through training data
- Each Epoch
	- 1. randomly shuffle the training data
	- 2. slice into mini-batches
	- 3. for each mini batch -> gradient descent and update weights/biases
- Backprop algorithm - computes the gradient of the cost for a given training example
- no bias is used in input layer
- feedforward() - takes inputs and computes output of the network
- hyperparameters - configure the neural network and training
	- different from the parameters (weights & biases) of the network itself
	- tuning - if making a change improves things, try doing more of that
- debugging and tuning neural networks are a bit of an art
- a simple algorithm + good training data is often better than a sophisticated algorithm
- deep neural networks - have many layers compare to a single hidden layer for shallow neural networks
	- recursive decomposition into subnetworks - many layers break problems down into smaller and smaller problems or more abstract concepts

# Chapter 2 - Backpropagation

Backprop algorithm - how we find the changes to bias and weights impact the cost function 
- partial derivative of the cost function with respect to any weight or bias
- this tells us how quickly the cost changes when we change weights or bias
- first need to compute the error in a given layer

Partial derivative of weights and bias - computing the gradient of the cost function

Error is the small change 


### 4 equations of backprop
- together they give a way to compute the error at a given layer and the gradient of the cost function

![4 equations of backprop](images/4_equations_of_backprop.png)

Image taken from the book directly [here](http://neuralnetworksanddeeplearning.com/chap2.html)


##### error in the output layer (BP1)


partial derivative of the cost with resepct to activation at jth output neuron

##### error in terms of error in next layer (BP2)


error of layer l

Transpose weight matrix to move the error backward through the network

sigmoid of z -> becomes flat when really small or really large, it learns "slowly" as it's alrady close to 0 or 1 activation. said to be "saturated" and stops learning

As the error gets small, the neuron is near saturation, it will learn more slowly


Using the above two equations - can compute the error for ANY LAYER in the network.

first get error in the output layer, then use second equation to get error in output layer -1, etc.


##### rate of change of the cost with respect to any bias (BP3)

the error is equal to the rate of change here (since bias is a constant?)

##### rate of change of the cost with respect to any weight in the network (BP4)

in other words the rate of change in cost for a given weights = the activations in previous layer (the input to the weights) * error in the current layer

weights output from low-activation neurons learn slowly

### The backpropagation algorithm
- computing the gradient of the cost function
- error vectors are first computed at the last (aka output) layer
- it computes the gradient of the cost function for a single training example

Algorithm for single example
1. Set the input layer
2. Feedforward: compute activations for each layer
3. Output Error: compute the error vector
4. Backpropagate the error: for each layer compute the error at that layer
5. Return the gradient of the cost fucntion

in practice we compute the gradient for many training examples like in stochastic gradient descent - given a mini-batch of m examples

Algorithm for batches
outer loop generates mini-batches of training data
1. input training examples
2. For each training example
	1. set input
	2. feedforward and save activations at each layer
	3. compute final output layer error
	4. backpropagate error -computing error at each layer
3. for each layer update weights and biases with gradient descent average over the batch


### Is backpropagation a fast algorithm?
It just needs one forward pass and one backward pass through the network to compute the gradient.

The computational efficiency here enabled neural networks to be used at other problems (seems to be a trend, that as we get more data, more compute/more efficiency -> networks get more powerful/useful)

We can also compute gradients for other layers in parallel? 


### Backprop the big picture
Changing a single weight in one layer -> will change the activation of the connected neuron in the next layer -> which has an impact on ALL the activations in the following layer (since that neuron is connected to all neurons in next layer)

So one tiny weight change in layer 1 can have ripple effects throughout the rest of the network

Backprop is basically the method that tracks how tiny changes to weights and biases propagate through the network to reach the output and impact the cost function
