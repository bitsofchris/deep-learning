# Notes from Neural Networks and Deep Learning
[http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

# intro

This book about core principles of neural networks 

Don’t stress the math in chapter 2

Definitely do the exercises as a sanity check I understand 

Skip the problems- use them as inspiration for my own neural network project (something on financial or time series data).

# chapter 1 - basic NN

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
- left off notes