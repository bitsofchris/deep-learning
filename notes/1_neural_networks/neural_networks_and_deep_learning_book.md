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


# Chapter 3 - Improving the way neural networks learn

Backprop is our vanilla/ basic algorithm

This chapter teaches
- a better cost function: cross-entropy
- regularization methods: L1, L2, dropout, artificial expansion
	- these help our networks generalize beyond the training data
- better method for initializing weights
- heuristics to choose good hyper-parameteres

### Cross-Entropy Cost Function
- We want things to learn faster when more wrong
- slow learning is really the same as having small partial derivatives of our cost function
- the graph of our sigmoid function - is an S, when close to 1 or 0 or curve is very flat
	- looking for a cost function that eliminates derivative of sigmoid to avoid these slow learning periods

Cross-entropy as a cost function addresses this slow down in learning.



n= total number of training samples


Properties of Cross-entropy
- it's non-negative
- if neuron output is close to desired output for all inputs, cross entropy is close to 0
- and it avoids the problem of learning slow down

Partial derivative of cross-entropy cost w.r.t. weights.
- the rate at which our weights learn is controlled by the error in the output (sigmoid z - y)

Partial derivative of cost w.r.t. bias


- cross entropy helps us learn faster when our neuron is very wrong
- cross entropy almost always better choice when using ouput sigmoid neurons - since starting with random weights, if we are really wrong and close to 0 or 1, we can learn fast when really wrong now
- neuron saturation is an important problem in neural networks (when its heavily 1 or 0)
	- it causes this learning slowdown when using a quadratic cost function


Cross-entropy is a measure of surprise. How surprised we are (on average) when we learn the true output value.



##### Softmax
Softmax layers of neurons - another approach to address the problem of learning slowdown.
- defines a new type of output layer
- sum of the output activations are guaranteed to sum to 1

The output from softmax are a probability distribution

Use with a log-likelihood cost function.

Softmax + log-likelihood cost = whenever you want to interpret output activations as probabilities



### Overfitting and regularization
- too many free parameters -> too much freedom for a model and eventually it can describe anything without being useful to generalize
- signs of overfitting 
	- cost on training goes down while cost on validation data goes up
	- accuracy on the training data is too high (gap between train and test is wide)
	- accuracy improves and then hits a wall
- need way of detecting overfitting so not to overtrain
	- compute accraucy on validation data each epoch, if saturated - stop training
- More data often helps in preventing overfitting

##### Regularization

- regularization techniques help reduce overfitting without shrinking the network or getting more data
- L2 (weight decay) regularization
	- adds an extra term to the cost function
- regularization gets the network to prefer small weights
	- a balance between minimizing original cost function and finding small weights
	- when regularization parameter is small - prefer original cost function, when high prefer small weights


Regularized Cross-Entropy
(The regularization term is the last part)
regularization parameter / 2n  *  the sum of all the weights squared


- The more samples (the bigger the n) - the bigger weight decay (regularization param) to use.
- L2 regularization works b/c with out it - over time the weight vector can get large which makes gradient descent have a harder time changing it's direction (with tiny steps) - so we get stuck pointing the weight vector in the same direction 

##### Why does regularization help reduce overfitting
- all else equal - go for the simpler model/ explanation
- smaller weights in regularized network -> means network wont chant too much if we change some random inputs
	- makes it harder to learn "noise" in the data
	- learns from data seen across the training set
- larger weights - a network can over calibrate to noise

```
In a nutshell, regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, and are resistant to learning peculiarities of the noise in the training data.
```


- sometimes though complex explanations are correct
- remember the true test of the model is how well it does in predicting on unseen data
- the human brain regularizes very well 


##### Other Techniques for Regularization
* L1 Regularization
	* different term but still shrinks the weights - but shrinks by a constant amount toward 0 (L2 shrinks by amount proportional to weight)
* Dropout
	* modifies the network itself
	* start by randomly deleting half the hidden neurons
	* then forward and back propagate
	* repeating over and over with a new random set of neurons deleted during backprop
	* gives the effect of averaging across many different networks
* Artificially increasing training set size
	* take an image of the digit and transform it in some way to get more training samples
	* operations should reflect real-world variation, not just adding noise

Some research may just be using new techniques to improve on a benchmark where the method only works on that training set (vs other/ previous techniques).


### Weight Initialization
- can we do better than just initializing with random?
- saturated hidden neurons (close to 0 or 1) will learn very slowly
- use a more narrow peak for our gaussian distribution of random weights 
	- 1 / radical(n in)
- will learn faster


### Revisiting Digit Recognition
