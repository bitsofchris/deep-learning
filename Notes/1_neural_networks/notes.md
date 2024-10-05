# My Explanation of Neural Networks

(Feynman Technique of continually trying to explain it in my own words)


### 2024-09-26
##### level 1

It’s a funciton
Takes some input
Black box magic you don’t define but can size
Returns an output

##### level 2

Now first you need your black box to train and know what to do

This works by providing it an expected output with an input.

The model takes your inputs
And then does its magic producing an output
During training you tell it how wrong it is.

The model then makes some internal adjustments and tries again

The magic is you don’t need to tell it what to do inside just what is the expected output given the inputs and it keeps figuring things out

##### level 3
Now get into neurons?




### 2024-09-25 
Neural network is a big function whose parameters aka model are set during training.

The inputs are normalized data and are the activation values of the input layer

The hidden layers are initialized with a random activation value

Each neurons value is the weighted sum of the connecting neurons activation x its weight. 

The activation function normalizes the output here. Relu is faster and newer than sigmoid.

There is a bias as well which is a constant that gets added to the weighted sum? I think. It’s another parameter that can like mask or prevent the activation.

The weights and bias are tuned during training. Math makes it happen. 

The difference between the output layer results and some desired outcome? (Not sure how the goal or targets are set) is the cost or loss of the network.

Then math aka gradient descent is used to take a step with all weights and biases in the direction of minimizing the loss function. This updates the parameters for the entire network.

This process of slightly tweaking the weights based on taking a step in the direction that most reduces the loss is how a network trains.

### 2024-09-24
Explain to 12 year old - identify gaps - refine.

How a neural network works.

Given some input can we have a model predict the output without us telling the model how to behave.

A neural network is made up of layers of neurons.
Each neuron is a number and some weight.
Inputs are multiplied by a weight and then added together to be the value.

When training the weights keep adjusting 
Something about forward pass, back propagation, and a loss function.

Each pass of the data or through the network? Adjusts the weights of each connection.

Are all neurons connected? I gues some weights just go to 0?

Is the simplest method supervised? So have labeled data?