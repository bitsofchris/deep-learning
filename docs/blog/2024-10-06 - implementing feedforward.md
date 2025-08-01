# Neural Network Basics

### Feedforward

We take the inputs of one layer, get the weight sum of all inputs * weights, add a bias, and put through activation function.

That's how you compute the value of a neuron in the current layer.


#### Talking with Claude

> my feedforward function is used to compute the output of my network? And this can in turn be used at inference time to predict any given input?But is also used at training time to compute the output of network which is then used to compute cost, etc?
 
You're absolutely correct on all points! Let's break down your understanding:

Purpose of the feedforward function:

Yes, the feedforward function computes the output of your network given an input.


Usage at inference time:

Correct, once your network is trained, you use the feedforward function to predict outputs for new inputs.


Usage during training:

Spot on! During training, the feedforward function is indeed used to compute the network's output, which is then used to calculate the cost/loss.



Let's elaborate a bit on each point:

Feedforward for output computation:

It takes an input and propagates it through all layers of the network.
The final element in the returned list of activations is your network's output for that input.


Inference (prediction):

After training, you use the same feedforward function with the learned weights and biases.
You give it new, unseen inputs, and it returns predictions based on what it has learned.


Training process:

During training, you use feedforward as part of a larger process:
a. Feedforward to get the current output
b. Compare this output to the true label to compute the loss/cost
c. Use backpropagation to compute gradients
d. Update weights and biases based on these gradients

### Python
Embarassed to admit I've gotten rusty to not realize assigning a variable within a for loop allows that variable ot persist within the lifecycle of the for loop. Meaning, the updated value will persist on the next iteration of the for loop. But it will be garbage collected at some point after the for loop ends.

# Left Off
Something not right when adding the bias. Getting an output array of 5 values instead of the expected single value for my last neuron. Np.dot seems to work as expected but maybe sometihng about lists vs lists of lists that's causing the unexpected behavior.