# 2024-10-09 - starting backpropagation
I'm going to spend some more time on understanding this algorithm, then will attempt to implement and seek help if get stuck. I don't want to get lost in details here but I do want to struggle for a bit and try to understand what's happening.

As I understand it today:

To train a neural network we take a single training example, give the inputs, feedforward and get our output.

We then take that output and compute the loss given our expected output for that input (from our training data).

Once we know 'how wrong' the network is for an input we can go back through the network to tweak the weights & biases of each neuron to be 'less wrong'.

This process of going back through the network is the backprop algorithm.

From my reading today - we do this layer by layer? Looking at the error given some inputs and finding the partial derivatives of the Cost with respect to the bias and weights. This I believe is what creates the gradient vector that tells us what direction to nudge all the weights and biases to minimize the error across the network for a given training example.

Then we take all those nudges for a given training batch, average those, and then actually update the weights and biases of our network before doing it again for the next batch.

Once we go through every training input in a batch, we move on to the next batch. Once we see every batch, we complete an epoch of training.

### next for me
keep working through chapter 2 and see the 3blue1brown video for intuition, write my understanding up again and attempt to code it from scratch before looking anything up

