# 2024-10-10 - backpropagation
We use backpropagation to update the weights and biases of our neural network during training.

At a high level - we segment our data into training batches of size m. Then for each training set of inputs in the batch we feedforward and get the activations at each layer. Then we compute the error for the final output layer. Once we have that we can work backwards computing the error at each previous layer for every neuron. We do that for all examples in our batch and average the gradients of all examples in our training batch to determine the update to every weight and bias in our network.

### Left off:
need to work through this a bit more