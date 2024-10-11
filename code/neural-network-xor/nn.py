from typing import List
import numpy as np

class Network:
    def __init__(self, sizes: List[int]):
        """
        Sizes is a list of the number of neurons in each layer of the network.
        0th element = input layer
        middle elements = hidden layers
        last element = output layer
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        # randn - random numbers in x, y shape
        # zip - combines two lists to iterate through in parallel
        # Biases - we want 1 bias for each neuron in every layer except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Weights we want 1 weight for each connection from every neuron in the previous layer to every neuron in the next layer
        # ie we need a connection from every input neuron to every hidden neuron
        # then from every hidden neuron to every output neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # weights are [ [weight for n inpout nuerons, ..], ..list of weights for n hidden neurons  ]

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _mse_loss_function(self, y_true, y_pred):
        # Mean Squared Error loss function
        return np.mean((y_true - y_pred) ** 2)
    
    def feedforward(self, activations):
        for bias, weights in zip(self.biases, self.weights):
            # First pass
            # bias is a list of 5 lists with 1 element each
            # weights is a list of 5 lists with 2 elements each (the weight from each of the 2 input neurons for each of the 5 hidden neurons)
            # the index of bias and index of weight correspond to the same hidden neuron
            dot = np.dot(weights, activations)
            z = dot + bias
            activations = self._sigmoid(z)
        return activations

    def update_mini_batch(self, mini_batch, eta):
        """
        Applies gradient descent to a single mini batch.
        eta = learning rate (controls step size of updates to weights & biases)
        nabla = gradient vector
        """
        # initialize as list of zeroes with same shape as our network
        # these will store the "accumulated gradients" for each mini batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # get the gradient vectors for this mini batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # add the gradient vectors to the initialized vectors
            # this creates a sum of the gradients for each mini batch
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # now update the weights & biases by averaging the sum of the gradients
        # zip pairs the accumulated gradients with the actual weights & biases
        # for each layer and it's gradient, compute the average gradient for this mini batch
        # and then we subtract this average from the current weights & biases to do the update
        self.weights = [current_weights - (eta/len(mini_batch)) * nabla_w for current_weights, nabla_w in zip(self.weights, nabla_w)]
        self.biases = [current_biases - (eta/len(mini_batch)) * nabla_b for current_biases, nabla_b in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple (nabla_biases, nabla_weights) that is the gradient
        for the cost function. The nablas are lists of numpy arrays in the 
        same structure as our self.biases and self.weights.
        """
        pass

# [(input x1 x2), (output y1)]
xor_data = [
    [(0, 0), (0,)],
    [(0, 1), (1,)],
    [(1, 0), (1,)],
    [(1, 1), (0,)],
]

if __name__ == "__main__":
    net = Network([2, 5, 1])
    print(f"Final Activation: {net.feedforward(np.array([[0], [0]]))}")