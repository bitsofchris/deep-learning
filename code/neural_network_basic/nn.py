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


xor_data = [
    [(0, 0), (0,)],
    [(0, 1), (1,)],
    [(1, 0), (1,)],
    [(1, 1), (0,)],
]

if __name__ == "__main__":
    net = Network([2, 5, 1])
    print(net.biases)
    print(net.weights)