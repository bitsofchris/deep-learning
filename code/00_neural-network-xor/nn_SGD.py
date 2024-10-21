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
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _mse_loss_function(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _cost_derivative(self, output_activations, y):
        # returns the partial derivatives of the cost function with respect to the output activations
        return output_activations - y

    def feedforward(self, activations):
        for bias, weights in zip(self.biases, self.weights):
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
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # Get the gradient for the cost function of a single sample
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Add the gradient for the sample to the total gradient for the mini batch
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Update weights and biases with the average gradient for the mini batch
        self.weights = [
            current_weights - (eta / len(mini_batch)) * nabla_w
            for current_weights, nabla_w in zip(self.weights, nabla_w)
        ]
        self.biases = [
            current_biases - (eta / len(mini_batch)) * nabla_b
            for current_biases, nabla_b in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Returns the gradient for the cost function of a given sample x.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activation = x
        activations = [x]
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = self._sigmoid(z)
            activations.append(activation)
        # backward pass for output layer
        delta = self._cost_derivative(activations[-1], y) * self._sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # backward pass for the rest of the layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self._sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return number of test inputs correct result.
        For XOR our result is the value of the output neuron
        """
        test_results = [(round(self.feedforward(x)[0][0]), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(
        self,
        training_data,
        epochs,
        learning_rate,
        mini_batch_size,
        test_data=None,
        eval_interval=10,
    ):
        """
        Stochastic Gradient Descent
        epochs: number of passes through the training data
        learning_rate: step size for updating weights and biases
        mini_batch_size: number of training samples in each mini batch
        if test_data provided, print out evalulation at each eval_interval epoch
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            # Shuffle training data
            np.random.shuffle(training_data)
            # Split into mini batches
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            # Update weights and biases for each mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data and epoch % eval_interval == 0:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")


# Temp for testing SGD
# [(input x1 x2), (output y1)]
xor_data = [
    [(0, 0), (0,)],
    [(0, 1), (1,)],
    [(1, 0), (1,)],
    [(1, 1), (0,)],
]
# Convert xor_data to an array of np.arrays
xor_data_np = [
    (np.array(input).reshape(-1, 1), np.array(output).reshape(-1, 1))
    for input, output in xor_data
]


if __name__ == "__main__":
    net = Network([2, 5, 1])
    net.SGD(xor_data_np, 10000, 0.1, 2, xor_data_np, 1000)
