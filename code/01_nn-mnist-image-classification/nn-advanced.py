from typing import List
import numpy as np


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        # Compute the cost for a single sample
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        # Compute the error for the output layer, used by the backpropagation algorithm
        # z is not needed in this function but keeps a consistent interfact for cost functions
        # cross entropy learns better because it avoids the learning slowdown that occurs when the sigmoid function saturates a neuron
        # so remember the z is the output of the neuron before the activation function is applied and is not used here
        return a - y


class Network:
    def __init__(self, sizes: List[int], cost=CrossEntropyCost):
        """
        Sizes is a list of the number of neurons in each layer of the network.
        0th element = input layer
        middle elements = hidden layers
        last element = output layer
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self._default_weight_initializer()
        self.cost = cost

    def _default_weight_initializer(self):
        """
        Initialize weights using a Gaussian distribution with mean 0 and standard deviation 1.
        """
        # dont need biases for input layer
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def _large_weight_initializer(self):
        """
        From the previous example, we know that the sigmoid function saturates when the input is very large.
        This one can lead to slower learning.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _cost_derivative(self, output_activations, y):
        # returns the partial derivatives of the cost function with respect to the output activations
        return output_activations - y

    def feedforward(self, activations):
        for bias, weights in zip(self.biases, self.weights):
            dot = np.dot(weights, activations)
            z = dot + bias
            activations = self._sigmoid(z)
        return activations

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Applies gradient descent to a single mini batch.
        eta = learning rate (controls step size of updates to weights & biases)
        nabla = gradient vector
        lmbda = regularization parameter
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
        # Regularization term is added to the weights (1-eta * (lmbda/n)) * <previous weight update>
        self.weights = [
            (1 - eta * (lmbda / n)) * current_weights
            - (eta / len(mini_batch)) * nabla_w
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
        delta = (self.cost).delta(zs[-1], activations[-1], y)
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
        Evaulate the test_data by feeding it through the network and comparing the output to the expected output.
        """
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]

        return sum(int(x == y) for (x, y) in test_results)

    def SGD(
        self,
        training_data,
        epochs,
        learning_rate,
        mini_batch_size,
        lmbda=0.0,
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
                self.update_mini_batch(
                    mini_batch, learning_rate, lmbda, len(training_data)
                )
            if test_data and epoch % eval_interval == 0:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")

    def load_data(self, max_samples=None):
        """
        Load the training and test data from the .npy files.

        Return a tuple of the training data and test data.
        Each elemnt is a list of tuples (x, y) where x is the input image and y is the label.
        """
        train_images = np.load(
            "code/01_nn-mnist-image-classification/data/train_images.npy"
        )
        train_labels = np.load(
            "code/01_nn-mnist-image-classification/data/train_labels.npy"
        )
        test_images = np.load(
            "code/01_nn-mnist-image-classification/data/test_images.npy"
        )
        test_labels = np.load(
            "code/01_nn-mnist-image-classification/data/test_labels.npy"
        )
        if max_samples:
            train_images = train_images[:max_samples]
            train_labels = train_labels[:max_samples]
            test_images = test_images[:max_samples]
            test_labels = test_labels[:max_samples]
        train_data = list(zip(train_images, train_labels))
        test_data = list(zip(test_images, test_labels))
        # Slice the first 10000 samples from the training data to use as validation data
        valid_data = train_data[:10000]
        train_data = train_data[10000:]
        return train_data, valid_data, test_data


if __name__ == "__main__":
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    train, valid, test = net.load_data(max_samples=None)
    net.SGD(
        train,
        epochs=30,
        learning_rate=0.25,
        mini_batch_size=10,
        test_data=valid,
        lmbda=5.0,
        eval_interval=1,
    )
