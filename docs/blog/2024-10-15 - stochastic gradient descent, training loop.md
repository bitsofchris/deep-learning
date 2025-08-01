# 2024-10-15 - SGD, basic training loop

Spent some time reviewing my notes from the book and chatting with Claude to check my understanding of what I know so far. Cleared up a few things and ready to implement SGD and some more training / eval basics in the MNIST neural net.

Implemented SGD and the training loop plus a quick eval function on XOR, moved most of that code to a new folder for image classification.

Here's my understanding of training a NN:

### Neural Network Components
At a high level:
1. Data - you have data process and split into three datasets
	1. training - for actually training the model
	2. validation - for optimizing hyperparameters
	3. testing - the out of sample data you test against once
2. Model Architecture
	1. you choose the number of layers, number of neurons
	2. the activation function
	3. the loss function
3. Optimization Algorithm
	1. Stochastic gradient descent being one of them
	2. learning rate (hyperparam)
4. Training - the loop
	1. Set number of epochs (passes through the data)
	2. for each epoch
		1. shuffle the training data, split into mini-batches
		2. for each mini-batch
			1. for each sample in mini-batch
				1. forward propagation (get activations)
				2. compute loss on output layer
				3. back propagation (get the gradients)
			2. update the weights & biases using SGD
		3. if epoch_number % eval_interval == 0:
			1. evaluate model on validation set
			2. log metrics
	3. After all epochs: eval on test set
5. Evaluation
	1. test against validation set
	2. tune hyperparameters

### Stochastic Gradient Descent
Uses the gradients computed from backprop to update the parameters of the network.

- compute gradients for every sample In the mini batch
- Update network parameters within learning rate

Backprop - computes the gradients - that are then used by SGD to update the model parameters.

### Next Steps
- [ ] write my own module to pre-process the MNIST training data (turn the images into arrays of pixel values)
- [ ] play with my network, try to get it to 96% on the validation set (I think this is common)
- [ ] parallelize this with ray - implement some profiling to see the performance

