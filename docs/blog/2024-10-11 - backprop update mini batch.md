# 2024-10-11 - backpropagation update_mini_batch code

### My Understanding of Backprop
I asked Claude to review my understanding of back prop as I wrote yesterday. Here are some notes from that as I restate my understanding of backprop.

1. Segment data into mini batches
	1. SGD uses a batch size of 1 sample
	2. batch gradient descent uses all data
	3. We are using batches of data then averaging the gradient before adjusting weights and biases
2. Feedforward - calculate activations at each layer, for each sample
3. Error - get the error at the output layer first using your chosen loss function
4. Backpropagation - work backward computing error for each neuron in each layer. 
	1. Using chain rule in calculus we compute the partial derivatives of the loss function with respect to every weight and every bias.
5. Update the weights and biases -  Average the gradient across all examples in the batch and update

Backprop is efficient b/c it reuses the gradients computed at later layers to find the gradients for earlier layers.

### Backprop in Code
"Nabla" (∇) is the mathematical symbol for the gradient operator.

I don't understand the details deep enough to implement backprop with my eyes closed, and I think that's fine at this stage. So I use the code from the book but I typed it out manually and add my own comments to try and join the theory with what the code is doing.


### Left off
Worked through the update_mini_batch() function from the book. Feel good about that. Will try back prop tomorrow.

Might need to tweak some things from the book to match my XOR data.