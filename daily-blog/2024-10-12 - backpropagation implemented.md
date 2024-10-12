# 2024-10-12 - backpropagation 
Between the book and the 3blue1brown videos - I have a good understanding of what's happening. The math details and the proof, eh, not fully but I think enough to move forward - at least not treating this entirely as a black box.

### TLDR
Backprop is a way to track how small changes in weights or bias ripple through the network and ultimately impact the final output which impacts the cost function. 

It's pretty efficient as it only requires one forward and one backward pass through the network to compute the gradient for a given sample.

In practice, we break our training data into mini batches, compute the gradient for each sample in the mini batch, then average the gradients for each sample in the minibatch to give us the actual update to apply to our weights and biases at each step.

### Today
I "implemented" the backprop algorithm from the book and used copilot along the way in VsCode to help explain lines as I tried to grok what was going on.

It makes sense.

update_mini_batch()
- takes a batch calls backprop() for each sample in a batch 
	- backprop() - computes the gradient for that sample using the backprop algorithm
- we then collect the gradients for each sample in our batch, average them, and actually update the network

backprop()
- computes the gradient for a given sample (nabla_weight, nabla_bias)
- first we feedforward the input and save the z and activation values of each neuron
	- z being just the weighted sum + bias of the neuron
	- activation being the sigmoid(z) of that neuron
- then using that we compute the delta and the gradient for our output layer
- once we have that we can compute the delta and gradient for every preceeding layer

*in our example we loop through each preceeding layer, but I believe the magic of backprop is that this can be done in parallel once you have the output layer*

### more data + more compute = more useful networks
This has been a theme I've noticed so far in my learning and the industry as a whole.

It seems each "breakthrough" has been some combination of either more processing power, more efficient algorithms (ie. Transformers for parallel computation, backprop in the 80s), or more good data.

Basically the idea that deep learning "works" seems to be holding. We just need more data, more compute/efficient and we can get more useful neural networks.

### Left off
now that I've implemented backprop and the update, I can actually try to use them in my xor network example. will probably quick run through the 3blue1brown videos here too just to crystallize my understanding intuitively, and then move on to chapter 3 of the book.

As well as experiment with training my network on XOR.
