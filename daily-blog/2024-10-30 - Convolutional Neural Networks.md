# 2024-10-30 - Gradient instability, Convolutional Neural Networks
Onto the final chapter, and it's actually a great fit for my learning plan as it takes me from basic NN into CNNs, then RNNs and LSTM which are on the list of concepts to understand.

Let me recap what I read today.

### Gradient instability
Our current method of computing the gradient will not really work on deep neural networks (networks with more than 1 hidden layer).

Exploding or vanishing gradient - basically the more terms we have to multiple together during backprop, the more unstable our method of finding the gradient.

If weights are really small, we continually reduce the value as we multiply more terms less than 1 which can cause earlier layers to learn much slower than later layers.


### Convolutional Neural Networks
A new type of network.  What I learned so far:

##### local receptive fields
instead of every pixel being an input neuron, we take a window or section of pixels as an input to a hidden neuron. In the book's example we take a 5x5 window of pixels.

We then slide this 5x5 window through the whole image, 1 pixel or some stride length, at a time. Each local receptive field maps to 1 hidden neuron.

##### shared weights, biases, feature map
For a given pass through the entire input image using the receptive field at some stride length - all those connections to the hidden neurons will have the SAME weights and biases.

This shared weights and biases for each receptive field through the image ensures this connection or "feature map"/ "kernel" / "filter" learns to detect the same feature.

We can then create a CNN with many feature maps.


### Next
- [ ] continue with this final chapter, there should be a lot more implementation to do
- [ ] still next up is likely doing some data pruning exercises to improve model training, I found a paper I want to implement, not sure when best to squeeze this in - maybe next week