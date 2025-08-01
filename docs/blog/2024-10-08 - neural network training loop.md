# 2024-10-08 - The NN Training Loop and starting Backprop
Now that I have feedforward working (ability to predict output given inputs), we need to train the network.

Training loop as I understand it:
1. Feedforward - get output
2. Compute loss
3. Backprop - adjust the weights based on the loss with the gradient*

I don't want to get lost too deep into the math.

Loss function is straightforward here, just mean-squared error.

Backprop - I don't know enough here to just copy and paste something. I will go back to the book.

From what I understand, backprop takes the loss and computes a gradient at each layer to nudge the weights/biases at each layer in a direction that minimizes the loss.

I think that's mostly right but there's a lot of details there I'm missing. I know we use the partial derivatives to help determine which direction, and I believe the gradient is a vector of all partial derivatives? Something like that.


### Partial Derivatives
Took 10 minutes to just look up the notation and what "partial" meant here.
Was a nice calculus review.
TLDR: partial derivative is a derivative of a multi-variate function (a function with more than 1 input).
The partial derivative has a different notation and assumes the other variabels are a constant while you take the normal derivative.
It basically says here is how much a change in this variable will effect the output of the function, but it's only "part" of the story given the other variables too.


### Left Off
taking notes on the NN book, working through chapter 2 to understand backprop a bit better before coding it 


