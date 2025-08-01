# 2024-10-19 - Cross-entropy and Softmax
I ran through the PyTorch tutorial a bit, again having struggled with the theory - it was pretty easy to follow along what PyTorch is abstracting away from you.

After getting the MNIST network working, I didn't see much point in doing another. I'm going to go through the rest of the book I started for now. But I think my initial goal of learning the basics of neural networks has been achieved.

I'll start thinking about next steps for this milestone or if I should move on to Transformers and use the LLMs from Scratch book (which I kind of want to start anyway!).

### Cross-Entropy & Softmax
I focused on reading Chapter 3 today. To perform active recall on what I just read, I asked Claude to clarify my understanding. 

Here's my reflection and prompt for Claude to elaborate. Turns out I was mostly correct?

Side note: I do like this method of recall after you read something - try to recap, in your own words, what you just learned. Bonus now is we can say it to an LLM and get corrected (hopefully).

##### My Prompt

Help check my understanding about neuron saturation and using "better" cost functions like cross-entropy and softmax to overcome the slow down in learning problem in neural networks. Here is my understanding, correct and elaborate to help me learn this:

Neurons are saturated when they are very close to 0 or 1. Using a quadratic cost function will lead to very slow learning in these cases, even if these neurons are very wrong because the sigmoid function is a flat s curve which means the derivative of this sigmoid function will learn slowly near the tails (0 and 1).

Cross entropy more directly measures how wrong we are to update the weights and biases faster when we are more wrong. It is a measure of surprise and doesnt have the sigmoid derivative term in the partial derivative of the cost for weights or biases.

Softmax is a variation of how we output our activations and can be combined with a log-likelihood cost function to give you an output layer that is equivalent to a probability distribution that sums to 1.

##### Claude's Response:

```
Your understanding is quite good! Let's break this down and elaborate on each point to help solidify your understanding:

1. Neuron Saturation:

You're correct. Neuron saturation occurs when the output of a neuron (typically after passing through an activation function like sigmoid) is very close to its extreme values (0 or 1 for sigmoid).

Elaboration:
- This is particularly problematic with sigmoid activation functions.
- In the saturated regions, the gradient becomes very small, leading to the "vanishing gradient" problem.
- This affects not just the output layer, but can propagate backwards, slowing down learning in earlier layers too.

2. Quadratic Cost Function and Slow Learning:

Your understanding is spot on. The quadratic cost function, combined with sigmoid activation, leads to slow learning when neurons are saturated.

Elaboration:
- The derivative of the cost with respect to the weights includes the term σ'(z), where σ is the sigmoid function.
- When the output is close to 0 or 1, σ'(z) is very small, making the overall gradient small.
- This small gradient means small updates to weights and biases, even when the neuron's output is far from the desired value.

3. Cross-Entropy Cost Function:

You've captured the essence of why cross-entropy is often preferred. It indeed helps to overcome the learning slowdown.

Elaboration:
- The partial derivative of the cross-entropy cost with respect to the weights doesn't include the σ'(z) term.
- This means the learning rate is not slowed down when neurons are saturated.
- The magnitude of the gradient is proportional to the error, so larger errors result in faster learning.
- Cross-entropy is particularly well-suited for classification problems where the outputs can be interpreted as probabilities.

4. Softmax and Log-Likelihood:

Your understanding of softmax is correct. It's often used in the output layer for multi-class classification problems.

Elaboration:
- Softmax converts a vector of real numbers into a probability distribution.
- It ensures all outputs are positive and sum to 1, which is ideal for multi-class problems.
- When combined with the negative log-likelihood cost function, it provides a smooth, differentiable alternative to the max function.
- This combination is mathematically equivalent to using sigmoid activations with cross-entropy cost in binary classification, but generalized to multiple classes.
```

### Next Steps
- [ ] Continue with the book, update my "from scratch" network with the improvements shown in Chapter 3
- [ ] After I finish this book, I think will just dive into the LLM book (or maybe a quick experiment of my own with a financial dataset?)