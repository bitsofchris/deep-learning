# 2024-11-16 - Self Attention, my explanation
Didn't have much time wrapping things up before Thanksgiving vacation but slowly starting to grok the what and how of self attention.

### What is Self-Attention?
(warning this my active recall exercise, and can be incorrect)

It's a way for token embeddings to be updated with context of the tokens around them.

It allow for tokens in other parts of the context window to alter the meaning of a current token.

At a high level here's how it works:
1. Every token in the context window is an embedding vector. A numerical representation of the token's meaning.
2. Then each token in the context window is compared to every other token in the window to see how "similar" the token is.
	1. this is done by computing the dot product (between each token's embedding vector?)
	2. Key and Query matrices are used
3. Once we know how similar each token is to the other tokens in the context window - we normalize these dot products (which were the attention scores) into the attention weights
	1. basically, the more "similar" a token is to another, the more attention it should pay to that token when updating it's meaning
4. We then take these attention weights and multiply that with the Value matrix
	1. the value matrix is a set of values that capture some direction of meaning in this attention head
	2. the attention weights are applied such that only tokens with "similarity" to the token in question really add their "direction of meaning" or "context" to the current token vector
5. These new vectors are then used. They are basically the original token embedding + the context we added to them using self-attention.

I know there is some depth missing and potentially a few things wrong, but that feels pretty good.