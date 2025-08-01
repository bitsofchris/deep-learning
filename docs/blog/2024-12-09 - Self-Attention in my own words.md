# 2024-12-09 - Self-Attention in My Words
I'm trying to move past the self-attention chapter, I probably should by now but I want to really just go deep this week and try to recap it all in my own words and re-implement it in my own Jupyter notebook.

Here's my raw pass at a Feynman technique that I'll ask Claude to correct, help me with later.

### Self-Attention

##### High Level
Pre-Requisites to understand:
- LLMs encode words as numbers.
- High dimensional vectors - aka a point with many values where each "axis" has some semantic meaning it captures for a word.

At a high level, attention means taking the context surrounding a word and updating the meaning of that word with what the model learns from the context.

Think of an ambiguous word "light". That word on it's own can have many different meanings, so the initial embedding is some average point that could mean any of the definitions for the word "light".

Attention means, how much attention should the word "light" pay to each word around it? Then let's update the meaning of the word "light" based on these words that provide context.

##### The Process
Attention let's us update the embedding of a token by adding the contextual meaning to that token. This enables us to include the contextual meaning into the current token.

First, the word "light" needs to know HOW MUCH to pay attention to every other word in the context window. 
- (Attention is computed within the context window - a fixed length of words the LLM can process in a single pass). 
- (I keep saying words but it's really tokens which could be subwords).

To do this, we compute the attention weights for the word "light". This is a two step process.
1. Get the dot product of "light" with every word in the context window.
	1. the dot product is a way to measure similarity. In this case if the vector of light is pointing more in the direction of another word their score will be higher.
	2. Words that are more "similar" get a higher score meaning their meaning contributes more to the context.
2. Normalize that result so it sums to 1.
	1. normalizing enhances numerical stability - which means it's easier for the model to do math downstream from this step

Second, we take this attention score and do some math to update the vector embedding for light to now include the attention weighted sum of the Value embeddings for each token in the context window.

This effectively says given my attention weights - let's take these Value embeddings of each token and add some context to the initial embedding for my word "light".

The result is the context vector for the word "light".

This is then done in parallel for every word in the context window producing a matrix of context vectors.

Other things to note
- we mask out words that come after the given token so we don't let the future impact the meaning before we know that word
- we can have multiple different context vectors for each token - called multi-attention where we have many attention heads that we aggregate together. they learn different patterns of meaning.
### Next
- [ ] do the math in a notebook - actually reimplement it again from my notes
- [ ] in this notebook detail out the KQV matrices
- [ ] improve my deeper explanation to not jump to the Value matrix - felt like this was introduced to soon and maybe I don't conceptually understand that piece as well