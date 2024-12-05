# Intro

Stage 1 - Building a LLM
- Data prep and sampling
- Attention mechanism
- LLM architecture
- Pre-Training to create the Foundation Model

Stage 2 - Foundation Model
- Training loop
- Model Eval
- Load pretrained weights
- Fine-tuning to solve your problem

Stage 3 - Use it for inference


# Chapter 1 - Understanding Large Language Models
- LLMs are deep neural networks trained on lots of data with 10s or 100s of billions of parameters.
- LLMs are trained to predict the next work in a sequence.
- LLMs use a TRANSFORMER
	- transformers allow for selective attention to different parts of the input
### Stages of Building & Using LLMs
- Pretraining - train model initially on large dataset of unlabeled data
	- LLMs use self-supervised learning where model generates its own labels from the input data
	- pretrained LLM aka base or foundation model
- Fine-tuning - train pretrained model on narrower, labeled dataset more specific to your task
	- instruction fine-tuning - instruction & answer pairs
	- classification fine-tuning - text & class labels
- Few-shot - give LLM a few examples to teach it a new task
- zero-shot - no training examples

### Transformer architecture
- 2017 paper “Attention Is All You Need” (https://arxiv.org/abs/1706.03762)
- Two submodules
	- encoder - takes input text, encodes into vectors that capture the context of input
	- decoder - takes encoded vectors and generates output text
- Encoder and decoder have many layers connected by self-attention mechanisms
	- self-attention allows the model to assess the importance of different tokens relative to each other in the sequence
	- self-attention is what allows capturing of long-range dependencies and context from the input
- Transformers can be used for other tasks, not just LLMs
Encoder/Decoder adapted to different tasks
- BERT - bidirectional encoder representations from transformers
	- specialized in masked word prediction, good for text classification
- GPT - generative pretrained transformers
	- focused on decoder and generation of text


### A Closer Look at GPT architecture
- next work prediction is a form of self-supervised learning - "create labels on the fly" using the structure of the data (ie. next word in a sentence)
- GPT is just decoders (no encoders)
- Autoregressive - use their previous outputs as inputs
- emergent capabilities - ability to perform tasks not explicitly trained for

### Summary - Chapter 1
- LLMs based on Transformers
	- Transformers - have an attention mechanism that gives LLMs selective access to the whole input sequence
- Original Transformer uses encoder for parsing text, and decoder for generating text
- GPT only implements decoder modules, specifically designed for generating text 

# Chapter 2 - Working with Text Data

### Understanding word embeddings
- embeddings are data converted to a vector format
	- words translated into numbers so we can do math on it
	- map discrete objects (words, images, etc) to points in a continuous vector space


### Process 
- Tokenization - breaks text into individual tokens
	- byte pair encoding - breaks down words not in it's vocabulary into smaller sub-words and even individual characters. can handle out-of-vocabulary words
	- can create special tokens to handle things like end of text
- Each unique token is mapped to a unique integer Token Id
	- have a map to go from token to id and id to token
- Input-Target pairs are generated - use a context window, a stride length, and slide through the text
- Creating token embeddings - initialized random weights, create embedding layer in PyTorch
	- `torch.nn.Embedding(vocab_size, output_dim)`
- encode positions
	- relative - distance between other tokens
	- absolute - at exact position
		- include an additional embedding layer that is of length context window

### Summary - Chapter 2

##### Pipeline
 1. Input Text -> broken into Tokens
 2. Tokens -> converted to Token IDs using a vocabulary
 3. Token IDs converted to embedding vectors
 4. Add positional embeddings
 5. Input embeddings = token id convert embeddings + positional embeddings
 6. Feed the input embeddings into the GPT decoder

##### Recap
- embeddings convert discrete data into continuous vectors
- sliding window on tokenized data to generate input-output pairs for training


# Chapter 3 - Coding Attention Mechanisms

Recurrent Neural Networks were the most popular encoder-decoder architecture. Takes output from previous steps to feed as inputs to the current step.
- encoder updates its hidden state at each step trying to capture the meaning of input
- decoder then takes this final hidden state to generate
- this hidden state == an embedding vector
- dont work well for longer texts

### Self-Attention
- Each position in the input sequence can consider how relevant all other positions in the same sequence are
- relates positions within a single input sequence

##### Simple Self-Attention
- end goal is to have a CONTEXT VECTOR for each input token
	- this context vector combines information from all other input tokens
- Context Vector = Attention Weights for each token * input vector of each token
- The context vector is an enriched embedding vector - it has information about the related tokens baked into it

Computing Self-Attention (aka getting the context vector)
1. We have our embeddings for each input token
2. For each input token, one at a time (aka our query token): 
	1. compute the dot product between that token and every input token
	2. dot product == a similarity score, how much does the current (query) token relate to the embedding of each input token
3. the resulting vector (one scalar value per input token) is the attention scores vector for that query token (query just means the token we are computing attention scores for)
4. normalize (softmax) the attention scores to get the attention weights
5. context vector = input embedding vector * attention weights 
	1. Basically: input vector *  normalized attention scores which are just the dot products of a query vector to each token


Simpler
1. Compute attention scores: dot products between every input
2. Compute attention weights: normalize via softmax
3. Compute context vectors: attention weighted sum of inputs


##### Self-attention with trainable weights
- Scaled dot-product attention
	- we still want to compute context vectors as weighted sums over the input vectors (token embeddings) for each input element
- Weight Matrices
	- Query
	- Key
	- Value
- These matrices are trainable parameters that project the embedded input tokens into these Q, K, V spaces
	- the matrices are shared only across the current context window for each attention layer and each attention head
	- meaning there is a set of Q, K, V matrices for each context window, for each attention head, in each attention layer
- Query and Key vectors - we use dot product to compute attention scores
	- dot product is a measure of similarity, how much do the two vectors point in the same direction
	- We take the query of 1 token and multiple by it by the key of every token to get a vector of attention scores
- Attention scores are then scaled with softmax to become attention weights
- Context vector for a token = weighted sum of attention weights with each tokens value vector
	- basically attention weight says how much of this tokens value vector do I want to use in the new context vector 
- Query - think search term or "current" word/token
- Key - used for indexing to match the query
- Value - the actual content

##### Self-Attention TLDR
Take input vectors of the token embedding -> use the Wquery, Wkey matrices to compute the attention weights. Then use those weights and the Wvalue to update the input embeddings with context (creates the context vectors for each token).


### Hiding Future Words with causal attention

Mask future words in a sequence so not looking ahead.

1. Mask the unnormalizd attention scores aobve the diagonal (future words) with negative infinity.
	1. `torch.triu()`
2. Then normalize with softmax.
3. Optionally apply dropout after computing the attention weights 
	1. `torch.nn.dropout(% to drop)`
	2. non-dropped weights get scaled according to `1/% to drop`

`register_buffer()` is used to store the mask in the model's state_dict, it is not a trainable parameter but allows it to follow the model around

using `_` after a pytorch function has it operate in-place

### Extending Single-head to Multi-head attention

- To this point, we have done single-head attention.
- If we stack muliple single head attention blocks we can have multi-head attention.
	- just concat the context vector matrices
- Running multiple attention heads in parallel allows the model to attend to different information 
##### Optimization
- while you can just stack single attention heads and concatenate the context vectors, it's more computationally efficent to have single QKV matrices and split them for each attention head
- To do this we need to use a `head_dim` size which is the `output_dimensions / number of attention heads`. 
	- We then take this `head_dim` and use `view()` to reslice the Tensors so we can split the tensors into another dimension, one for each attention head
	- We have to be sure to transpose the QKV matrices after the spliut so that it goes `batch, heads, tokens, dimensionality of each token in each head`. This order in the tensor allows us to operate on each head separately.
