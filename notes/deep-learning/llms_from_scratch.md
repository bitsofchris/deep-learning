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


