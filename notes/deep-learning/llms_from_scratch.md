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


### Utilizing large datasets