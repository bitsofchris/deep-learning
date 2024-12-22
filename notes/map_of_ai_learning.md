# AI Learning Framework Resource Guide

A comprehensive guide to understanding and learning AI at different levels. This resource accompanies the [video]().

![AI Learning Map](/notes/map.png)

## 🎯 How to Use This Guide
- Identify your current level
- Focus on basic understanding first
- Only specialize if needed for your goals
- I prefer to focus on one resource deeply, that clicks with me. But I move on fast if it's not providing the answers I want.

**Basic Understanding** - What's worth getting the gist of at each level.
**Specialization Path** - Who should go deep into the weeds here and understand the details, implementations, etc.

**Note**: While math is level 6, a basic level of understanding of Vectors and Matrix operations can go far. I suggest the [math basic resources](#level-6-math) for everyone willing to go deeper.

## 📚 Quick Navigation
- [Level 1: Applications](#level-1-applications)
- [Level 2: Models](#level-2-models)
- [Level 3: Architecture](#level-3-architecture)
- [Level 4: Components](#level-4-components)
- [Level 5: Mechanisms](#level-5-mechanisms)
- [Level 6: Math](#level-6-math)

## Level 1: Applications
### Basic Understanding
- Basic prompt engineering
- Basic limitations (e.g., context window, hallucinations)
- Tool selection for your use case (LLM, Multimodal, Image)

### Resources for Basics
- [Generative AI for Everyone](https://www.deeplearning.ai/courses/generative-ai-for-everyone/) by Andrew Ng
  - [My Notes](/notes/courses/Gen-Ai%20for%20Everyone.md)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

**Some of my writing on the topic:**
- [A Framework for Writing Useful Prompts I Learned from a Prompt Engineer](https://bitsofchris.com/p/a-framework-for-beginners-to-write) 
- [How to Prompt AI for Useful Summaries](https://bitsofchris.com/p/new-to-ai-prompt-writing-learn-how)

**My Video on thinking in higher dimensions:**
- [Thinking in Higher Dimensions: A simple way to do it](https://bitsofchris.com/p/thinking-in-higher-dimensions-a-simple)

### Specialization Path
- Prompt Engineers
- Custom Workflows via Custom GPTs, Gems, etc.
- If your optimizing and customizing workflows with AI tools

## Level 2: Models
### Basic Understanding
- Nuances of different models and their strengths (e.g., GPT4o vs o1)
- Model selection for your use case (i.e, thinking vs breadth vs speed)
- RAG (retrival augmented generation) - chatting with your own data

### Resources for Basics
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
  - [My Notes](/notes/courses/Langchain%20-%20Chat%20with%20your%20data.md)
- [Reasoning with o1](https://www.deeplearning.ai/short-courses/reasoning-with-o1/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [PrivateGPT](https://github.com/zylon-ai/private-gpt)

**My writing:**
- [What is RAG?](https://bitsofchris.com/p/what-is-rag-and-how-can-it-be-used)
- [I trained a local LLM on my Obsidian](https://bitsofchris.com/p/i-trained-a-local-llm-on-my-obsidian)

### Specialization Path
- AI Engineers
- Application developers
- If your building products with AI

## Level 3: Architecture
### Basic Understanding
- High-level understanding of common architectures (Transformers, CNN, RNN)
- High-level understanding of neural networks
- Pre-Training vs Fine-Tuning

### Resources for Basics
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
  - [My Notes](/notes/pytorch/intro_tutorial_notes.md)
- [Neural Networks & Deep Learning Book](http://neuralnetworksanddeeplearning.com/) - great introduction to neural networks
  - [My Notes](/notes/deep-learning/neural_networks_and_deep_learning_book.md)
- [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/en/training)

**My videos:**
- [Building the Simplest Neural Network](https://bitsofchris.com/p/building-the-simplest-neural-network) - 5min - single layer perceptrons
- [What is a Neural Network](https://bitsofchris.com/p/what-is-a-neural-network) - 6min
- [How Neural Networks Work (with PyTorch)](https://bitsofchris.com/p/how-neural-networks-work-understanding) - 6min
- [12,000 Dimensions of Meaning: How I Understood Self-Attention](https://bitsofchris.com/p/12000-dimensions-of-meaning-how-i) - 6min

### Specialization Path
- ML Engineers doing model fine-tuning
- AI Developers optimizing model performance
- Engineers scaling AI applications
- If your fine-tuning models for custom use cases

## Level 4: Components
### Basic Understanding
- Neural Networks in depth (feed forward networks)
- Transformer Blocks
- Convolutional Layers (CNNs)
- Self-Attention (specifically scaled dot-product attention) at a high-level
- Layer normalization, numerical stability

### Resources for Basics
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)
  - [My Notes](/notes/deep-learning/llms_from_scratch.md)
- [Andrej Karpathy's Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [3Blue1Brown - Neural Networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=HwrhpqahytAixshd)

**My videos:**
- [How I Finally Understood Self-Attention](https://bitsofchris.com/p/the-key-to-modern-ai-how-i-finally) - 18min

### Specialization Path
- ML Engineers
- Advanced AI Engineers
- If you're building custom models, deploying fine-tuned models at scale

## Level 5: Mechanisms

*This level is really just going deeper, into the implementation details of Components. I'm still on my learning journey and I don't have much to offer yet here.
This level also feels like where you go directly to the research, reading papers and trying to re-create on your own.*

### Basic Understanding
- Self-Attention implementation & it's variants
- Activation functions
- Architectural variants like MAMBA

### Resources for Basics
- [Illustrated Guide to Transformers](https://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need Paper](https://arxiv.org/abs/1706.03762)
- Other Foundational Papers
  - [A Logical Calculus of the Ideas Immanent in Nervous Activity (1943)](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)
  - [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain - 1958](http://homepages.math.uic.edu/~lreyzin/papers/rosenblatt58.pdf)
  - [Gradient-Based Learnning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

### Specialization Path
- Researchers
- Advanced ML Engineers
- If you're building foundation models, experimenting with cutting edge architectures

## Level 6: Math
### Basic Understanding
- Vectors
- Matrix Multiplication
- Partial Derivatives
- Gradient Descent

### Resources for Basics
- [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Khan Academy
  - [Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
  - [Partial Derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives/v/partial-derivatives-introduction)
- [My Notes](/notes/math/linear_algebra.md)

### Specialization Path
- Researchers
- If you're developing new ways of doing AI, improving algorithimic efficiency, new architectures, etc.



## 🤝 Contributing
Feel free to suggest resources by opening an issue or PR.