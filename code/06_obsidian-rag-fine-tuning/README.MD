# Start simple - Distill an MOC

Given an MOC note, traverse links and summarize/ distill/ chat with the full context here.


# Fine Tuning a LLM on my Second Brain

This is a project I've had in mind for sometime that builds on my [early experiments with RAG](https://bitsofchris.com/p/i-trained-a-local-llm-on-my-obsidian).

# v0 - Fine-tuned Deepseek R1 - 1.5B on my Second Brain

I actually think this is pretty straight forward.

1. Turn my second brain into a fine-tuning dataset.
2. Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune.
3. Serve the model with [Ollama](https://github.com/ollama/ollama) locally.
4. [Use Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) for RAG with my fine-tuned model.

Here we go.


# Future Plans

- This [paper on chain-of-rag](https://arxiv.org/abs/2501.14342)
- Mixture of Experts combined with hierarchical summarization and something about concept learning I saw from Meta
- Building an AGI using Mixutre of Experts on the blockchain where users own their data and model weights and are compensated for usefulness provided to the overall system. Something like that.