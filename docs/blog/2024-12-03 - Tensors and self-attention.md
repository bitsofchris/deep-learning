# 2024-12-3 - Tensors and Implementing Self-Attention
Was good getting into a Jupyter notebook to implement some of the self-attention code I've been seeing. And as usual when you spend time taking action- you uncover things you didn't quite know.

I went on a small deep dive into [tensors](../notes/pytorch/tensors.md). The big takeaways:
- tensors are basically lists of lists of .. lists repeateded
- In general the shape is something like Batch -> Number of features of each vector

Self-Attention is starting to make sense - figure out how much each token relates to the other tokens in the context window. Then update the token's embedding with that context.

### Next 
Continuing implementing self-attention, finish the chapter then maybe do a word2vec example.