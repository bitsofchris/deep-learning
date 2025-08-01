# 2024-11-25 - Self Attention
This was a good chapter to take my time with. I've also been using YouTube and the Feynman technique with Claude to understand it. Sebastian even says in the book this is the toughest part to grok.

After a week though it's starting to come together, I think this is the topic of a YT video for me to make to really drive it home.

The 3 Blue 1 Brown video definitely helped, and then stepping through the book multiple times too. I'm still in it but I have a good grasp on the overall concept and the simple first pass of self-attention.


### Self-Attention

Input data is in the form of an embedding vector that captures it's meaning. 

Self-attention's point is to imbue or enrich these embeddings with additional meaning that comes from the context surrounding a given token.

We compute this by first getting a similarity score for a given token compared to every token in the sequence. Dot product is used here. 
The dot product between a token and every token in the sequence gives us an "attention score" vector. 

This then gets normalized via softmax into an "attention weights" vector. Which basically says how much each token in the sequence matters to the current token.

You then take these attention weights and compute the weighted sum with the embedding vectors to get the context vector -> our new enriched embedding vector for each token that has some context baked into it.


### Next
Still working through the attention chapter and understanding it, but getting there.
