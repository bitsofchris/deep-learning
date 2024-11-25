# 2024-11-18 - Preparing Text
I have some competing priorities this week so I anticipate a few less updates than usual.

Still reading LLMs from Scratch. Over the weekend and this morning I wrapped up chapter 2. It probably be worth while to practice the text processing pipeline discussed in the book but I don't have the bandwidth too nor the interest. A lot of that feels mostly familiar.

##### What I did learn
- Transformers typically have encoder and decoder, GPT is decoder only
- Process for taking text and feeding it into a decoder
	- tokenize text
	- convert to integer ids (didn't know about this part)
	- use those ids to generate embeddings
	- add positional embeddings
	- pass in that combined embedding to the decoder (token + positional embeddings)