# Fourier Transforms


## Journal

### 2025-05-31

I'm working on time series representation learning. I'm wondering if I can use Fourier transforms to give me weak labels on my time series data. I can then use these weak labels to train a classifier on my embeddings to evaluate the quality of the embeddings I produce.

I don't know enough about Fourier transforms but the idea of decomposing longer term and shorter term signals from the data seems interesting. Maybe there's a way to do a more hierarchical embedding or decomposition that captures a coarser, longer term pattern and then within each of those we learn the nuances.