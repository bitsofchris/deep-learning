
https://arxiv.org/abs/2206.14486

##### Student Probe and their Pruning process
- Most/ all the pruning metrics require a model to compute the pruning metric
	- many pruning metrics can't be computed from the raw data alone
- A "student probe" is a lightly trained model on the full dataset. Like train for a few epochs
- Then this student probe is used to compute the various pruning metrics for each example in the training dataset
- These pruning metrics are then used with a pruning strategy to filter out the dataset
- The pruned dataset is the new 'teacher' in their theory and they train a model ('student') on this new dataset
- Then evaluate the new model

##### My Notes
- dataset size vs model loss - from power law to exponential through data pruning
- they look at existing data pruning metrics but most are expensive to compute (see above)
- theory of data pruning
	- when little data, keep easy examples - more general, hard example can cause overfitting
	- when lots of data, keep the hard examples - finer decision boundaries
- process of data pruning above
- more efficient learning - pareto optimal - dataset size vs model loss
- data pruning improves transfer learning - pre-trained models allow for more pruning
- their metric
	- k-means clustering on the embeddings, difficulty defined as the cosine distance to nearest centroid
	- prototype example - avg embeddings for all examples  -> discovered this as effective
- TLDR
	- self-supervised pruning metric to get exponential scaling error vs dataset size
	- pruning metrics are hard to compute but do help
	- foundation datasets - well pruned data that makes all downstream model training more cost-effective