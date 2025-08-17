# Expected Signatures of Time Series

From this paper:


https://arxiv.org/pdf/2505.20465

# Notes

Signature
- S(X) is a stack of tensors of features that encode how a sequence moves over time
	- ![[Pasted image 20250817094331.png]]
	- S^(1)  = signautre at level 1
	- Each level (order/ degree/ level) k is a k-tensor built from iterated integrals 
- Iterated Integral
	- ![[Pasted image 20250817094442.png]]
	- Signature at the level k from level 1 through level k is the integral over ORDERED TIMES of TINY CHANGES of changes level 1 through level k
	- Ordered times - what moved virst
	- Tiny Changes - dX 
- Levels
	- Univariate -> single channel -> level 1 collapses its all redundanct
		- Single metric - add time as a channel, now you have level 2 -> encodes when changes happen
	- Multivariate metrics
		- level 1 - net changes per channel
		- level 2 - ordered co-movements
		- level 3 - ordered triples 
- Channels - sumlataneously observeid


Expected Signature
- The average of many window-level signatures (regardless if from one metric, one host, many services, etc - population can be anything)
- It's a fingerprint / summary of the signature over many samples from a given generator
	- its a fingerprint of the distribution of paths produced over T
	- Two cases
		- Deterministic generator (emits same sequence) - then `E[S(X)] == S(x)` its the same
		- Stochastic (variable) - you average across the emitted sequences
- A fixed window length T - average signature of its outputs
- `E[S(X)] == the average of the signature S(X)`


Benefits
- level-2 terms capture ordering (better than plan mean/variance/ correlation)
- many path properities -> become linear roughyl in signature space -> use logicsitic rgression
- distrubion shift tracking expected signature over time

Example
- Metric, with window, sampling.
- Pick depth
- Compue signature for each window
- Average across windows to get expected signature

Use Cases
- change in expected signature -> distribution shift
- anomaly scoring per window (vs a baseline)
	- `D(x)=(S(x)−μ​)⊤Σ−1(S(x)−μ​)` 
	- mu here is expected signature
- Similarity search


