# 2024-10-24 - Regularization
Read a lot today - this is a long chapter - but got to a good stopping point right before he starts modifying the code again with the new techniques we learned.

### TLDR 
L2 Regularization - helps prevent overfitting by making the weights smaller, so we are more likely to ignore or at least not overcalibrate to noisy training data.

There are many other regularization techniques - but the goal is better generalization, less overfitting.

Also, we can learn faster if we initialize our weights in a narrower Guassian than standard so fewer neurons are saturated (and therefore fewer neurons learn slowly).


### Next Steps
- [ ] MNIST network from scratch add
	- [ ] regularized cross-entropy cost
	- [ ] initialize weights better
	- [ ] labmda = 5, 100 hidden neurons -> 98%?