# 2024-10-28 - Implemented L2 and small weights
Some very minor changes to the network to implement L2 Regularization. Just needed to tweak the `mini_batch_update` function to include the regularization term there.

Added the `CrossEntropyCost` function. Remember this function removes the need for the `z` term from our cost which avoids the slow learning when the activation of a neuron is saturated (close to 1 or 0).

And finally, changed how the weights were initialized. Still Gaussian but will be much smaller so we can avoid the initial slow learning from larger weights.

These tiny changes are showing an impact on my results, finally see my little network getting to 96% accuracy.

### Next Steps
- [ ] Keep reading the book