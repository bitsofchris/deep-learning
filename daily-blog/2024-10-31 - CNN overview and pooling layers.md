# 2024-10-31 - Pooling Layer of CNNs
Busier day for me with Halloween and needing to do stuff with the family. But I got to a good stopping point in the book, I think tomorrow we will start implementing a CNN.

### Pooling Layers
For each feature map in the convolutional layer we will have a pooled layer that takes the feature map and basically condenses a region so we have a smaller encoding of the feature map.

My understanding is for max-pooling method we just take a region from the feature map and return the max activation from that entire region. Which effectively just says if a feature is present or not, rather than keeping the positional information.


### Putting it all together
I recreated the final diagram in the book to map out the overall architecture of a CNN.

Your input layer is stepped through by a window called the local receptive field. This stepping through with a window size creates a feature map in the convolutional layer. You can have multiple feature maps. 

For each feature map we connect 1 pooled layer which further condenses the information.

Then finally you have a fully connected output layer.

![CNN Architecture](../notes/1_neural_networks/images/cnns.png)

### Next Steps
- [ ] continue with this book
- [ ] after, dive into the data pruning papers and experiments