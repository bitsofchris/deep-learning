# 2024-11-04 - Convolutional Neural Networks on MNIST

### A bunch of small improvements
The section on how CNNs avoid the instability in gradient learning problems was interesting. It really seems like - it doesn't but just that a host of techniques and optimizations allow us to still learn in a very deep neural network.

I was watching Andrej Karpathy's zero to hero course this weekend and one line stood out. At one point he was showing a simple neural network and said something like "that's it - everything else is just optimization".

I'm really starting to see that trend that in general - all deep learning is at it's core a basic neural network with a lot of techinques around it that improve the ability to scale either by adding more data, bigger networks, or more efficient algorithms to increase the compute. 

I guess also are various techniques that specifically improve the learning efficiency too.


### Next Steps
The book references an older ML library Theano, rather than dig this up I'll attempt to implement everything from this chapter in PyTorch.

- [ ] Implement a CNN following the book architecture (which follows LeNet 5) using PyTorch
- [ ] Apply some data augmentation
	- [ ] shift each image around by a pixel up, down, left or right.
	- [ ] get distracted and learn about other data augmentation methods applicable here (rotation, translation, skew)