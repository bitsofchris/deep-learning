# 2024-10-14 - XOR solved, image classification next

### I have XOR solved and working
Proud of this first milestone. A lot more I'd like to play with here - like visualizing how the network learns, the loss curve over epochs, and how different hyper parameters impact all that. But I'll do this visualization and hyper parameter exploration on a later problem.

### Starting image classification
This seems the next step, following the [book](http://neuralnetworksanddeeplearning.com/chap1.html)too - its the problem they use. I realized I skipped the SGD step from chapter 1 and for my XOR problem I just iterate through the entire XOR dataset (4 rows) thousands of time to train my network.

For the image recognition problem, 70k rows vs 4, that won't really work and implementing/ understanding SGD will be my next sep.

### Next Steps
- [ ] implement/ understand SGD
- [ ] write my own module to pre-process the MNIST training data (turn the images into arrays of pixel values)
- [ ] play with my network, try to get it to 96% on the validation set (I think this is common)
- [ ] parallelize this with ray - implement some profiling to see the performance