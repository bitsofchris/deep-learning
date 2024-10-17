# 2024-10-17 - fixing the mnist neural network

### We are training!
Spent sometime using the debugger today walking through parts of my network. My adapted XOR code was expecting only 1 output element, and because of that I was computing the cost wrong.

Makes sense - the network was running but not learning because my label data was not in the right structure to effectively compute the cost of the network.

Not sure where I deviated from the book but given that I took my own path first with XOR and Claude, it's plausible.

Had to fix my label data to match expected input of cost function and then had to tweak my evaluation function to compare my predicted output to the label correctly, but it's working as expected now!

### One-hot encoding
The labels from the MNIST training set are just integers. Part of the pre-processing step is to convert the labels from a single integer to a one-hot encoded vector.

A one-hot encoded vector is used to track categorical data where only one value is "hot" or true. So in our example, the output is any digit 0-9, we have 10 potential outputs, but only one that is true.

We convert our labels of say a 3 to a one-hot encoded vector `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. To do this we initialize an array of zeroes, then take the label as the index of our zero array and set that index to 1. A label of 0 would set the 0th element to 1, a label of 9 would set the 9th element of our array to 1.


### Next Steps
- [ ] Maybe review 1d vs 2d arrays in numpy and the notion of column vs row vector. Also how the shapes here matter for matrix multiplication.
- [ ] parallelize the training (and maybe data processing) with Ray as an exercise to work with Ray