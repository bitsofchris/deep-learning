# 2024-11-07 - CNN in PyTorch

I got the network architecture complete and now understand what is happening inside `nn.Sequential` in order to create the desired network architecture from the book.

The fun part was working through how each convolutional layer and pooling layer reduces the initial image size in order to accurately compute the number of input neurons to the first linear layer.

Of course you can just iterate through each layer and print the shape but thinking through with the configured window sizes felt satisfying.



### CNN in PyTorch
`conv2D()` 
- does not need the image size
- in_channels: 1 for grayscale, 3 for rgb images.
- out_channels: number of feature maps to produce
- kernel_size: local receptive field size as a square ie 5= 5x5 square
`ReLU()`
- applies the Rectified Linear Unit activation function to the output above it (in the context of `nn.Sequential`)
`MaxPool2d`
- kernal_size: window size to take the max from to reduce size of feature map
- the kernel_size is moved across the feature map at stride_length (defaults to kernel size) and the max is taken from each window at each stride resulting in a smaller feature map
`Flatten`
- Takes a tensor as input and returns a 1D tensor of the same size
- Takes tensor of (batch_size, channels, height, width) and returns tensor of (batch_size, channels * height * width)
- In the case of CNN - it takes all the feature maps and flattens it into a 1D array/ tensor so we can feed it to our Linear layers

### Sequential in PyTorch
- Basically an order of operations to apply to the Tensor passed to the network
- Unlike the code from scratch in the book - we have our layer, then we call an activation function on the output of that layer. These are two distinct calls, for example:

```
nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
nn.ReLU()
```

- we first define our convolutional layer, then call the ReLU activation function on it


### Next Steps
- [ ] continue implementing the network and get it to run
- [ ] add more features like dropout (from the project read me)