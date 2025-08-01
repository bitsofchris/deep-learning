https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

# Tensors
Arrays that can run on GPUs.

Very similar to Numpy Arrays


Tensor attributes
```
tensor.shape # tuple of (rows, columns)
tensor.dtype
tensor.device # which device it is stored on
```

Operations on Tensors
https://pytorch.org/docs/stable/torch.html
- typically run on a GPU

```
# Matrix Multiplication
tensor @ tensor1
tensor.matmul(tensor1)

# Element-wise multiplication
tensor * tensor
tensor.mul(tensor)
```

In-place operations denoted with `_` as suffix on the operation, they change the value in place.


# Datasets & DataLoaders
Dataset stores samples and labels.
DataLoader - wraps an iterable around Dataset (extends the functionality)

### Custom Datasets
Three functions
```
__init__()

__len__()
# return the number of samples in the dataset


__getitem__(idx)
# return a sample at the given index
```

### Training
Typically:
- process samples in groups of mini-batches
- reshuffle data at every epoch
- multi-processing to speed up data retrieval

DataLoader helps handle this.

# Transforms
Datasets can specify a `transform` and `target_transform`.

`ToTensor()` - converts to FloatTensor, range 0,1.


# Build the Neural Network
- Set the device
- Define your network by inheriting from `nn.Module`
- Create instance of your Network and move it to device
	- `NeuralNetwork().to(device)`
- Do not call `model.forward()` directly - just pass the input to your network.
- `nn.Linear` - linear layer 
- `nn.ReLU` - non-linear activation to help model learn complex relationships
- logits are the raw values returned from the last layer of our network, Softmax scales them to 0-1 to represent predicted probabilities


# Automatic Differentiation with Autograd
Back propagation - model weights are adjusted according to the gradient of the loss function w.r.t. a given parameter.

Set `requires_grad=True` on tensors that are your model parameters - the things you can optimize.

### Compute Gradients
```
loss.backward()
```

### For Inference
```
with torch.no_grad():
```


# Optimizing Parameters

### Optimization Loop
Each Epoch
- iterate over training dataset to convert to optimal params
- validation/test - iterate over test data to check if model perf is improving

##### Loss Function
- measure how off our prediction
```
nn.MSELoss
nn.NLLLoss
nn.CrossEntropyLoss # (combines LogSoftmax and NLLLoss)
```

##### Optimizer
process of adjust model parameters to reduce error in each training step

ie Stochastic Gradient Descent

Pass in the model parameters to be trained (`model.parameters()`) and the laerning rate

### Train loop in PyTorch
```
for batch, (X, y) in enumerate(<DataLoader object>):
	# compute prediction and loss
	pred = model(X)
	loss = <loss function>(pred, y)

	# Backpropagation
	loss.backward()
	optimizer.step() # adjust the params by the gradients just collected in backward()
	optimizer.zero_grad() # resets the gradeitns of model params


```
