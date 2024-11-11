# 2024-11-11 - PyTorch CNN Basics

Today I spent sometime reading PyTorch docs trying to generalize my previous PyTorch networks to a CNN here (it's basically the same). But I also spent a lot of time trying to understand why or what PyTorch requires from me vs what it does in the background.

Still in-progress.

##### PyTorch Requirements
1. Define your Network using `nn.Module`
2. Define how data move forward `nn.Sequential()` or `forward()`
3. Define a loss function and optimizer
4. Define a train() and test() function 

##### Network
Define your layers in `__init__()`.

##### Forward()
Defines how the data flows through the Network. Automatically called when you call `model(data)`. Never call it directly.

But `nn.Sequential()` is an ordered container for layers in our network. You do not need `forward()` when you have `nn.Sequential()`. You should define a `forward()` when you have more advanced data flow like branching or skip connections.

The `backward()` is implemented automatically as part of autograd.

##### Loss Function and Optimizer
Define these outside of your network. These work together to train your network

```
# in the training loop
outputs = model(inputs)  # Get outputs from the model for given sample
loss = criterion(outputs, labels) # Computes the loss 
loss.backward() # Computes gradients 
optimizer.step() # Updates weights
```


##### Define train and eval
These are not protected function names in PyTorch but rather conventions.

Train loop

##### Model vs Eval Mode
What is protected though is the `model.eval()` or `model.train()` is used to set the "mode" of the network.

From [Stackoverflow](https://stackoverflow.com/a/66526891)

| `model.train()`                                                                                                                                                                                                                                      | [`model.eval()`](https://stackoverflow.com/a/66843176/9067615)                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sets model in **train**ing mode i.e.  <br>  <br>• `BatchNorm` layers use per-batch statistics  <br>• `Dropout` layers activated [etc](https://stackoverflow.com/questions/66534762/which-pytorch-modules-are-affected-by-model-eval-and-model-train) | Sets model in **eval**uation (inference) mode i.e.  <br>  <br>• `BatchNorm` layers use running statistics  <br>• `Dropout` layers de-activated etc |
|                                                                                                                                                                                                                                                      | Equivalent to `model.train(False)`.                                                                                                                |

### Next Steps
Continue to templatize how a typical train and eval loop work in PyTorch.
Then implement this for my CNN.