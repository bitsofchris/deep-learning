# 2024-10-22 - Intraday Stock Classification
More data processing today - cleaning and pre-processing for training. This really, to no surprise, is this bulk of the work it feels like in building neural networks. Getting good, clean data that you can use for model training. 

The model training is the fun reward for having prepared a good dataset.

I also got the new model intialized and train/ test code in place. The overall process seems pretty formulaic at this stage. As the PyTorch code I used here is very similar to what I was using in the PyTorch NN code for digit classification.

### Next Steps
I am hitting an error that I'll work on tomorrow:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)
```

Seems trivial, just didn't have time to dig further.
- [ ] Get the model training, play with some hyper parameters then probably move on from this for now
