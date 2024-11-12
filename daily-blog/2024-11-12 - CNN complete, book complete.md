# 2024-11-12 - CNN Training and Hyperparameters
Small correction to yesterday, you do need to define a `forward()` even if using `nn.Sequential()`.

### Training
Started with a CrossEntropy Loss function and after 10 epochs we got to 88% accuracy.

Switched to Negative log-liklihood and log softmax like the book does, and the first epoch was at 93% accuracy, which after 10 epochs hit 98.76% accuracy!

I'm surprised how swapping the loss function led to such a quicker training improvement.

The book ultimately got over 99.6% using 60 epochs and tuning things a bit more, I don't feel the need to do that. I want to start the LLM book I've been patiently waiting for :).

### Regularization
Adding regularization initially didn't quite work - we weren't training at all with a `weight_decay=0.1`. I think this is possibly due to not initializing weights more efficiently?.

Turns out I guess the 0.1 lambda from the book doesn't translate exactly, so I played around with tuning this a bit and found that `0.0001` with dropout worked pretty well.


### Dropout
Using dropout and a small weight decay - I didn't see much improvement from just running without these two. Possibly because I didn't extend my dataset like the book did.


### Next Steps
- [ ] An Idea for later - how much can I prune the MNIST dataset and still get 98% or more? I've been reading some data pruning research and think this could be an interesting exercise for a later time. I want to try a concept I've indirectly found "prototype" learning - where we averge or combine the embeddings of various clusters to create a "prototypical" model of some concept in the data and then train only on that.
- [ ] Looking forward to starting the LLMs from Scratch book