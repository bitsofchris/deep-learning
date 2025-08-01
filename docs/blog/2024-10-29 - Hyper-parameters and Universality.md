# 2024-10-29 - Hyperparameters and Universality

Wow, chapter 3 was intense and long. I'm glad subsequent chapters start to lighten up. I can see the end of this book coming and am excited to next pick up the LLMs from Scratch book I've had on the shelf for a few weeks now.

### Good data is all you need
Related though, I might take a detour before that book to work on some ideas around "good data is all you need." I've come across a few papers about reducing large datasets down to the "best" samples and still training a model that's very close to the ability of one trained on the full dataset.

As a data engineer and overly pragmatic 'its good enough' kind of person, I find this line of work very interesting.

### Hyper-parameter Tuning
There are automated methods like grid search and more recently tools like Ray Tune to help here. But the process from the book is basically:
1. set up fast experimentation cycle (smaller dataset, validation set for accuracy)
2. get the order of magnitude correct
3. tune it a bit
4. try another parameter now
5. bounce back and forth

### Universality
This chapter was a bit over my head but conceptually from the nice visuals - I get it. Using a neural network, with just one hidden layer, there exists a way to approximate any continuous function.

Whether you can find it or should (with 1 layer) depends on the problem.
Adding multiple layers can be exponentially more efficient as the problem or function you are approximating gets broken down into sub-problems (different levels of abstraction).

### Next
- [ ] Keep reading the book, got a feeling will be extending my NN to be a deep network
- [ ] good data experiments