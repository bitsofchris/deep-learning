# Data Pruning - MNIST Image Classification

# Goal
How far can we reduce the size of MNIST dataset and still get "good" performance?

Let's say good is 95% accuracy or better.


# Set Up

The baseline model will be LeNet-5.

![Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J., CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons](./LeNet-5_architecture.svg "Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J., CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons")



Train on the full 60,000 MNIST training images, test on the standard 10,000 images.

Final test accuracy expected ~98–99%.

# Results
Baseline, full dataset: 99%

Pruning
Keep Random 50%: 98.93%
Keep Random 10%: 97.20%
Keep Random 1% : 90.69%

Clustering
k=10000 (16.7% of data): 98.04%
k=5000  ( 8.3% of data): 97.55%
k=1000  ( 1.7% of data): 93.03%
k=600   (   1% of data): 93.57%

### With my Test harness now, initial results:
--- FINAL RESULTS ---
{'method': 'random', 'size': 600, 'best_acc': 89.11, 'best_time': 3.5133628845214844}
{'method': 'random', 'size': 1200, 'best_acc': 95.02, 'best_time': 4.036647081375122}
{'method': 'cluster', 'size': 600, 'best_acc': 92.0, 'best_time': 10.385218143463135}
{'method': 'cluster', 'size': 1200, 'best_acc': 94.26, 'best_time': 17.74043107032776}


# Resources

## LetNet-5
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
https://en.wikipedia.org/wiki/LeNet
https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python


## MNIST Problem
https://en.wikipedia.org/wiki/MNIST_database

http://neuralnetworksanddeeplearning.com/chap1.html


## Data Pruning
[Beyond Neural Scaling Laws](../../notes/papers/beyond%20neural%20scaling%20laws.md)
