"""
The Five Basic Parts of Every PyTorch Neural Network
=====================================================
1. Data
2. Model
3. Loss
4. Optimizer
5. Training Loop
"""

import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(0)

# ==============================================================================
# 1) Data
# Generate fake data for equation y = 2x + 3
# ==============================================================================

# Tensor of shape 200, 1
n_samples = 200
x_all = torch.rand(n_samples, 1) * 2 - 1
noise = 0.1 * torch.rand(n_samples, 1)
y_all = 2 * x_all + 3 + noise

# Train/ test split
n_train = int(0.8 * n_samples)
perm = torch.randperm(n_samples)  # randomize indices
train_idx = perm[:n_train]
test_idx = perm[n_train:]

# Create train/ test splits from generated data using random indices
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_test, y_test = y_all[test_idx], y_all[test_idx]


# ==============================================================================
# 2) Model
# Simple linear layer (aka dense/ fully connected) - where every node is connected
# Takes input performs linear transformation to get output
# ouput = input * weight + bias
# ==============================================================================

# 1 input node, 1 ouput node
# 1 connection means 1 weight, 1 output node means 1 bias
# 2 learnable parameters
model = nn.Linear(1, 1)


# ==============================================================================
# 3) Loss
# How we measure the predictions
# ==============================================================================

# Mean square error
loss_fn = nn.MSELoss()

# ==============================================================================
# 4) Optimizer
# How we adjust our parameters to reduce the loss (and make a better fit)
# ==============================================================================

learning_rate = 0.1
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# ==============================================================================
# 5) Training Loop
# How we tie it all together to train a model
# ==============================================================================

steps_to_train_for = 100
print("Training...")

for step in range(steps_to_train_for):
    # Forward Pass: Get predictions with current model
    # We use all the data for this example at each step
    y_pred = model(x_train)

    # Compute loss with current predictions
    loss = loss_fn(y_pred, y_train)

    # Backward Pass: compute gradients
    optimizer.zero_grad()  # clear previous gradients
    loss.backward()  # backprop

    # Update model parameters
    optimizer.step()

    # Track progress
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1:3d} | Loss: {loss.item():.6f}")
        print(f"Weight: {model.weight.item():.4f} | Bias: {model.bias.item():.4f}")
        print("=" * 50)
