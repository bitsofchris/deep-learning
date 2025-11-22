"""
Five Ingredients of Every PyTorch Neural Network
=================================================
A minimal example that fits y = 2x + 3 using linear regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# ONE: Data
# ==============================================================================
# Sample x between -1 and 1, build targets y = 2x + 3 plus a bit of noise

torch.manual_seed(42)
x = torch.rand(100, 1) * 2 - 1  # 100 samples between -1 and 1
y = 2 * x + 3 + torch.randn(100, 1) * 0.1  # y = 2x + 3 + noise

# ==============================================================================
# TWO: Model
# ==============================================================================
# A tiny nn.Linear(1, 1) with one weight and one bias

model = nn.Linear(1, 1)

# ==============================================================================
# THREE: Loss
# ==============================================================================
# Mean squared error to measure how wrong predictions are

loss_fn = nn.MSELoss()

# ==============================================================================
# FOUR: Optimizer
# ==============================================================================
# SGD looks at gradients and decides how to nudge weight and bias

optimizer = optim.SGD(model.parameters(), lr=0.1)

# ==============================================================================
# FIVE: Training Loop
# ==============================================================================
# For each step: predict, compute loss, backprop, optimizer step

print("Training...")
for step in range(500):
    # Forward pass: predict
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass: compute gradients
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()  # Backprop

    # Update parameters
    optimizer.step()

    # Print progress every 100 steps
    if (step + 1) % 100 == 0:
        print(f"Step {step + 1:3d} | Loss: {loss.item():.6f}")

# ==============================================================================
# Results
# ==============================================================================
# After training, weight and bias should be close to 2 and 3

weight = model.weight.item()
bias = model.bias.item()

print("\n" + "=" * 50)
print("Results after training:")
print(f"  Weight: {weight:.4f} (target: 2.0)")
print(f"  Bias:   {bias:.4f} (target: 3.0)")
print("=" * 50)
print("\nThat's the whole recipe: data, model, loss, optimizer, loop.")
print("Every bigger network is just a scaled-up version of this!")
