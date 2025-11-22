"""
OPTION 1: Live Updating Plot (BEST FOR SCREEN RECORDING)
=========================================================
Watch the model improve in real-time - perfect for YouTube!
Just run this and screen record the window.

BONUS: Also saves a GIF animation that you can replay anytime!
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

torch.manual_seed(0)

# 1) Generate full dataset
n_samples = 200
x_all = torch.rand(n_samples, 1) * 2 - 1
noise = 0.1 * torch.randn(n_samples, 1)
y_all = 2 * x_all + 3 + noise

# 2) Train/test split (random for this toy problem)
n_train = int(0.8 * n_samples)
perm = torch.randperm(n_samples)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

x_train, y_train = x_all[train_idx], y_all[train_idx]
x_test, y_test = x_all[test_idx], y_all[test_idx]

# 3) Model, loss, optimizer
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

# 4) Setup live plotting
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))

# Pre-compute for plotting
xs = torch.linspace(-1, 1, 100).unsqueeze(1)
ys_true = 2 * xs + 3

# Storage for replay animation
history = []

# 5) Training loop with live updates
for step in range(101):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Update plot
    if step % 1 == 0:
        with torch.no_grad():
            ys_pred = model(xs)

        # Store this frame for replay animation
        w = model.weight.item()
        b = model.bias.item()
        history.append(
            {
                "step": step,
                "loss": loss.item(),
                "weight": w,
                "bias": b,
                "predictions": ys_pred.numpy().copy(),
            }
        )

        ax.clear()
        ax.scatter(x_train, y_train, label="train", alpha=0.4, s=30)
        ax.scatter(x_test, y_test, label="test", alpha=0.8, s=30, c="orange")
        ax.plot(xs, ys_true, label="true line (y=2x+3)", linewidth=2, c="green")
        ax.plot(xs, ys_pred, label="model", linestyle="--", linewidth=2, c="red")

        # Show current parameters and loss in title
        ax.set_title(
            f"Step {step} | Loss: {loss.item():.4f} | w={w:.3f}, b={b:.3f}",
            fontsize=14,
        )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0.5, 5.5)

        plt.pause(0.01)  # Small pause to update display

    if step % 100 == 0:
        print(f"step {step:3d} | train loss = {loss.item():.4f}")

# 6) Final evaluation
with torch.no_grad():
    y_test_pred = model(x_test)
    test_loss = loss_fn(y_test_pred, y_test).item()

print(f"\nFinal test MSE = {test_loss:.4f}")
print(f"Final w={model.weight.item():.2f}, b={model.bias.item():.2f}")

plt.ioff()  # Turn off interactive mode

# 7) Create replay animation and save as GIF
print(f"\nCreating replay animation from {len(history)} frames...")


def update_frame(frame_idx):
    """Update function for animation - replays stored history"""
    frame = history[frame_idx]
    ax.clear()

    ax.scatter(x_train, y_train, label="train", alpha=0.4, s=30)
    ax.scatter(x_test, y_test, label="test", alpha=0.8, s=30, c="orange")
    ax.plot(xs, ys_true, label="true line (y=2x+3)", linewidth=2, c="green")
    ax.plot(
        xs, frame["predictions"], label="model", linestyle="--", linewidth=2, c="red"
    )

    ax.set_title(
        f"Step {frame['step']} | Loss: {frame['loss']:.4f} | "
        f"w={frame['weight']:.3f}, b={frame['bias']:.3f}",
        fontsize=14,
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0.5, 5.5)


# Create animation that replays the training
anim = FuncAnimation(fig, update_frame, frames=len(history), interval=100, repeat=True)

# Save as GIF (loops forever when opened)
anim.save("training_animation.gif", writer=PillowWriter(fps=10))
print("✓ Animation saved to training_animation.gif")
print("  (Opens in any browser/viewer and loops automatically!)")

plt.show()  # Show the looping animation
