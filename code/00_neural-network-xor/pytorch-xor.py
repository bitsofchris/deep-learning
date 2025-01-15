import torch
import torch.nn as nn
import torch.optim as optim

# Define the XOR input and output
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Define the neural network
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Create the model, loss function, and optimizer
model = XORNet()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    output = model(X)
    loss = loss_function(output, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the parameters of our network

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    test_output = model(X)
    predictions = (test_output >= 0.5).float()
    print("\nFinal Predictions:")
    for i in range(len(X)):
        print(
            f"Input: {X[i].numpy()}, Predicted: {predictions[i].item():.0f}, Actual: {y[i].item():.0f}"
        )
