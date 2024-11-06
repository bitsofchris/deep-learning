import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 1 input channel, 32 output channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),  # 32 input channels, 64 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),  # 64 * 12 * 12 input neurons
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 output neurons
        )

    def forward(self, x):
        logits = self.conv_stack(x)
        return logits


model = ConvolutionalNeuralNetwork().to(device)
print(model)