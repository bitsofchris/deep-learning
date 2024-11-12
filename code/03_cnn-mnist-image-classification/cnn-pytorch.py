import torch
from torch import nn
from torch import optim
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
            # initial in_channels are the number of color channels in our input data
            # grayscale = 1, RGB = 3. Our data using 1
            # number of feature maps = out_channels
            # kernel_size is the size of the local receptive field as a  square (5x5)
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),  # 1 input channel, 32 output channels
            nn.ReLU(), # applies Rectified Linear Unit activation function to output of convlutional layer
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer, downsamples the feature maps reducing their size
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Max pooling layer
            # Flatten takes a tensor as input and returns a 1D tensor of the same size
            # takes tensor of (batch_size, channels, height, width) and returns tensor of (batch_size, channels * height * width)
            nn.Flatten(),
            # 40 feature maps * the 4x4 size of each feature map
            # the 28x28 image becaomes a 24x24 feature map after the first convolutional layer because of the 5x5 window
            # the 24x24 feature map becomes a 12x12 feature map after the first max pooling layer (2x2 window)
            # the 12x12 feature map becomes a 8x8 feature map after the second convolutional layer because of the 5x5 window
            # the 8x8 feature map becomes a 4x4 feature map after the second max pooling layer (2x2 window)
            # input to the fully connected layer is the flattened tensor of size 40 * 4 * 4
            nn.Linear(in_features=40 * 4 * 4, out_features=100),  # 64 * 12 * 12 input neurons
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.softmax(x) # apply softmax to the output of the network
        return x


model = ConvolutionalNeuralNetwork().to(device)


# Define the loss function and optimizer
criterion = nn.NLLLoss() # Negative Log Likelihood Loss
# weight_decay is our regularization term (penalizes large weights)
optimizer = optim.SGD(model.parameters(), lr=0.03)


# Train
# Train Loop in general
# forward pass to compute the output
# compute the loss
# backward pass to compute the gradients
# update the weights
# Define the training function
def train(dataloader, model, criterion, optimizer, device):
    model.train() # Set Pytorch to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"Train loss: {running_loss / 100:.3f}")
            running_loss = 0.0


# Evaluate
# forward pass to compute the output
# compute the loss? Why?
# compute the accuracy
# Define the evaluation function
def evaluate(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation loss: {running_loss / len(dataloader):.3f}")
    print(f"Accuracy: {100 * correct / total:.2f}%")



# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(train_dataloader, model, criterion, optimizer, device)
    evaluate(test_dataloader, model, criterion, device)

print("Finished Training")

# Save the model
# torch.save(model.state_dict(), "cnn_mnist.pth")