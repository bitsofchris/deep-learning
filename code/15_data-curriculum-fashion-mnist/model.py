import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet5 architecture adapted for Fashion-MNIST classification.
    
    Expects 32x32 input images (Fashion-MNIST resized from 28x28).
    
    Architecture:
    - Conv1: 1→6 channels, 5x5 kernel, ReLU, MaxPool 2x2 (32x32 → 14x14)
    - Conv2: 6→16 channels, 5x5 kernel, ReLU, MaxPool 2x2 (14x14 → 5x5)  
    - FC1: 400→120, ReLU (16*5*5 = 400 features)
    - FC2: 120→84, ReLU
    - FC3: 84→10 (Fashion-MNIST classes)
    """
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(400, 120)  # 16*5*5 = 400 after conv layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv1(x)  # [batch, 6, 12, 12]
        x = self.conv2(x)  # [batch, 16, 4, 4]
        
        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)  # [batch, 400]
        
        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before final classification layer.
        
        Useful for clustering and curriculum learning analysis.
        Returns features from fc2 layer (84-dimensional).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


def get_model(num_classes=10, device='cpu'):
    """Factory function to create and initialize LeNet5 model.
    
    Args:
        num_classes: Number of output classes (10 for Fashion-MNIST)
        device: Device to place model on ('cpu', 'cuda', 'mps')
        
    Returns:
        Initialized LeNet5 model on specified device
    """
    model = LeNet5(num_classes=num_classes)
    model = model.to(device)
    
    # Initialize weights using Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model


def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'