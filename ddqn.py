import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Define a fully connected layer to output four actions
        self.fc1 = nn.Linear(in_features=64 * 10 * 10, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Apply the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
