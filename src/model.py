import torch
import torch.nn as nn
import torch.optim as optim
from pycaret.regression import *
import pandas as pd

class LinReg(nn.Module):
    def __init__(self, input_dim):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # More neurons
        self.fc2 = nn.Linear(128, 64)  # More layers
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.3)  # Increase dropout to prevent overfitting

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))  # Ensure non-negative predictions
