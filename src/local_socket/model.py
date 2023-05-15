import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.set_seed()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def set_seed(self, Seed=100):    
        random.seed(Seed)    
        np.random.seed(Seed)    
        torch.manual_seed(Seed)    
        torch.cuda.manual_seed(Seed)    
        torch.cuda.manual_seed_all(Seed)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output