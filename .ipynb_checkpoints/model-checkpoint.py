import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Model architecture.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # fully connected layers, 1 hidden with 256 neurons
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, action_size)
        

    def forward(self, state):
        """network that maps state to action values."""
        
        state = F.relu(self.fc1(state))
        action = self.fc2(state)
        return action
