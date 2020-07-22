import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Deep Q-Network model architecture.

        @param state_size: dimension of the input layer, for n bits -> 2n (n current + n target bits)
        @param action_size: dimension of output layer = n, value for every bit where we choose the maximum
        """
        super(QNetwork, self).__init__()
        # fully connected layers, 1 hidden with 256 neurons
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        """
        network that maps state to action values.

        @param state: current state (2n bits)
        @return: value for all n bits we can flip
        """
        state = F.relu(self.fc1(state))
        action = self.fc2(state)
        return action

