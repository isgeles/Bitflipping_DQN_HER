import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


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
        self.reset_parameters()

    def forward(self, state):
        """
        Network that maps state to action values.

        @param state: current state (2n bits)
        @return: value for all n bits we can flip
        """
        state = F.relu(self.fc1(state))
        action = self.fc2(state)
        return action

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
