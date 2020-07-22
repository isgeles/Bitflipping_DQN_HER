import numpy as np
import torch
import random
from collections import namedtuple, deque

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Experience tuple:  (s||g, a, r, s'||g, done).
    """

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer

        @param action_size: (int) dimension of each action
        @param buffer_size: (int) maximum size of buffer
        @param batch_size: (int) size of each training batch
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state_goal", "action", "reward", "next_state_goal", "done"])

    def add(self, state_goal, action, reward, next_state_goal, done):
        """Add a new experience to memory"""
        e = self.experience(state_goal, action, reward, next_state_goal, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        state_goals = torch.from_numpy(np.vstack([e.state_goal for e in experiences if e is not None])).float().to(
            self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_state_goals = torch.from_numpy(np.vstack([e.next_state_goal for e in experiences if e is not None])
                                            ).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
                                 ).float().to(self.device)

        return (state_goals, actions, rewards, next_state_goals, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)

