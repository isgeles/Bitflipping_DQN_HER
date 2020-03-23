import numpy as np
import torch
import random
from collections import namedtuple, deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
        Experience tuple:  (s||g, a, r, s'||g, done).
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_goal", "action", "reward", "next_state_goal", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state_goal, action, reward, next_state_goal, done):
        """Add a new experience to memory."""
        e = self.experience(state_goal, action, reward, next_state_goal, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        state_goals = torch.from_numpy(np.vstack([e.state_goal for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_state_goals = torch.from_numpy(np.vstack([e.next_state_goal for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (state_goals, actions, rewards, next_state_goals, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)