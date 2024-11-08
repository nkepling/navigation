import numpy as np
import torch
from torch.utils.data import Dataset

class ReplayBuffer(Dataset):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

    def add(self, reward_map, S1, S2, value, pi):
        # Pack the experience with all required elements
        experience = (reward_map, S1, S2, value, pi)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
            self.idx = (self.idx + 1) % self.max_size

    def __len__(self):
        # Return the current number of items in the buffer
        return len(self.buffer)

    def __getitem__(self, idx):
        # Retrieve experience at index idx
        reward_map, S1, S2, value, pi = self.buffer[idx]

        # Convert to torch tensors
        reward_map = torch.tensor(reward_map, dtype=torch.float32)
        S1 = torch.tensor(S1, dtype=torch.long)
        S2 = torch.tensor(S2, dtype=torch.long)
        value = torch.tensor(value, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)

        return reward_map, S1, S2, value, pi

