import numpy as np
import torch


class ReplayBuffer:
    """
    Single step replay buffer with a flat ring-buffer on GPU OR CPU (both compatible).

    Stores transitions:
      (state, action, next_state, reward, terminated, truncated)

    Key points:
    - Stores all transitions
    - Sampling is uniform over stored transitions.
    """ 

    def __init__(self, capacity, obs_dim, action_dim, buffer_device: str = "cpu"):
        
        # Capacity = Maximal Number of steps stored in buffer
        self.capacity = capacity

        # Branch for storing buffer on GPU or CPU
        if buffer_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else: 
            self.device = "cpu"

        # Allocate Memory on Device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device = self.device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device = self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device = self.device)
        self.terminated = torch.zeros((capacity, 1), dtype=torch.float32, device = self.device)
        self.truncated = torch.zeros((capacity, 1), dtype=torch.float32, device = self.device)

        # Ring Pointer
        self.pointer = 0
        self.size = 0

    def add_batch(self, obs, actions, next_obs, rewards, terminated, truncated):

        # Convert steps from numpy to tensor
        obs = torch.as_tensor(obs, dtype = torch.float32, device = self.device)
        actions = torch.as_tensor(actions, dtype = torch.float32, device = self.device)
        next_obs = torch.as_tensor(next_obs, dtype = torch.float32, device = self.device)
        rewards = torch.as_tensor(rewards, dtype = torch.float32, device = self.device)
        truncated= torch.as_tensor(truncated, dtype = torch.float32, device = self.device)
        terminated = torch.as_tensor(terminated, dtype = torch.float32, device = self.device)

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if truncated.ndim == 1:
            truncated = truncated.unsqueeze(-1)
        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)

        # Check dimensions
        batch_size = obs.shape[0]
        assert actions.shape[0] == batch_size
        assert next_obs.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert truncated.shape[0] == batch_size
        assert terminated.shape[0] == batch_size

        # Compute idx range (pointer + batch_size)
        # If self.pointer > self.capacity pointer starts at zero again 
        # FIFO logic automatically included
        idx = torch.arange(self.pointer, self.pointer + batch_size, device=self.device) % self.capacity

        # Store batch in buffer at positions from idx
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.next_obs[idx] = next_obs
        self.rewards[idx] = rewards
        self.truncated[idx] = truncated
        self.terminated[idx] = terminated

        self.pointer = (self.pointer + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample_batch(self, batch_size):

        if self.size == 0:
            raise ValueError("Can not sample from an empty ReplayBuffer")
        
        idx = torch.randint(0, self.size, (batch_size,), device = self.device)

        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "next_obs": self.next_obs[idx],
            "rewards": self.rewards[idx],
            "truncated": self.truncated[idx],
            "terminated": self.terminated[idx]
        }

        return batch
    
    # Statistics
    def __len__(self):
        return self.size
    
    def mean_reward(self):
        if self.size == 0:
            return 0.0
        return self.rewards[:self.size].mean()