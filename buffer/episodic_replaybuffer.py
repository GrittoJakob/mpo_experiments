import numpy as np
import torch
from collections import deque 



class EpisodicReplayBuffer:
    """
    Simple episodic replay buffer.

    - Stores full episodes
    - FIFO eviction happens on EPISODE level
    - Capacity is measured in number of timesteps
    - Sampling returns full episodes or n-steps (defined in the input)
    """

    def __init__(self, capacity, obs_dim, action_dim, buffer_device="cpu"):
        self.capacity = capacity

        if buffer_device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # FIFO store of full episodes
        self.episodes = deque()

        # Number of currently stored timesteps across all episodes
        self.total_steps = 0
     
    def get_episode_length(self, episode):
        return episode["obs"].shape[0]

    def add_episode_batch(self, obs, actions, next_obs, rewards, terminated, truncated):
        """
        Add one COMPLETE episode at once.
        Shape: [T, ...]
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        truncated = torch.as_tensor(truncated, dtype=torch.float32, device=self.device)

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)
        if truncated.ndim == 1:
            truncated = truncated.unsqueeze(-1)

        episode = {
            "obs": obs,
            "actions": actions,
            "next_obs": next_obs,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
        }

        ep_len = self.get_episode_length(episode)

        while self.total_steps + ep_len > self.capacity and len(self.episodes) > 0:
            old_ep = self.episodes.popleft()
            self.total_steps -= self.get_episode_length(old_ep)

        self.episodes.append(episode)
        self.total_steps += ep_len

    def sample_episodes_batch(self, batch_size):
        
        if len(self.episodes) == 0:
            raise ValueError("Cannot sample from an empty EpisodicReplayBuffer")

        idx = torch.randint(0, len(self.episodes), (batch_size,), device = self.device)
        sampled_episodes = [self.episodes[i] for i in idx.tolist()]

        batch = {
            "obs": [ep["obs"] for ep in sampled_episodes],
            "actions": [ep["actions"] for ep in sampled_episodes],
            "next_obs": [ep["next_obs"] for ep in sampled_episodes],
            "rewards": [ep["rewards"] for ep in sampled_episodes],
            "truncated": [ep["truncated"] for ep in sampled_episodes],
            "terminated": [ep["terminated"] for ep in sampled_episodes],
        }

        return batch


    def num_episodes(self):
        return len(self.episodes)

    def num_steps(self):
        return self.total_steps

    def mean_reward(self):
        if len(self.episodes) == 0:
            return 0.0

        all_rewards = torch.cat([ep["rewards"] for ep in self.episodes], dim=0)
        return all_rewards.mean().item()
    
    def sample_n_step_batch(self, batch_size, num_steps= 2):
      
      if len(self.episodes) == 0:
          raise ValueError("Cannot sample from an empty buffer")
      if num_steps <= 0:
          raise ValueError("num_steps must be >= 1")

      # Get all valid start indexes
      valid_starts = []
      episodes = list(self.episodes)   

      for ep_idx, ep in enumerate(episodes):
          T = ep["obs"].shape[0]
          for start in range(T - num_steps + 1):
              valid_starts.append((ep_idx, start))

      if len(valid_starts) == 0:
          raise ValueError(f"No episode is long enough for n={num_steps}")

      # sample
      sample_idx = torch.randint(0, len(valid_starts), (batch_size,))
      chosen = [valid_starts[i] for i in sample_idx.tolist()]

      obs = []
      actions = []
      next_obs = []
      rewards = []
      terminated = []
      truncated = []

      for ep_idx, start in chosen:
          ep = episodes[ep_idx]
          end = start + num_steps

          obs.append(ep["obs"][start:end])                 # [num_steps, obs_dim]
          actions.append(ep["actions"][start:end])         # [num_steps, action_dim]
          next_obs.append(ep["next_obs"][start:end])       # [num_steps, obs_dim]
          rewards.append(ep["rewards"][start:end])         # [num_steps, 1]
          terminated.append(ep["terminated"][start:end])   # [num_steps, 1]
          truncated.append(ep["truncated"][start:end])     # [num_steps, 1]

      batch = {
          "obs": torch.stack(obs, dim=0),                 # [B, num_steps, obs_dim]
          "actions": torch.stack(actions, dim=0),         # [B, num_steps, action_dim]
          "next_obs": torch.stack(next_obs, dim=0),       # [B, num_steps, obs_dim]
          "rewards": torch.stack(rewards, dim=0),         # [B, num_steps, 1]
          "terminated": torch.stack(terminated, dim=0),   # [B, num_steps, 1]
          "truncated": torch.stack(truncated, dim=0),     # [B, num_steps, 1]
      }

      return batch