import numpy as np
import torch
from collections import deque 


class GoalConditioned_EpisodicReplayBuffer:
    """
    Simple episodic replay buffer.

    - Stores full episodes
    - FIFO eviction happens on EPISODE level
    - Capacity is measured in number of timesteps
    - Sampling returns full episodes or n-steps (defined in the input)
    """

    def __init__(self, capacity, obs_wrapper, obs_dim, action_dim, buffer_device="cpu"):
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

        self.obs_wrapper = obs_wrapper

    def add_batch(self, obs, actions, next_obs, rewards, terminated, truncated):
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

    def get_random_idx(self, batch_size):

        if len(self.episodes) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Get all valid start indexes
        valid_starts = []
        episodes = list(self.episodes)   

        for ep_idx, ep in enumerate(episodes):
            T = ep["obs"].shape[0]
            if T>= 3:
                for start_idx in range(T - 2):
                    valid_starts.append((ep_idx, start_idx))

        if len(valid_starts) == 0:
            raise ValueError(f"No episode is long enough")
        
        # Sample random idx
        sample_idx = torch.randint(0, len(valid_starts), (batch_size,))
        start_idx = [valid_starts[i] for i in sample_idx.tolist()]
        return start_idx

    def sample_batch(self, batch_size):
        
        # Sample start idx
        start_idx_list = self.get_random_idx(batch_size)

        # Initilialize list
        # Obs
        start_obs_list = []
        midpoint_obs_list = []    

        # Goals
        goal_list = []
        midpoint_goal_list = []
        original_goal_list = []

        offset_list = []
        midpoint_offset_list = []
        actions_list = []
        midpoint_actions_list=[]


        for ep_idx, start in start_idx_list:
            ep = self.episodes[ep_idx]
            
            # T = Length Episode
            T = ep["obs"].shape[0]
            
            # Sample goal idx and midpoint idx
            goal_idx = torch.randint(start + 1, T, ()).item()   # goal_idx in {start+1, ..., T-1}    
            midpoint_idx = torch.randint(start, goal_idx, ()).item()    # midpoint_idx in {start, ..., goal_idx-1}

            # Value offset
            value_offset = goal_idx - start
            value_midpoint_offset = midpoint_idx - start
            
            # Get new desired goal from achieved goal in endpoint 
            goal_obs = ep["obs"][goal_idx].clone()
            goal = self.obs_wrapper.get_achieved_goal(goal_obs)
            
            # Get Midpoint obs
            midpoint_obs = ep["obs"][midpoint_idx].clone()
            midpoint_achieved_goal = self.obs_wrapper.get_achieved_goal(midpoint_obs)        

            # Relabel start goal with sampled midpoint goal
            start_obs = ep["obs"][start].clone()
            original_goal = self.obs_wrapper.get_achieved_goal(start_obs)      

            # Get Actions of start idx and midpoint idx
            actions_list.append(ep["actions"][start])         # [action_dim]
            midpoint_actions_list.append(ep["actions"][midpoint_idx])   # [action_dim]
            
            # Append to list
            # Obs
            start_obs_list.append(start_obs)           # [obs_dim]
            midpoint_obs_list.append(midpoint_obs)          # [obs_dim]

            # Goals
            goal_list.append(goal)                                  # [goal_dim]
            midpoint_goal_list.append(midpoint_achieved_goal)       # [goal_dim]
            original_goal_list.append(original_goal)

            offset_list.append(value_offset)
            midpoint_offset_list.append(value_midpoint_offset)
            

        batch = {
            "start_obs": torch.stack(start_obs_list, dim=0),                                                # [B, obs_dim]
            "midpoint_obs": torch.stack(midpoint_obs_list, dim= 0),                                         # [B, obs_dim]
            "goal": torch.stack(goal_list, dim = 0),                                                        # [B, goal_dim]
            "midpoint_goal":torch.stack(midpoint_goal_list, dim = 0),                                       # [B, goal_dim]
            "original_goal":torch.stack(original_goal_list, dim = 0),                                       # [B, goal_dim]
            "actions": torch.stack(actions_list, dim=0),                                                    # [B, action_dim]
            "midpoint_actions": torch.stack(midpoint_actions_list, dim = 0),                                # [B, action_dim]
            "offset": torch.tensor(offset_list, dtype=torch.long, device=self.device),                      # [B, 1]     
            "midpoint_offset": torch.tensor(midpoint_offset_list, dtype=torch.long, device=self.device)     # [B, 1]
        }

        return batch


    def num_episodes(self):
        return len(self.episodes)

    def __len__(self):
        return self.total_steps

    def mean_reward(self):
        if len(self.episodes) == 0:
            return 0.0

        all_rewards = torch.cat([ep["rewards"] for ep in self.episodes], dim=0)
        return all_rewards.mean().item()
    
    def get_episode_length(self, episode):
        return episode["obs"].shape[0]