import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch

def make_maze_env(env_id, args, render_mode: None):
    
    kwargs = {}

    if "ant" in env_id.lower():
        kwargs = dict(
            include_cfrc_ext_in_observation = args.include_cfrc_ext_in_observation
        )
    
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **kwargs)
    env = GC_Obs_Wrapper(env)
    return env


class GC_Obs_Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_space = env.observation_space
        obs_dim = int(np.prod(obs_space["observation"].shape))
        achieved_goal = int(np.prod(obs_space["achieved_goal"].shape))
        desired_goal = int(np.prod(obs_space["desired_goal"].shape))

        self.obs_slice = slice(0, obs_dim)
        self.achieved_goal_slice = slice(obs_dim, obs_dim + achieved_goal)
        self.desired_goal_slice = slice(obs_dim + achieved_goal, obs_dim + achieved_goal + desired_goal)

        low = np.concatenate([
            obs_space["observation"].low.reshape(-1),
            obs_space["achieved_goal"].low.reshape(-1),
            obs_space["desired_goal"].low.reshape(-1),
        ]).astype(np.float32)

        high = np.concatenate([
            obs_space["observation"].high.reshape(-1),
            obs_space["achieved_goal"].high.reshape(-1),
            obs_space["desired_goal"].high.reshape(-1),
        ]).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([
            obs["observation"].reshape(-1),
            obs["achieved_goal"].reshape(-1),
            obs["desired_goal"].reshape(-1),
        ]).astype(np.float32)

    def get_achieved_goal(self, obs):
        return obs[..., self.achieved_goal_slice]

    def get_desired_goal(self, obs):
        return obs[..., self.desired_goal_slice]

    def replace_desired_goal(self, obs, new_goal):
        if isinstance(obs, torch.Tensor):
            obs = obs.clone()
        else:
            obs = obs.copy()
        obs[..., self.desired_goal_slice] = new_goal
        return obs