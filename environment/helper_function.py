import gymnasium as gym
import torch
from typing import Optional
from .Ant_Wrappers.task_wrapper import GoalPositionWrapper
from .Ant_Wrappers.meta_task_wrapper import Meta_InvertedWrapper 
from .Ant_Wrappers.ERFI_Wrappers import RAOActionWrapper, RFIActionWrapper, ERFIEvalActionWrapper
from gymnasium.wrappers import TransformObservation
import numpy as np
from gymnasium import spaces


# def stack_maze_observation(env):
#     obs_space = env.observation_space

#     if not isinstance(obs_space, spaces.Dict):
#         return env

#     required_keys = ("observation", "achieved_goal", "desired_goal")
#     if not all(k in obs_space.spaces for k in required_keys):
#         return env

#     obs_box = obs_space["observation"]
#     achieved_box = obs_space["achieved_goal"]
#     goal_box = obs_space["desired_goal"]

#     new_low = np.concatenate([
#         np.asarray(obs_box.low, dtype=np.float32).reshape(-1),
#         np.asarray(achieved_box.low, dtype=np.float32).reshape(-1),
#         np.asarray(goal_box.low, dtype=np.float32).reshape(-1),
#     ])

#     new_high = np.concatenate([
#         np.asarray(obs_box.high, dtype=np.float32).reshape(-1),
#         np.asarray(achieved_box.high, dtype=np.float32).reshape(-1),
#         np.asarray(goal_box.high, dtype=np.float32).reshape(-1),
#     ])

#     new_obs_space = spaces.Box(
#         low=new_low,
#         high=new_high,
#         dtype=np.float32,
#     )

#     def _stack_obs(obs):
#         return np.concatenate([
#             np.asarray(obs["observation"], dtype=np.float32).reshape(-1),
#             np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1),
#             np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1),
#         ])

#     return TransformObservation(
#         env,
#         _stack_obs,
#         observation_space=new_obs_space,
#     )



def wrap_task_for_robust_ant(
    env,
    args,
    *,
    rank: Optional[int] = None,
    split_idx: Optional[int] = None,
    eval_env: bool = False,
):
    """
    Apply task and randomization wrappers.

    task_mode:
        - "default"
        - "target_goal"
        - "inverted_without_task_hint"

    rand_mode:
        - "default"
        - "RFI"
        - "RAO"
        - "ERFI"
    """
    task_mode = getattr(args, "task_mode", "default")
    rand_mode = getattr(args, "rand_mode", "default")
    noise_limit = getattr(args, "noise_limit", 0.05)

  
    # Task wrappers
    if task_mode == "target_goal":
        env = GoalPositionWrapper(env, args)

    elif task_mode == "inverted_without_task_hint":
        env = Meta_InvertedWrapper(
            env,
            args,
            args.history_len,
            args.append_task_reward,
        )

    # Randomization wrappers
    if rand_mode == "RFI":
        print("RFI is activated")
        env = RFIActionWrapper(env, noise_limit)

    elif rand_mode == "RAO":
        print("RAO is activated")
        env = RAOActionWrapper(env, noise_limit)

    elif rand_mode == "ERFI":
        if eval_env:
            env = ERFIEvalActionWrapper(
                env,
                rfi_noise=noise_limit,
                rao_noise=noise_limit,
                rng_seed=int(getattr(args, "seed", 0)) + 12345,
            )
        else:
            if rank is None or split_idx is None:
                raise ValueError("ERFI requires both rank and split_idx for training environments.")

            if rank < split_idx:
                env = RFIActionWrapper(env, noise_limit)
            else:
                env = RAOActionWrapper(env, noise_limit)

    return env


class NegativeDistanceRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
        dg = np.asarray(obs["desired_goal"], dtype=np.float32)
        reward = -float(np.linalg.norm(ag - dg))

        return obs, reward, terminated, truncated, info