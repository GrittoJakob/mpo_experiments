import torch
from typing import Optional
import os
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TransformObservation
gym.register_envs(gymnasium_robotics)
from .helper_function import wrap_task_for_robust_ant, stack_maze_observation


"""
Env creator functions:
Envs are created seperately to wrap envs differently.
That is necessary for ERFI!
If no ERFI is used, this whole structur is unnecessary, and one could use single gym.make(env_id, AsyncVectorEnv)

"""

def _make_base_env(env_id: str, args, render_mode: Optional[str] = None):
    """Central Builder, for equality"""
    if env_id == "Ant-v5":
        # For custom designed reward function
        kwargs = dict(
            ctrl_cost_weight=args.ctrl_cost_weight,
            healthy_reward=args.healthy_reward_weight,
            contact_cost_weight=args.contact_cost_weight,
            forward_reward_weight=args.forward_reward_weight,
            include_cfrc_ext_in_observation=args.include_cfrc_ext_in_observation,
        )
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(env_id, **kwargs)
    
    elif "maze" in env_id.lower():
        # Sparse or dense rewards in maze
        kwargs = dict(
            reward_type = args.reward
        )
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(env_id, **kwargs)

    else:
        if render_mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, render_mode=render_mode)
    return env


def make_train_single_env(args, env_id, seed, *, rank: Optional[int] = None, split_idx: Optional[int] = None):
    
    env = _make_base_env(env_id, args, render_mode=None)

    if env_id == "Ant-v5":
        env = wrap_task_for_robust_ant(env, args, rank=rank, split_idx=split_idx)
    
    if "maze" in env_id.lower():
        env = stack_maze_observation(env)
   
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def make_eval_env(args, env_id, seed, capture_video, run_name, name_prefix="rollout"):
    seed_offset = seed + 1000

    if capture_video:
        env = _make_base_env(env_id, args, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep == 0,
        )
    else:
        env = _make_base_env(env_id, args, render_mode=None)
        
    if args.env_id == "Ant-v5":
        env = wrap_task_for_robust_ant(env, args, eval_env = True)

    if "maze" in env_id.lower():
        env = stack_maze_observation(env)

    env.action_space.seed(seed_offset)
    env.observation_space.seed(seed_offset)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def make_train_vec_env(args, env_id: str, seed: int, num_envs: int):
    assert num_envs >= 1

    # Wrap envs differently when rand_mode == ERFI
    split_idx = None
    if getattr(args, "rand_mode", None) == "ERFI":
        ratio = float(getattr(args, "rand_split_ratio", 0.5))
        ratio = max(0.0, min(1.0, ratio))
        split_idx = int(round(num_envs * ratio))

    env_fns = [
        train_env_thunk(args, env_id, seed, rank=i, split_idx=split_idx)
        for i in range(num_envs)
    ]

    envs = gym.vector.AsyncVectorEnv(env_fns)

    print("Environments created:", envs)

    return envs


def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

def train_env_thunk(args, env_id: str, base_seed: int, rank: int, split_idx: Optional[int]):
    def _thunk():
        seed = int(base_seed) + int(rank)
        return make_train_single_env(args, env_id, seed, rank=rank, split_idx=split_idx)
    return _thunk

def make_video_env(args, run_name, name_prefix: str):
    return make_eval_env(args, args.env_id, args.seed, True, run_name, name_prefix)
