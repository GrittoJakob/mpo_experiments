import torch
import os
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

from .gc_wrappers.gc_obs_wrapper import make_maze_env
from .Ant_Wrappers.helpers_ant import make_ant_env 


"""
Env creator functions:
Envs are created seperately to wrap envs differently.

"""

def make_base_env(env_id: str, args, render_mode=None):
    
    # Ant env has own make_base_env function for seperately setting the rewards
    if "ant" in env_id.lower():
        env = make_ant_env(env_id, args, render_mode)
        return env

    # Wrap observation if env is goal-conditioned/maze_env
    if "maze" in env_id.lower():
        env = make_maze_env(env_id, args, render_mode)
        return env
    
    # For creating any other environment
    kwargs = {}
    # For video recording
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    
    # Create environment
    env = gym.make(env_id, **kwargs)
    return env


def make_train_single_env(args, env_id, seed):
    
    
    env = make_base_env(env_id, args, render_mode=None)
    env.reset(seed=seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    return env

def make_eval_env(args, env_id, seed, capture_video, run_name, name_prefix="rollout"):
    seed_offset = seed + 1000

    if capture_video:
        env = make_base_env(env_id, args, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep == 0,
        )
    else:
        env = make_base_env(env_id, args, render_mode=None)

    #env.reset(seed= seed_offset)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    return env


def make_train_vec_env(args, env_id: str, seed: int, num_envs: int):
    assert num_envs >= 1

    env_fns = [
        train_env_thunk(args, env_id, seed, i)
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

def train_env_thunk(args, env_id: str, base_seed: int, rank: int):
    def _thunk():
        seed = int(base_seed) + int(rank)
        return make_train_single_env(args, env_id, seed)
    return _thunk

def make_video_env(args, run_name, name_prefix: str):
    return make_eval_env(args, args.env_id, args.seed, True, run_name, name_prefix)


