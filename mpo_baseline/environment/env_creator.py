import torch
from typing import Optional
import os
import gymnasium as gym
from Ant_Wrappers.task_wrapper import GoalPositionWrapper
from Ant_Wrappers.meta_task_wrapper import Meta_InvertedWrapper 
from Ant_Wrappers.ERFI_Wrappers import RAOActionWrapper, RFIActionWrapper, ERFIEvalActionWrapper


def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    # NumPy / BLAS threads (muss vor Imports passieren, aber auch so meist ok)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

def maybe_wrap_task(env, args, *, rank: Optional[int] = None, split_idx: Optional[int] = None, eval_env: Optional[bool]= False):
    """ Apply Task wrappers:
    task_mode: 
        -default
        -target_goal: random goals in hyperplane randomly sampled with task hints where goal is
        -inverted_without_task_hint: inverted goals without task hints 
    rand_mode:
        -default
        -RFI
        -RAO
        -ERFI
    """

    # TASK-MODE
    if getattr(args, "task_mode", "default") == "target_goal":
        env = GoalPositionWrapper(env, args)
    elif getattr(args, "task_mode", "default") == "inverted_without_task_hint":
        env = Meta_InvertedWrapper(env, args, args.history_len, args.append_task_reward)

    # RANDOMIZATION MODE
    if getattr(args, "rand_mode", "default") == "RFI":
        print("RFI is activated")
        env = RFIActionWrapper(env, args.noise_limit)

    elif getattr(args, "rand_mode", "default") == "RAO":
        print("RAO is activated")
        env = RAOActionWrapper(env, args.noise_limit)

    elif getattr(args, "rand_mode", "defaut") == "ERFI" and not eval_env:
        if rank is None or split_idx is None:
            raise ValueError("ERFI requires rank and split_idx")
        if rank < split_idx:
            env = RFIActionWrapper(env, args.noise_limit)
        else:
            env = RAOActionWrapper(env, args.noise_limit)

    elif eval_env and getattr(args, "rand_mode", "default") == "ERFI":
        env = ERFIEvalActionWrapper(
            env,
            rfi_noise=getattr(args, "noise_limit", 0.05),
            rao_noise=getattr(args, "noise_limit", 0.05),
            rng_seed=int(getattr(args, "seed", 0)) + 12345,
        )

    return env

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
    else:
        if render_mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, render_mode=render_mode)
    return env

def make_train_env(args, env_id, seed, *, rank: Optional[int] = None, split_idx: Optional[int] = None):
    env = _make_base_env(env_id, args, render_mode=None)
    if getattr(args, "env_id") == "Ant-v5":
        env = maybe_wrap_task(env, args, rank=rank, split_idx=split_idx)

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
        
    if getattr(args, "env_id") == "Ant-v5":
        env = maybe_wrap_task(env, args, eval_env = True)

    env.action_space.seed(seed_offset)
    env.observation_space.seed(seed_offset)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def make_video_env(args, run_name, name_prefix: str):
    return make_eval_env(args, args.env_id, args.seed, True, run_name, name_prefix)

def _train_env_thunk(args, env_id: str, base_seed: int, rank: int, split_idx: Optional[int]):
    def _thunk():
        seed = int(base_seed) + int(rank)
        return make_train_env(args, env_id, seed, rank=rank, split_idx=split_idx)
    return _thunk


def make_train_vec_env(
    args,
    env_id: str,
    seed: int,
    num_envs: int,
):
    """Main function for creating Async Vector Training envs"""
    assert num_envs >= 1

    # Compute split index for ERFI envs
    split_idx = None
    if getattr(args, "rand_mode", None) == "ERFI":
        ratio = float(getattr(args, "rand_split_ratio", 0.5))  
        ratio = max(0.0, min(1.0, ratio))
        split_idx = int(round(num_envs * ratio))  


    env_thunk = [
        _train_env_thunk(
            args, 
            env_id,
            seed,
            rank=i,
            split_idx=split_idx,
        )
        for i in range(num_envs)
    ]

    return gym.vector.AsyncVectorEnv(env_thunk)
